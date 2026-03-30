import os
import argparse
import functools
import numpy as np
import jax
import jax.numpy as jp
import tensorflow as tf
import tf2onnx
import onnxruntime as rt
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.checkpoint import load
from brax.training.acme import running_statistics
from mujoco_playground.config import locomotion_params
from mujoco_playground import locomotion

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_experiment_dir(base_dir):
    """Creates and returns a new expN directory path."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    counter = 1
    while True:
        exp_dir = os.path.join(base_dir, f"exp{counter}")
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
            return exp_dir
        counter += 1

# -----------------------------------------------------------------------------
# Model Definition
# -----------------------------------------------------------------------------

class MLP(tf.keras.Model):
    def __init__(self, layer_sizes, activation, mean_std):
        super().__init__()
        self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
        self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
        
        self.mlp_block = tf.keras.Sequential(name="MLP_0")
        for i, size in enumerate(layer_sizes):
            self.mlp_block.add(layers.Dense(
                size, 
                activation=activation if i < len(layer_sizes)-1 else None,
                name=f"hidden_{i}",
                kernel_initializer="lecun_uniform"
            ))

    def call(self, inputs):
        x = (inputs - self.mean) / self.std
        logits = self.mlp_block(x)
        loc, _ = tf.split(logits, 2, axis=-1)
        return tf.tanh(loc)

def transfer_weights(jax_params, tf_model):
    print("[INFO] Transferring weights from JAX to Keras...")
    for layer_name, layer_params in jax_params.items():
        try:
            tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
            if isinstance(tf_layer, tf.keras.layers.Dense):
                kernel = np.array(layer_params['kernel'])
                bias = np.array(layer_params['bias'])
                tf_layer.set_weights([kernel, bias])
                print(f"  - Transferred: {layer_name}")
        except ValueError:
            print(f"  - Skipped: {layer_name}")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert Brax policies to ONNX.")
    parser.add_argument("--ckpt", required=True, help="Path to the JAX checkpoint directory.")
    parser.add_argument("--env", default="Op3Joystick", help="Environment name.")
    parser.add_argument("--out_dir", default="../onnx_model/", help="Base output directory.")
    args = parser.parse_args()

    # Environment Setup
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # 1. Setup Folders
    exp_path = get_experiment_dir(args.out_dir)
    onnx_file_path = os.path.join(exp_path, "op3_policy.onnx")
    plot_path = os.path.join(exp_path, "verification_plot.png")
    print(f"[INFO] Experiment directory created at: {exp_path}")

    # 2. Load Env & Metadata
    print(f"[INFO] Loading environment: {args.env}")
    ppo_params = locomotion_params.brax_ppo_config(args.env)
    env_cfg = locomotion.get_default_config(args.env)
    env = locomotion.load(args.env, config=env_cfg)

    obs_size = env.observation_size
    act_size = env.action_size
    input_dim = obs_size["state"][0] if isinstance(obs_size, dict) else obs_size

    # 3. Load JAX Weights
    print(f"[INFO] Loading JAX parameters from: {args.ckpt}")
    params = load(args.ckpt)

    # 4. Initialize Keras Model
    mean_std = (tf.convert_to_tensor(params[0].mean, dtype=tf.float32), 
                tf.convert_to_tensor(params[0].std, dtype=tf.float32))

    tf_policy_network = MLP(
        layer_sizes=[128, 128, 128, 128, act_size * 2], 
        activation=tf.nn.swish, 
        mean_std=mean_std
    )
    _ = tf_policy_network(tf.zeros((1, input_dim))) 

    transfer_weights(params[1]['params'], tf_policy_network)

    # 5. Export to ONNX
    print(f"[INFO] Converting to ONNX...")
    spec = [tf.TensorSpec(shape=(1, input_dim), dtype=tf.float32, name="obs")]
    tf_policy_network.output_names = ['continuous_actions']
    
    tf2onnx.convert.from_keras(
        tf_policy_network, input_signature=spec, opset=11, output_path=onnx_file_path
    )
    print(f"[SUCCESS] ONNX model saved to: {onnx_file_path}")

    # 6. Verification & Plotting
    print("[INFO] Running numerical verification...")
    test_input_np = np.ones((1, input_dim), dtype=np.float32)

    # Predictions
    tf_pred = tf_policy_network(test_input_np).numpy()[0]
    
    m = rt.InferenceSession(onnx_file_path, providers=['CPUExecutionProvider'])
    onnx_pred = m.run(['continuous_actions'], {'obs': test_input_np})[0][0]

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory,
        preprocess_observations_fn=running_statistics.normalize,
    )
    ppo_network = network_factory(obs_size, act_size)
    inference_fn = ppo_networks.make_inference_fn(ppo_network)((params[0], params[1]), deterministic=True)
    
    jax_obs = {'state': jp.ones((input_dim,))} if isinstance(obs_size, dict) else jp.ones((input_dim,))
    jax_pred, _ = inference_fn(jax_obs, jax.random.PRNGKey(0))

    # Print first few for visual check
    print(f"  JAX Sample:  {np.array(jax_pred[:3])}")
    print(f"  ONNX Sample: {onnx_pred[:3]}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(onnx_pred, label='ONNX', linestyle='--', linewidth=2)
    plt.plot(jax_pred, label='JAX', alpha=0.6)
    plt.title(f"Action Comparison - {args.env}")
    plt.legend()
    plt.savefig(plot_path)
    print(f"[INFO] Plot saved to: {plot_path}")

    # Final Check
    np.testing.assert_allclose(jax_pred, onnx_pred, atol=1e-4)
    print("[SUCCESS] All outputs match within tolerance.")

if __name__ == "__main__":
    main()