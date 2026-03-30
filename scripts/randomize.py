# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Domain randomization for OP3 locomotion environments."""

import jax
from mujoco import mjx

_N_ACT = 20

_TORSO_BODY_ID = 1


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Randomise OP3 physics parameters across a batch of environments.

  Called once per episode reset by BraxDomainRandomizationVmapWrapper.

  Args:
    model: Base mjx.Model (un-batched).
    rng:   JAX PRNG key array of shape (num_envs, 2).

  Returns:
    (randomised_model, in_axes) where in_axes marks which fields are batched.
  """

  @jax.vmap
  def rand_dynamics(rng):
    # ------------------------------------------------------------------
    # 1. Floor geom friction: U(0.4, 1.0)
    #    geom[0] is the floor in the feetonly scene XML.
    # ------------------------------------------------------------------
    rng, key = jax.random.split(rng)
    floor_friction = jax.random.uniform(key, minval=0.4, maxval=1.0)
    geom_friction = model.geom_friction.at[0, 0].set(floor_friction)

    # ------------------------------------------------------------------
    # 2. Explicit pair friction (if npair > 0): U(0.4, 1.0)
    # ------------------------------------------------------------------
    rng, key = jax.random.split(rng)
    friction = jax.random.uniform(key, minval=0.4, maxval=1.0)
    pair_friction = model.pair_friction.at[0:2, 0:2].set(friction)

    # ------------------------------------------------------------------
    # 3. Joint friction loss (Dynamixel internal resistance): *U(0.5, 2.0)
    # ------------------------------------------------------------------
    rng, key = jax.random.split(rng)
    frictionloss_scale = jax.random.uniform(
        key, shape=(_N_ACT,), minval=0.5, maxval=2.0
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(
        model.dof_frictionloss[6:] * frictionloss_scale
    )

    # ------------------------------------------------------------------
    # 4. Rotor armature (reflected inertia): *U(1.0, 1.05)
    # ------------------------------------------------------------------
    rng, key = jax.random.split(rng)
    armature_scale = jax.random.uniform(
        key, shape=(_N_ACT,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[6:].set(
        model.dof_armature[6:] * armature_scale
    )

    # ------------------------------------------------------------------
    # 5. All link masses: *U(0.85, 1.15)
    #    OP3 total ~3 kg so proportional variation is significant.
    # ------------------------------------------------------------------
    rng, key = jax.random.split(rng)
    mass_scale = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.85, maxval=1.15
    )
    body_mass = model.body_mass * mass_scale

    # ------------------------------------------------------------------
    # 6. Torso payload: +U(-0.3, 0.3) kg
    # ------------------------------------------------------------------
    rng, key = jax.random.split(rng)
    payload = jax.random.uniform(key, minval=-0.3, maxval=0.3)
    body_mass = body_mass.at[_TORSO_BODY_ID].set(
        body_mass[_TORSO_BODY_ID] + payload
    )

    # ------------------------------------------------------------------
    # 7. PD stiffness (Kp): *U(0.85, 1.15)
    #    actuator_gainprm[:,0] is the position gain.
    #    actuator_biasprm[:,1] is the matching -Kp bias term.
    # ------------------------------------------------------------------
    rng, key = jax.random.split(rng)
    kp_scale = jax.random.uniform(
        key, shape=(model.nu,), minval=0.85, maxval=1.15
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(
        model.actuator_gainprm[:, 0] * kp_scale
    )
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(
        model.actuator_biasprm[:, 1] * kp_scale
    )

    # ------------------------------------------------------------------
    # 8. PD damping (Kd): *U(0.8, 1.2)
    # ------------------------------------------------------------------
    rng, key = jax.random.split(rng)
    kd_scale = jax.random.uniform(
        key, shape=(_N_ACT,), minval=0.8, maxval=1.2
    )
    dof_damping = model.dof_damping.at[6:].set(
        model.dof_damping[6:] * kd_scale
    )

    # ------------------------------------------------------------------
    # 9. Initial pose jitter: +U(-0.05, 0.05) rad
    # ------------------------------------------------------------------
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0.at[7:].set(
        model.qpos0[7:]
        + jax.random.uniform(key, shape=(_N_ACT,), minval=-0.05, maxval=0.05)
    )

    return (
        geom_friction,
        pair_friction,
        dof_frictionloss,
        dof_armature,
        body_mass,
        actuator_gainprm,
        actuator_biasprm,
        dof_damping,
        qpos0,
    )

  (
      geom_friction,
      pair_friction,
      dof_frictionloss,
      dof_armature,
      body_mass,
      actuator_gainprm,
      actuator_biasprm,
      dof_damping,
      qpos0,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "pair_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
      "dof_damping": 0,
      "qpos0": 0,
  })

  model = model.tree_replace({
      "geom_friction": geom_friction,
      "pair_friction": pair_friction,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "body_mass": body_mass,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
      "dof_damping": dof_damping,
      "qpos0": qpos0,
  })

  return model, in_axes