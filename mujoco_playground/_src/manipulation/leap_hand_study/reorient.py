"""Reorient task for leap hand."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
from etils import epath

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
# from mujoco_playground._src.manipulation.leap_hand_study import base as leap_hand_base
from mujoco_playground._src.manipulation.leap_hand_study import leap_hand_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.01,
      action_scale=0.5,
      action_repeat=1,
      ema_alpha=1.0,
      episode_length=1000,
      success_threshold=0.1,
      history_len=1,
      obs_noise=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
              cube_pos=0.02,
              cube_ori=0.1,
          ),
          random_ori_injection_prob=0.0,
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              orientation=5.0,
              position=0.5,
              termination=-100.0,
              hand_pose=-0.5,
              action_rate=-0.001,
              joint_vel=0.0,
              energy=-1e-3,
          ),
          success_reward=100.0,
      ),
      pert_config=config_dict.create(
          enable=False,
          linear_velocity_pert=[0.0, 3.0],
          angular_velocity_pert=[0.0, 0.5],
          pert_duration_steps=[1, 100],
          pert_wait_steps=[60, 150],
      ),
      impl="jax",
      naconmax=30 * 8192,
      njmax=160,
  )


def get_assets() -> Dict[str, bytes]:
  assets = {}
  path = mjx_env.MENAGERIE_PATH / "leap_hand"
  mjx_env.update_assets(assets, path / "assets")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(
      assets, consts.ROOT_PATH / "xmls" / "reorientation_cube_textures"
  )
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "meshes")
  return assets


def uniform_quat(rng: jax.Array) -> jax.Array:
  """Generate a random quaternion from a uniform distribution."""
  u, v, w = jax.random.uniform(rng, (3,))
  return jp.array([
      jp.sqrt(1 - u) * jp.sin(2 * jp.pi * v),
      jp.sqrt(1 - u) * jp.cos(2 * jp.pi * v),
      jp.sqrt(u) * jp.sin(2 * jp.pi * w),
      jp.sqrt(u) * jp.cos(2 * jp.pi * w),
  ])


class CubeReorient(mjx_env.MjxEnv):

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    self._model_assets = get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mj_model.opt.ccd_iterations = 10

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path
    self._post_init()

  def _post_init(self) -> None:
    home_key = self._mj_model.keyframe("home")
    self._init_q = jp.array(home_key.qpos, dtype=float)
    self._init_mpos = jp.array(home_key.mpos, dtype=float)
    self._init_mquat = jp.array(home_key.mquat, dtype=float)
    self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
    self._uppers = self._mj_model.actuator_ctrlrange[:, 1]
    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
    self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._cube_geom_id = self._mj_model.geom("cube").id
    self._cube_body_id = self._mj_model.body("cube").id
    self._cube_mass = self._mj_model.body_subtreemass[self._cube_body_id]
    self._default_pose = self._init_q[self._hand_qids]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Randomize the goal orientation.
    rng, goal_rng = jax.random.split(rng)
    goal_quat = uniform_quat(goal_rng)

    # Randomize the hand pose.
    rng, pos_rng, vel_rng = jax.random.split(rng, 3)
    q_hand = jp.clip(
        self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
        self._lowers,
        self._uppers,
    )
    v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

    # Randomize the cube pose.
    rng, p_rng, quat_rng = jax.random.split(rng, 3)
    start_pos = jp.array([0.1, 0.0, 0.05]) + jax.random.uniform(
        p_rng, (3,), minval=-0.01, maxval=0.01
    )
    start_quat = uniform_quat(quat_rng)
    q_cube = jp.array([*start_pos, *start_quat])
    v_cube = jp.zeros(6)

    # Initialize the state.
    qpos = jp.concatenate([q_hand, q_cube])
    qvel = jp.concatenate([v_hand, v_cube])
    data = mjx_env.make_data(
        self._mj_model,
        qpos=qpos,
        ctrl=q_hand,
        qvel=qvel,
        mocap_pos=self._init_mpos,
        mocap_quat=goal_quat,
        impl=self._mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )

    # Randomize the perturbation.
    rng, pert1, pert2, pert3 = jax.random.split(rng, 4)
    pert_wait_steps = jax.random.randint(
        pert1,
        (1,),
        minval=self._config.pert_config.pert_wait_steps[0],
        maxval=self._config.pert_config.pert_wait_steps[1],
    )
    pert_duration_steps = jax.random.randint(
        pert2,
        (1,),
        minval=self._config.pert_config.pert_duration_steps[0],
        maxval=self._config.pert_config.pert_duration_steps[1],
    )
    pert_lin = jax.random.uniform(
        pert3,
        minval=self._config.pert_config.linear_velocity_pert[0],
        maxval=self._config.pert_config.linear_velocity_pert[1],
    )
    pert_ang = jax.random.uniform(
        pert3,
        minval=self._config.pert_config.angular_velocity_pert[0],
        maxval=self._config.pert_config.angular_velocity_pert[1],
    )
    pert_velocity = jp.array([pert_lin] * 3 + [pert_ang] * 3)

    # Update the internal monitoring information
    info = {
        "rng": rng,
        "step": 0,
        "steps_since_last_success": 0,
        "success_count": 0,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "position_targets": data.ctrl,
        "qpos_error_history": jp.zeros(self._config.history_len * 16),
        "cube_pos_error_history": jp.zeros(self._config.history_len * 3),
        "cube_ori_error_history": jp.zeros(self._config.history_len * 6),
        "goal_quat_dquat": jp.zeros(3),
        # Perturbation.
        "pert_wait_steps": pert_wait_steps,
        "pert_duration_steps": pert_duration_steps,
        "pert_vel": pert_velocity,
        "pert_dir": jp.zeros(6, dtype=float),
        "last_pert_step": jp.array([-jp.inf], dtype=float),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["reward/success"] = jp.zeros((), dtype=float)
    metrics["steps_since_last_success"] = 0
    metrics["success_count"] = 0

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # add perturbation
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state, state.info["rng"])

    # compute position targets
    delta = action * self._config.action_scale
    position_targets = state.data.ctrl + delta
    position_targets = jp.clip(position_targets, self._lowers, self._uppers)
    position_targets = (
        self._config.ema_alpha * position_targets
        + (1 - self._config.ema_alpha) * state.info["position_targets"]
    )

    # apply control and step the physics
    data = mjx_env.step(
        self.mjx_model, state.data, position_targets, self.n_substeps
    )
    state.info["position_targets"] = position_targets

    ori_error = self._cube_orientation_error(data)
    success = ori_error < self._config.success_threshold
    state.info["steps_since_last_success"] = jp.where(
        success, 0, state.info["steps_since_last_success"] + 1
    )
    state.info["success_count"] = jp.where(
        success, state.info["success_count"] + 1, state.info["success_count"]
    )
    state.metrics["steps_since_last_success"] = state.info[
        "steps_since_last_success"
    ]
    state.metrics["success_count"] = state.info["success_count"]

    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  # tools function ------------------------------------------------------------

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> mjx_env.Observation:
    # Hand joint angles.
    joint_angles = data.qpos[self._hand_qids]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.obs_noise.level
        * self._config.obs_noise.scales.joint_pos
    )

    # Joint position error history.
    qpos_error_history = (
        jp.roll(info["qpos_error_history"], 16)
        .at[:16]
        .set(noisy_joint_angles - info["motor_targets"])
    )
    info["qpos_error_history"] = qpos_error_history

    def _get_cube_pose(data: mjx.Data) -> jax.Array:
      """Returns (potentially) noisy cube pose (xyz,wxyz)."""
      cube_pos = self.get_cube_position(data)
      cube_quat = self.get_cube_orientation(data)
      info["rng"], pos_rng, ori_rng = jax.random.split(info["rng"], 3)
      noisy_cube_quat = mjx._src.math.normalize(
          cube_quat
          + jax.random.normal(ori_rng, shape=(4,))
          * self._config.obs_noise.level
          * self._config.obs_noise.scales.cube_ori
      )
      noisy_cube_pos = (
          cube_pos
          + (2 * jax.random.uniform(pos_rng, shape=cube_pos.shape) - 1)
          * self._config.obs_noise.level
          * self._config.obs_noise.scales.cube_pos
      )
      return jp.concatenate([noisy_cube_pos, noisy_cube_quat])

    # Noisy cube pose.
    noisy_pose = _get_cube_pose(data)
    info["rng"], key1, key2, key3 = jax.random.split(info["rng"], 4)
    rand_quat = uniform_quat(key1)
    rand_pos = jax.random.uniform(key2, (3,), minval=-0.5, maxval=0.5)
    rand_pose = jp.concatenate([rand_pos, rand_quat])
    m = self._config.obs_noise.level * jax.random.bernoulli(
        key3, self._config.obs_noise.random_ori_injection_prob
    )
    noisy_pose = noisy_pose * (1 - m) + rand_pose * m

    # Cube position error history.
    palm_pos = self.get_palm_position(data)
    cube_pos_error = palm_pos - noisy_pose[:3]
    cube_pos_error_history = (
        jp.roll(info["cube_pos_error_history"], 3).at[:3].set(cube_pos_error)
    )
    info["cube_pos_error_history"] = cube_pos_error_history

    # Cube orientation error history.
    goal_quat = self.get_cube_goal_orientation(data)
    quat_diff = mjx._src.math.quat_mul(
        noisy_pose[3:], mjx._src.math.quat_inv(goal_quat)
    )
    xmat_diff = mjx._src.math.quat_to_mat(quat_diff).ravel()[3:]
    cube_ori_error_history = (
        jp.roll(info["cube_ori_error_history"], 6).at[:6].set(xmat_diff)
    )
    info["cube_ori_error_history"] = cube_ori_error_history

    # Uncorrupted cube pose for critic.
    cube_pos_error_uncorrupted = palm_pos - self.get_cube_position(data)
    cube_quat_uncorrupted = self.get_cube_orientation(data)
    quat_diff_uncorrupted = math.quat_mul(
        cube_quat_uncorrupted, math.quat_inv(goal_quat)
    )
    xmat_diff_uncorrupted = math.quat_to_mat(quat_diff_uncorrupted).ravel()[3:]

    # observations for the policy.
    state = jp.concatenate([
        noisy_joint_angles,  # 16
        qpos_error_history,  # 16 * history_len
        cube_pos_error_history,  # 3 * history_len
        cube_ori_error_history,  # 6 * history_len
        info["last_act"],  # 16
    ])

    # observations for the critic.
    privileged_state = jp.concatenate([
        state,
        data.qpos[self._hand_qids],
        data.qvel[self._hand_dqids],
        self.get_fingertip_positions(data),
        cube_pos_error_uncorrupted,
        xmat_diff_uncorrupted,
        self.get_cube_linvel(data),
        self.get_cube_angvel(data),
        info["pert_dir"],
        data.xfrc_applied[self._cube_body_id],
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _maybe_apply_perturbation(
      self, state: mjx_env.State, rng: jax.Array
  ) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      directory = jax.random.normal(rng, (6,))
      return directory / jp.linalg.norm(directory)

    def get_xfrc(
        state: mjx_env.State, pert_dir: jax.Array, i: jax.Array
    ) -> jax.Array:
      u_t = 0.5 * jp.sin(jp.pi * i / state.info["pert_duration_steps"])
      force = (
          u_t
          * self._cube_mass
          * state.info["pert_vel"]
          / (state.info["pert_duration_steps"] * self.dt)
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._cube_body_id].set(force * pert_dir)
      return xfrc_applied

    step, last_pert_step = state.info["step"], state.info["last_pert_step"]
    start_pert = jp.mod(step, state.info["pert_wait_steps"]) == 0
    start_pert &= step != 0  # No perturbation at the beginning of the episode.
    last_pert_step = jp.where(start_pert, step, last_pert_step)
    duration = jp.clip(step - last_pert_step, 0, 100_000)
    in_pert_interval = duration < state.info["pert_duration_steps"]

    pert_dir = jp.where(start_pert, gen_dir(rng), state.info["pert_dir"])
    xfrc = get_xfrc(state, pert_dir, duration) * in_pert_interval

    state.info["pert_dir"] = pert_dir
    state.info["last_pert_step"] = last_pert_step
    data = state.data.replace(xfrc_applied=xfrc)
    return state.replace(data=data)

  # Sensor readings. ---------------------------------------------------------

  #   def get_palm_position(self, data: mjx.Data) -> jax.Array:
  #     return mjx_env.get_sensor_data(self.mj_model, data, "palm_position")

  #   def get_cube_position(self, data: mjx.Data) -> jax.Array:
  #     return mjx_env.get_sensor_data(self.mj_model, data, "cube_position")

  #   def get_cube_orientation(self, data: mjx.Data) -> jax.Array:
  #     return mjx_env.get_sensor_data(self.mj_model, data, "cube_orientation")

  #   def get_cube_linvel(self, data: mjx.Data) -> jax.Array:
  #     return mjx_env.get_sensor_data(self.mj_model, data, "cube_linvel")

  #   def get_cube_angvel(self, data: mjx.Data) -> jax.Array:
  #     return mjx_env.get_sensor_data(self.mj_model, data, "cube_angvel")

  #   def get_cube_angacc(self, data: mjx.Data) -> jax.Array:
  #     return mjx_env.get_sensor_data(self.mj_model, data, "cube_angacc")

  #   def get_cube_upvector(self, data: mjx.Data) -> jax.Array:
  #     return mjx_env.get_sensor_data(self.mj_model, data, "cube_upvector")

  #   def get_cube_goal_orientation(self, data: mjx.Data) -> jax.Array:
  #     return mjx_env.get_sensor_data(self.mj_model, data, "cube_goal_orientation")

  #   def get_cube_goal_upvector(self, data: mjx.Data) -> jax.Array:
  #     return mjx_env.get_sensor_data(self.mj_model, data, "cube_goal_upvector")

  #   def get_fingertip_positions(self, data: mjx.Data) -> jax.Array:
  #     """Get fingertip positions relative to the grasp site."""
  #     return jp.concatenate([
  #         mjx_env.get_sensor_data(self.mj_model, data, f"{name}_position")
  #         for name in consts.FINGERTIP_NAMES
  #     ])

  # Accessors. -----------------------------------------------------------------

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu


#   @property
#   def mj_model(self) -> mujoco.MjModel:
#     return self._mj_model

#   @property
#   def mjx_model(self) -> mjx.Model:
#     return self._mjx_model
