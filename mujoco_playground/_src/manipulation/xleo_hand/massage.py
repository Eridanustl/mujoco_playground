"""Massage task for dual xleo hands."""

import pickle
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from etils import epath

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.xleo_hand import constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.002,
      sim_dt=0.002,
      action_scale=0.5,
      action_repeat=5,
      episode_length=1000,
      # PD gains for torque control (all motors).
      wrist_kp=100.0,
      wrist_kd=5.0,
      finger_kp=5.0,
      finger_kd=0.1,
      # Future target observation steps (in env steps).
      tar_obs_steps=[1, 2, 3],
      obs_noise=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              pose=0.5,
              vel=0.1,
              key_pos=0.25,
              action_rate=-0.001,
              energy=-1e-4,
          ),
          # Gaussian kernel sigma: r = exp(-err / (2 * sigma²)).
          pose_sigma=0.3,
          vel_sigma=2.0,
          key_pos_sigma=0.1,
      ),
      # Termination: max body cartesian position error (meters).
      pose_termination_dist=0.1,
      pert_config=config_dict.create(
          enable=False,
          # Force magnitude applied to fingertip bodies (N).
          force_pert=[0.0, 0.5],
          torque_pert=[0.0, 0.1],
          # Duration and wait between perturbations (in env steps).
          pert_duration_steps=[1, 50],
          pert_wait_steps=[50, 150],
      ),
      impl="jax",
      naconmax=30 * 8192,
      njmax=160,
  )


def get_assets() -> Dict[str, bytes]:
  assets = {}
  mjx_env.update_assets(
      assets, consts.ROOT_PATH / "models" / "ftl_meshes", "*.stl"
  )
  mjx_env.update_assets(
      assets, consts.ROOT_PATH / "models" / "ftl_meshes", "*.STL"
  )
  mjx_env.update_assets(assets, consts.ROOT_PATH / "models" / "xmls", "*.xml")
  # Load decimated convex meshes with 'convex_new/' key prefix to match XML paths.
  convex_dir = epath.Path(
      consts.ROOT_PATH / "models" / "ftl_meshes" / "convex_new"
  )
  for f in convex_dir.glob("*.stl"):
    assets[f"convex_new/{f.name}"] = f.read_bytes()
  return assets


# --- Rotation helpers (JAX) ------------------------------------------------


def _axis_angle_to_quat(axis: jax.Array, angle: jax.Array) -> jax.Array:
  """Convert axis-angle to quaternion [x, y, z, w].

  Args:
    axis: (N, 3) unit rotation axes.
    angle: (N,) rotation angles in radians.

  Returns:
    (N, 4) quaternions in [x, y, z, w] format.
  """
  half = angle[..., None] * 0.5  # (N, 1)
  xyz = axis * jp.sin(half)  # (N, 3)
  w = jp.cos(half)  # (N, 1)
  return jp.concatenate([xyz, w], axis=-1)


def _quat_rotate(q: jax.Array, v: jax.Array) -> jax.Array:
  """Rotate vector v by quaternion q. q is [x, y, z, w]."""
  q_v = q[..., :3]
  q_w = q[..., 3:]
  t = 2.0 * jp.cross(q_v, v)
  return v + q_w * t + jp.cross(q_v, t)


def _quat_to_tan_norm(q: jax.Array) -> jax.Array:
  """Convert quaternion [x, y, z, w] to 6D continuous rotation representation.

  Returns the rotated [1,0,0] (tangent) and [0,0,1] (normal) vectors,
  concatenated to give a 6D representation per quaternion.

  Args:
    q: (..., 4) quaternions.

  Returns:
    (..., 6) tan_norm vectors.
  """
  ref_tan = jp.zeros_like(q[..., :3]).at[..., 0].set(1.0)
  ref_norm = jp.zeros_like(q[..., :3]).at[..., 2].set(1.0)
  tan = _quat_rotate(q, ref_tan)
  norm = _quat_rotate(q, ref_norm)
  return jp.concatenate([tan, norm], axis=-1)


def _hinge_to_tan_norm(angles: jax.Array, axes: jax.Array) -> jax.Array:
  """Convert hinge joint angles to 6D rotation representation.

  Args:
    angles: (N,) joint angles.
    axes: (N, 3) per-joint rotation axes.

  Returns:
    (N, 6) tan_norm representation.
  """
  q = _axis_angle_to_quat(axes, angles)
  return _quat_to_tan_norm(q)


class Massage(mjx_env.MjxEnv):

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    self._xml_path = consts.SCENE_XML.as_posix()
    self._model_assets = get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(self._xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self._config.sim_dt

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model)
    self._post_init()

  def _post_init(self) -> None:
    self._joint_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
    self._joint_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
    # Joint IDs (for indexing jnt_range etc.) — distinct from qpos addresses.
    self._joint_ids = jp.array(
        [self._mj_model.joint(n).id for n in consts.JOINT_NAMES],
        dtype=jp.int32,
    )
    self._default_pose = jp.array(self._mj_model.qpos0[self._joint_qids])

    # Build per-joint kp/kd arrays for PD torque control.
    # Wrist: indices 0..5 (left) and 15..20 (right).
    # Fingers: indices 6..14 (left) and 21..29 (right).
    kp = np.zeros(consts.NU)
    kd = np.zeros(consts.NU)
    wrist_ids = list(range(0, 6)) + list(range(15, 21))
    finger_ids = list(range(6, 15)) + list(range(21, 30))
    kp[wrist_ids] = self._config.wrist_kp
    kd[wrist_ids] = self._config.wrist_kd
    kp[finger_ids] = self._config.finger_kp
    kd[finger_ids] = self._config.finger_kd
    self._kp = jp.array(kp)
    self._kd = jp.array(kd)

    # Classify joints: slide (scalar) vs hinge (quat → 6D encoding).
    # Indices are local to JOINT_NAMES (0..29).
    slide_ids = []
    hinge_ids = []
    hinge_axes = []
    for i, name in enumerate(consts.JOINT_NAMES):
      jid = self._mj_model.joint(name).id
      if self._mj_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
        slide_ids.append(i)
      else:
        hinge_ids.append(i)
        hinge_axes.append(self._mj_model.jnt_axis[jid].copy())
    self._slide_ids = jp.array(slide_ids, dtype=jp.int32)
    self._hinge_ids = jp.array(hinge_ids, dtype=jp.int32)
    self._hinge_axes = jp.array(np.array(hinge_axes))  # (N_hinge, 3)
    # Encoded dim: n_slide * 1 + n_hinge * 6
    self._encoded_pos_dim = len(slide_ids) + len(hinge_ids) * 6

    # Per-joint termination margin: smaller for slide joints (meters)
    # than for hinge joints (radians).
    jnt_margin = np.full(consts.NQ, 0.05)  # default: 0.05 rad for hinge
    jnt_margin[slide_ids] = 0.01  # 0.01 m for slide
    self._jnt_margin = jp.array(jnt_margin)

    # Load expert trajectory.
    traj_path = consts.DATA_PATH / "massage_data.pkl"
    with open(epath.Path(traj_path), "rb") as f:
      traj = pickle.load(f)
    self._traj_qpos = jp.array(traj["qpos"])  # (T, 30)
    self._traj_qvel = jp.array(traj["qvel"])  # (T, 30)
    self._traj_len = self._traj_qpos.shape[0]
    self._traj_data_freq = float(traj["data_freq"])
    self._traj_period = float(traj["duration"])
    self._traj_omega = 2.0 * jp.pi / self._traj_period

    # Pre-compute reference body positions for the entire trajectory via CPU FK.
    tracked_body_ids_np = np.array(
        [self._mj_model.body(n).id for n in consts.TRACKED_BODY_NAMES]
    )
    self._tracked_body_ids = jp.array(tracked_body_ids_np, dtype=jp.int32)
    traj_xpos = self._precompute_traj_body_pos(
        traj["qpos"], tracked_body_ids_np
    )
    self._traj_xpos = jp.array(traj_xpos)  # (T, N_bodies, 3)

    # Key bodies for key_pos reward (fingertips + wrists).
    key_body_ids_np = np.array(
        [self._mj_model.body(n).id for n in consts.KEY_BODY_NAMES]
    )
    self._key_body_ids = jp.array(key_body_ids_np, dtype=jp.int32)
    traj_key_xpos = self._precompute_traj_body_pos(
        traj["qpos"], key_body_ids_np
    )
    self._traj_key_xpos = jp.array(traj_key_xpos)  # (T, N_key, 3)

    # Fingertip body ids for perturbation forces.
    self._fingertip_body_ids = jp.array(
        [self._mj_model.body(n).id for n in consts.FINGERTIP_BODY_NAMES],
        dtype=jp.int32,
    )
    self._n_fingertips = len(consts.FINGERTIP_BODY_NAMES)

  def _precompute_traj_body_pos(
      self, qpos_data: np.ndarray, body_ids: np.ndarray
  ) -> np.ndarray:
    """Run MuJoCo CPU FK on each trajectory frame to get tracked body xpos.

    Args:
      qpos_data: (T, 30) expert joint positions.
      body_ids: (N_tracked,) body indices to extract.

    Returns:
      (T, N_tracked, 3) body positions for each frame.
    """
    mj_data = mujoco.MjData(self._mj_model)
    T = qpos_data.shape[0]
    n_bodies = len(body_ids)
    xpos_all = np.zeros((T, n_bodies, 3))
    for t in range(T):
      # Set joint positions from trajectory.
      mj_data.qpos[:] = self._mj_model.qpos0
      mj_data.qpos[self._joint_qids] = qpos_data[t]
      mj_data.qvel[:] = 0
      mujoco.mj_forward(self._mj_model, mj_data)
      xpos_all[t] = mj_data.xpos[body_ids]
    return xpos_all

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Random phase offset: start from a random point in the trajectory cycle.
    rng, phase_rng, pos_rng, vel_rng, kp_rng, kd_rng = jax.random.split(rng, 6)
    phase_offset = jax.random.uniform(
        phase_rng, minval=0.0, maxval=self._traj_period
    )

    # Get expert target at the random phase.
    target_qpos, target_qvel = self._get_traj_target(phase_offset)

    # Initialize qpos from expert target + small perturbation, clipped to joint range.
    jnt_range_low = jp.array(self._mj_model.jnt_range[self._joint_ids, 0])
    jnt_range_high = jp.array(self._mj_model.jnt_range[self._joint_ids, 1])
    qpos = jp.array(self._mj_model.qpos0)
    qpos = qpos.at[self._joint_qids].set(
        jp.clip(
            target_qpos + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
            jnt_range_low,
            jnt_range_high,
        )
    )
    qvel = jp.zeros(self._mj_model.nv)
    qvel = qvel.at[self._joint_dqids].set(
        target_qvel + 0.1 * jax.random.normal(vel_rng, (consts.NV,))
    )

    # All actuators are motors (torque control): zero torque at init.
    ctrl = jp.zeros(self.mjx_model.nu)

    data = mjx_env.make_data(
        self._mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        impl=self._config.impl,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )

    # Randomize PD gains: *U(0.8, 1.2) per joint.
    kp = self._kp * jax.random.uniform(
        kp_rng, shape=(consts.NU,), minval=0.8, maxval=1.2
    )
    kd = self._kd * jax.random.uniform(
        kd_rng, shape=(consts.NU,), minval=0.8, maxval=1.2
    )

    # Perturbation parameters.
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
    pert3, pert4 = jax.random.split(pert3)
    pert_force = jax.random.uniform(
        pert3,
        minval=self._config.pert_config.force_pert[0],
        maxval=self._config.pert_config.force_pert[1],
    )
    pert_torque = jax.random.uniform(
        pert4,
        minval=self._config.pert_config.torque_pert[0],
        maxval=self._config.pert_config.torque_pert[1],
    )
    # (6,): [force_x, force_y, force_z, torque_x, torque_y, torque_z]
    pert_magnitude = jp.array([pert_force] * 3 + [pert_torque] * 3)

    info = {
        "rng": rng,
        "step": 0,
        "phase_offset": phase_offset,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "kp": kp,
        "kd": kd,
        # Perturbation state.
        "pert_wait_steps": pert_wait_steps,
        "pert_duration_steps": pert_duration_steps,
        "pert_magnitude": pert_magnitude,
        "pert_dir": jp.zeros((self._n_fingertips, 6)),
        "last_pert_step": jp.array([-jp.inf]),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["tracking_pos_error"] = jp.zeros(())
    metrics["tracking_vel_error"] = jp.zeros(())
    metrics["max_body_pos_error"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # Apply perturbation forces to fingertips.
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)

    # Policy outputs target joint positions relative to default pose.
    position_targets = self._default_pose + action * self._config.action_scale

    # PD torque: tau = kp * (target - q) - kd * qvel
    joint_pos = state.data.qpos[self._joint_qids]
    joint_vel = state.data.qvel[self._joint_dqids]
    torque = (
        state.info["kp"] * (position_targets - joint_pos)
        - state.info["kd"] * joint_vel
    )

    # Step physics.
    data = mjx_env.step(self.mjx_model, state.data, torque, self.n_substeps)

    # Increment step BEFORE computing obs/reward/termination so that
    # sim_time = (step+1)*dt matches the post-step physics state.
    state.info["step"] += 1

    # Observations, termination, rewards.
    obs = self._get_obs(data, state.info)
    done = self._get_termination(data, state.info)
    rewards = self._get_reward(data, action, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt if rewards else jp.zeros(())

    # Compute tracking errors for monitoring.
    sim_time = state.info["step"] * self.dt + state.info["phase_offset"]
    target_qpos, target_qvel = self._get_traj_target(sim_time)
    joint_pos = data.qpos[self._joint_qids]
    joint_vel = data.qvel[self._joint_dqids]
    state.metrics["tracking_pos_error"] = jp.mean(
        jp.square(joint_pos - target_qpos)
    )
    state.metrics["tracking_vel_error"] = jp.mean(
        jp.square(joint_vel - target_qvel)
    )
    # Body cartesian position error for monitoring.
    ref_body_pos = self._get_traj_body_pos(sim_time)
    cur_body_pos = data.xpos[self._tracked_body_ids]
    body_dist_sq = jp.sum(jp.square(cur_body_pos - ref_body_pos), axis=-1)
    state.metrics["max_body_pos_error"] = jp.sqrt(jp.max(body_dist_sq))

    # Update info and metrics.
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    return state.replace(data=data, obs=obs, reward=reward, done=done)

  # Helper methods. -----------------------------------------------------------

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> mjx_env.Observation:
    # Current joint state.
    joint_pos = data.qpos[self._joint_qids]
    joint_vel = data.qvel[self._joint_dqids]
    joint_pos_rel = joint_pos - self._default_pose

    # Add noise to joint_pos_rel.
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_pos_rel = (
        joint_pos_rel
        + (2 * jax.random.uniform(noise_rng, shape=joint_pos_rel.shape) - 1)
        * self._config.obs_noise.level
        * self._config.obs_noise.scales.joint_pos
    )

    # Encode current joint positions: slide=scalar, hinge=quat→6D tan_norm.
    cur_pos_encoded = self._encode_joint_pos(noisy_joint_pos_rel)

    # Current simulation time.
    sim_time = info["step"] * self.dt + info["phase_offset"]

    # Future target observations (N steps ahead).
    tar_obs_list = []
    for step_offset in self._config.tar_obs_steps:
      tar_time = sim_time + step_offset * self.dt
      tar_qpos, tar_qvel = self._get_traj_target(tar_time)
      tar_pos_rel = tar_qpos - self._default_pose
      tar_pos_encoded = self._encode_joint_pos(tar_pos_rel)
      tar_obs_list.append(tar_pos_encoded)

    # Phase encoding.
    phase = jp.array([
        jp.sin(self._traj_omega * sim_time),
        jp.cos(self._traj_omega * sim_time),
    ])

    # State for policy (662-dim).
    state_obs = jp.concatenate([
        cur_pos_encoded,  # 150: slide(6) + hinge 6D(24×6=144)
        joint_vel,  # 30
        *tar_obs_list,  # 150 × 3 = 450: future target positions
        phase,  # 2: [sin(ωt), cos(ωt)]
        info["last_act"],  # 30
    ])

    # Current-frame target for critic error computation.
    target_qpos, target_qvel = self._get_traj_target(sim_time)
    qpos_error = joint_pos - target_qpos
    qvel_error = joint_vel - target_qvel

    # Privileged state for critic (782-dim).
    # Includes uncorrupted joint state + tracking errors.
    privileged_state = jp.concatenate([
        state_obs,  # 662
        joint_pos_rel,  # 30: true joint pos (no noise)
        joint_vel,  # 30: true joint vel
        qpos_error,  # 30: position tracking error (no noise)
        qvel_error,  # 30: velocity tracking error
    ])

    return {
        "state": state_obs,
        "privileged_state": privileged_state,
    }

  def _encode_joint_pos(self, joint_pos_rel: jax.Array) -> jax.Array:
    """Encode joint positions: slide as scalar, hinge as quat → 6D (tan_norm).
    [1] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, “On the Continuity of Rotation Representations in Neural Networks,” presented at the Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 5745–5753. Accessed: Mar. 16, 2026. [Online]. Available: https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html
    """
    slide_vals = joint_pos_rel[self._slide_ids]  # (N_slide,)
    hinge_angles = joint_pos_rel[self._hinge_ids]  # (N_hinge,)
    hinge_6d = _hinge_to_tan_norm(
        hinge_angles, self._hinge_axes
    )  # (N_hinge, 6)
    return jp.concatenate([slide_vals, hinge_6d.ravel()])

  def _interp_traj(self, time: jax.Array, traj_array: jax.Array) -> jax.Array:
    """Linearly interpolate a trajectory array at a given time.

    Args:
      time: scalar simulation time.
      traj_array: (T, ...) trajectory data sampled at self._traj_data_freq.

    Returns:
      Interpolated value with shape matching traj_array[0].
    """
    traj_time = jp.mod(time, self._traj_period)
    idx_f = traj_time * self._traj_data_freq
    i0 = jp.floor(idx_f).astype(jp.int32)
    i0 = jp.clip(i0, 0, self._traj_len - 2)
    i1 = i0 + 1
    alpha = idx_f - i0.astype(jp.float32)
    return (1.0 - alpha) * traj_array[i0] + alpha * traj_array[i1]

  def _get_traj_target(self, time: jax.Array):
    """Get expert qpos/qvel at a given time via linear interpolation."""
    target_qpos = self._interp_traj(time, self._traj_qpos)
    target_qvel = self._interp_traj(time, self._traj_qvel)
    return target_qpos, target_qvel

  def _get_traj_body_pos(self, time: jax.Array) -> jax.Array:
    """Get reference body positions at a given time via linear interpolation.

    Returns:
      (N_tracked, 3) body positions.
    """
    return self._interp_traj(time, self._traj_xpos)

  def _get_traj_key_pos(self, time: jax.Array) -> jax.Array:
    """Get reference key body positions at a given time via linear interpolation.

    Returns:
      (N_key, 3) key body positions.
    """
    return self._interp_traj(time, self._traj_key_xpos)

  def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    # 1. NaN detection in qpos/qvel.
    nan_in_qpos = jp.any(jp.isnan(data.qpos))
    nan_in_qvel = jp.any(jp.isnan(data.qvel))
    nan_fail = jp.logical_or(nan_in_qpos, nan_in_qvel)

    # 2. Body cartesian position error
    # Compare current body xpos with reference trajectory body xpos.
    sim_time = info["step"] * self.dt + info["phase_offset"]
    ref_body_pos = self._get_traj_body_pos(sim_time)  # (N_tracked, 3)
    cur_body_pos = data.xpos[self._tracked_body_ids]  # (N_tracked, 3)
    body_pos_diff = cur_body_pos - ref_body_pos
    body_pos_dist_sq = jp.sum(
        body_pos_diff * body_pos_diff, axis=-1
    )  # (N_tracked,)
    max_body_dist_sq = jp.max(body_pos_dist_sq)
    threshold_sq = self._config.pose_termination_dist**2
    pose_fail = max_body_dist_sq > threshold_sq

    # 3. Joint limit violation
    # Use different margins for slide (0.01m) and hinge (0.05rad) joints.
    joint_pos = data.qpos[self._joint_qids]
    jnt_range_low = self._mj_model.jnt_range[self._joint_ids, 0]
    jnt_range_high = self._mj_model.jnt_range[self._joint_ids, 1]
    below_limit = jp.any(joint_pos < jnt_range_low - self._jnt_margin)
    above_limit = jp.any(joint_pos > jnt_range_high + self._jnt_margin)
    joint_fail = jp.logical_or(below_limit, above_limit)

    # Don't terminate on the first step (allow initial settling)
    not_first_step = info["step"] > 0
    done = jp.logical_or(
        nan_fail,
        jp.logical_and(
            not_first_step,
            jp.logical_or(pose_fail, joint_fail),
        ),
    )
    return done.astype(jp.float32)

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    """Apply periodic sinusoidal force perturbation to fingertip bodies."""
    info = state.info
    step = info["step"]
    last_pert_step = info["last_pert_step"]

    # Check if a new perturbation should start.
    start_pert = jp.mod(step, info["pert_wait_steps"]) == 0
    start_pert &= step != 0  # No perturbation at step 0.
    last_pert_step = jp.where(start_pert, step, last_pert_step)
    duration = jp.clip(step - last_pert_step, 0, 100_000)
    in_pert = duration < info["pert_duration_steps"]

    # Generate random perturbation directions for each fingertip.
    info["rng"], dir_rng = jax.random.split(info["rng"])
    new_dirs = jax.random.normal(dir_rng, (self._n_fingertips, 6))
    new_dirs = new_dirs / (
        jp.linalg.norm(new_dirs, axis=-1, keepdims=True) + 1e-8
    )
    pert_dir = jp.where(start_pert, new_dirs, info["pert_dir"])

    # Sinusoidal envelope: 0.5 * sin(pi * t / T).
    u_t = 0.5 * jp.sin(jp.pi * duration / info["pert_duration_steps"])

    # Compute xfrc_applied: (nbody, 6).
    force_per_tip = u_t * info["pert_magnitude"] * pert_dir  # (N_tips, 6)
    xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
    xfrc_applied = xfrc_applied.at[self._fingertip_body_ids].set(
        force_per_tip * in_pert
    )

    # Update info.
    info["pert_dir"] = pert_dir
    info["last_pert_step"] = last_pert_step
    data = state.data.replace(xfrc_applied=xfrc_applied)
    return state.replace(data=data)

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
  ) -> dict[str, jax.Array]:
    sim_time = info["step"] * self.dt + info["phase_offset"]
    return {
        "pose": self._reward_pose(data, sim_time),
        "vel": self._reward_vel(data, sim_time),
        "key_pos": self._reward_key_pos(data, sim_time),
        "action_rate": self._reward_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._reward_energy(data),
    }

  # Reward functions. --------------------------------------------------------

  def _reward_pose(self, data: mjx.Data, sim_time: jax.Array) -> jax.Array:
    """Joint pose tracking in 6D encoding space: exp(-err / (2 * sigma²))."""
    target_qpos, _ = self._get_traj_target(sim_time)
    joint_pos = data.qpos[self._joint_qids]
    cur_encoded = self._encode_joint_pos(joint_pos - self._default_pose)
    tar_encoded = self._encode_joint_pos(target_qpos - self._default_pose)
    pose_err = jp.mean(jp.square(cur_encoded - tar_encoded))
    sigma = self._config.reward_config.pose_sigma
    return jp.exp(-pose_err / (2.0 * sigma**2))

  def _reward_vel(self, data: mjx.Data, sim_time: jax.Array) -> jax.Array:
    """Joint velocity tracking: exp(-err / (2 * sigma²))."""
    _, target_qvel = self._get_traj_target(sim_time)
    joint_vel = data.qvel[self._joint_dqids]
    vel_err = jp.mean(jp.square(joint_vel - target_qvel))
    sigma = self._config.reward_config.vel_sigma
    return jp.exp(-vel_err / (2.0 * sigma**2))

  def _reward_key_pos(self, data: mjx.Data, sim_time: jax.Array) -> jax.Array:
    """Key body cartesian position tracking: exp(-err / (2 * sigma²))."""
    ref_key_pos = self._get_traj_key_pos(sim_time)  # (N_key, 3)
    cur_key_pos = data.xpos[self._key_body_ids]  # (N_key, 3)
    key_pos_err = jp.mean(jp.sum(jp.square(cur_key_pos - ref_key_pos), axis=-1))
    sigma = self._config.reward_config.key_pos_sigma
    return jp.exp(-key_pos_err / (2.0 * sigma**2))

  def _reward_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    """Action smoothness penalty: first + second order differences."""
    c1 = jp.sum(jp.square(act - last_act))
    c2 = jp.sum(jp.square(act - 2 * last_act + last_last_act))
    return c1 + c2

  def _reward_energy(self, data: mjx.Data) -> jax.Array:
    """Energy consumption penalty: sum(|qvel * actuator_force|)."""
    joint_vel = data.qvel[self._joint_dqids]
    return jp.sum(jp.abs(joint_vel * data.actuator_force))

  # Accessors. -----------------------------------------------------------------

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Domain randomization for massage task."""
  mj_model = Massage().mj_model

  # Identify collision geom ids (contype == 1) for friction randomization.
  collision_geom_ids = [
      i for i in range(mj_model.ngeom) if mj_model.geom_contype[i] == 1
  ]

  # Hand body ids (all bodies except world=0 and human=last).
  hand_body_ids = np.array(
      [mj_model.body(n).id for n in consts.TRACKED_BODY_NAMES]
  )

  # Joint DOF ids.
  joint_dof_ids = mjx_env.get_qvel_ids(mj_model, consts.JOINT_NAMES)

  @jax.vmap
  def rand(rng):
    # 1. Contact friction: =U(0.3, 1.0) for all collision geoms.
    rng, key = jax.random.split(rng)
    friction_val = jax.random.uniform(key, (1,), minval=0.3, maxval=1.0)
    geom_friction = model.geom_friction.at[collision_geom_ids, 0].set(
        friction_val
    )

    # 2. Link mass: *U(0.9, 1.1) for each hand body independently.
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[hand_body_ids].set(
        model.body_mass[hand_body_ids] * dmass
    )

    # 3. Link center-of-mass offset: +U(-5e-3, 5e-3) per body per axis.
    rng, key = jax.random.split(rng)
    dpos = jax.random.uniform(
        key, (len(hand_body_ids), 3), minval=-5e-3, maxval=5e-3
    )
    body_ipos = model.body_ipos.at[hand_body_ids].set(
        model.body_ipos[hand_body_ids] + dpos
    )

    # 4. Joint friction loss: *U(0.5, 2.0).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[joint_dof_ids] * jax.random.uniform(
        key, shape=(consts.NV,), minval=0.5, maxval=2.0
    )
    dof_frictionloss = model.dof_frictionloss.at[joint_dof_ids].set(
        frictionloss
    )

    # 5. Joint armature: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[joint_dof_ids] * jax.random.uniform(
        key, shape=(consts.NV,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[joint_dof_ids].set(armature)

    # 6. Joint damping: *U(0.8, 1.2).
    rng, key = jax.random.split(rng)
    damping = model.dof_damping[joint_dof_ids] * jax.random.uniform(
        key, shape=(consts.NV,), minval=0.8, maxval=1.2
    )
    dof_damping = model.dof_damping.at[joint_dof_ids].set(damping)

    return (
        geom_friction,
        body_mass,
        body_ipos,
        dof_frictionloss,
        dof_armature,
        dof_damping,
    )

  (
      geom_friction,
      body_mass,
      body_ipos,
      dof_frictionloss,
      dof_armature,
      dof_damping,
  ) = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_mass": 0,
      "body_ipos": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "dof_damping": 0,
  })

  model = model.tree_replace({
      "geom_friction": geom_friction,
      "body_mass": body_mass,
      "body_ipos": body_ipos,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "dof_damping": dof_damping,
  })

  return model, in_axes
