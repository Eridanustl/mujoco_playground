"""Massage task for dual xleo hands."""

import functools
import pickle
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from etils import epath

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.xleo_hand import constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      finger_action_scale=0.5,
      wrist_action_scale=0.05,
      action_repeat=1,
      episode_length=1000,
      # PD gains for torque control (all motors).
      wrist_kp=100.0,
      wrist_kd=10,
      wrist_rot_kp=10.0,
      wrist_rot_kd=1,
      finger_kp=5.0,
      finger_kd=0.1,
      # Future target observation steps (in env steps).
      target_obs_steps=[1, 2, 3],
      obs_noise=config_dict.create(
          level=1,
          scales=config_dict.create(
              joint_pos=0.01,
          ),
      ),
      reward_config=config_dict.create(
          # DeepMimic-style sub-reward weights (should sum to 1.0).
          scales=config_dict.create(
              pose=0.3,
              vel=0.1,
              root_pose=0.2,
              root_vel=0.1,
              key_pos=0.1,
              contact_force=0.2,
              # Regularization penalties (unchanged).
              action_rate=-1e-3,
              # action_smooth=-5e-3,
              # energy=-1e-6
          ),
          # DeepMimic exponential reward scales: r = exp(-scale * err).
          pose_scale=0.25,
          vel_scale=0.01,
          root_pose_scale=10.0,
          root_vel_scale=1.0,
          key_pos_scale=10.0,
          contact_force_scale=5.0,
          # Coefficient for rotation error within root_pose / root_vel.
          root_pose_rot_coeff=0.1,
          root_vel_rot_coeff=0.1,
          # Per-finger-joint error weights (18 = 9 left + 9 right).
          finger_err_w=[1.0] * 18,
      ),
      # Termination: max body cartesian position error (meters).
      pose_termination_dist=0.02,
      terminate_on_nan=True,
      terminate_on_pose=True,
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
      naconmax=30 * 16384,
      njmax=60,
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


@functools.lru_cache(maxsize=1)
def _domain_randomization_metadata():
  """Loads static ids used by domain randomization once."""
  mj_model = mujoco.MjModel.from_xml_string(
      epath.Path(consts.SCENE_XML).read_text(), assets=get_assets()
  )

  hand_body_ids = jp.array(
      [mj_model.body(n).id for n in consts.TRACKED_BODY_NAMES], dtype=jp.int32
  )
  joint_dof_ids = jp.array(
      mjx_env.get_qvel_ids(mj_model, consts.JOINT_NAMES), dtype=jp.int32
  )

  hand_body_id_set = set(int(body_id) for body_id in hand_body_ids.tolist())
  collision_geom_ids = jp.array(
      [
          geom_id
          for geom_id in range(mj_model.ngeom)
          if mj_model.geom_bodyid[geom_id] in hand_body_id_set
          and (
              mj_model.geom_contype[geom_id] != 0
              or mj_model.geom_conaffinity[geom_id] != 0
          )
      ],
      dtype=jp.int32,
  )

  return {
      "collision_geom_ids": collision_geom_ids,
      "hand_body_ids": hand_body_ids,
      "joint_dof_ids": joint_dof_ids,
  }


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
    self._wrist_joint_ids = jp.array(
        [self._mj_model.joint(n).id for n in consts.WRIST_NAMES],
        dtype=jp.int32,
    )
    self._finger_joint_ids = jp.array(
        [self._mj_model.joint(n).id for n in consts.FINGER_NAMES],
        dtype=jp.int32,
    )
    self._default_pose = jp.array(self._mj_model.qpos0[self._joint_qids])

    # Build per-joint kp/kd arrays for PD torque control.
    # Wrist slide (linear): wrist_kp / wrist_kd
    # Wrist hinge (rotation): wrist_rot_kp / wrist_rot_kd
    # Fingers: finger_kp / finger_kd
    wrist_slide_ids = jp.array(consts.WRIST_SLIDE_INDICES)
    wrist_hinge_ids = jp.array(consts.WRIST_HINGE_INDICES)
    finger_ids = jp.array(consts.FINGER_INDICES)
    kp = jp.zeros(consts.NU)
    kd = jp.zeros(consts.NU)
    kp = kp.at[wrist_slide_ids].set(self._config.wrist_kp)
    kd = kd.at[wrist_slide_ids].set(self._config.wrist_kd)
    kp = kp.at[wrist_hinge_ids].set(self._config.wrist_rot_kp)
    kd = kd.at[wrist_hinge_ids].set(self._config.wrist_rot_kd)
    kp = kp.at[finger_ids].set(self._config.finger_kp)
    kd = kd.at[finger_ids].set(self._config.finger_kd)
    self._kp = kp
    self._kd = kd

    # Load expert trajectory (already resampled to policy frequency).
    traj_path = consts.DATA_PATH / "massage_traj.pkl"
    with open(epath.Path(traj_path), "rb") as f:
      traj = pickle.load(f)
    traj_period = float(traj["duration"])
    self._traj_period = traj_period
    self._traj_omega = 2.0 * jp.pi / traj_period

    self._traj_qpos = jp.array(traj["qpos"])  # (T, 30)
    self._traj_qvel = jp.array(traj["qvel"])  # (T, 30)
    self._traj_len = self._traj_qpos.shape[0]

    # Body ids for cartesian position tracking.
    self._tracked_body_ids = jp.array(
        [self._mj_model.body(n).id for n in consts.TRACKED_BODY_NAMES],
        dtype=jp.int32,
    )
    self._key_body_ids = jp.array(
        [self._mj_model.body(n).id for n in consts.KEY_BODY_NAMES],
        dtype=jp.int32,
    )

    self._traj_xpos = jp.array(traj["tracked_body_xpos"])  # (T, 20, 3)
    self._traj_key_xpos = jp.array(traj["key_body_xpos"])  # (T, 8, 3)

    # Wrist body IDs for world-to-local coordinate transformation.
    self._l_wrist_body_id = self._mj_model.body("L_WRIST").id
    self._r_wrist_body_id = self._mj_model.body("R_WRIST").id

    # Index mapping within TRACKED_BODY_NAMES (20 bodies):
    #   LEFT_BODY_NAMES:  [0]=L_WRIST, [1..9]=left finger bodies
    #   RIGHT_BODY_NAMES: [10]=R_WRIST, [11..19]=right finger bodies
    n_left = len(consts.LEFT_BODY_NAMES)  # 10
    self._left_wrist_idx = 0
    self._right_wrist_idx = n_left  # 10
    self._left_finger_slice = slice(1, n_left)  # 1..9
    self._right_finger_slice = slice(
        n_left + 1, len(consts.TRACKED_BODY_NAMES)
    )  # 11..19

    # KEY_BODY_NAMES: first N are left fingertips, rest are right fingertips.
    self._n_left_key = len([n for n in consts.KEY_BODY_NAMES if "_L" in n])

    # Contact force reference trajectory (optional).
    if "tracked_contact_force" in traj:
      self._traj_contact_force = jp.array(
          traj["tracked_contact_force"]
      )  # (T, 8, 3)
    else:
      # Fallback: zeros if data doesn't contain tracked_contact_force.
      self._traj_contact_force = jp.zeros(
          (self._traj_len, len(consts.CONTACT_FORCE_BODY_NAMES), 3)
      )

    # Body IDs for cfrc_ext contact force reading (privileged obs + reward).
    self._contact_body_ids = jp.array(
        [self._mj_model.body(n).id for n in consts.CONTACT_FORCE_BODY_NAMES],
        dtype=jp.int32,
    )

    # Sensor addresses for contact force tracking (force sensors in sensordata).
    # Pre-compute flat indices for all 6×3=18 sensordata entries so we can
    # gather them in one static-index operation inside JIT.
    sensor_adrs = [
        self._mj_model.sensor_adr[self._mj_model.sensor(n).id]
        for n in consts.CONTACT_FORCE_SENSOR_NAMES
    ]
    # Flat index array: [adr0, adr0+1, adr0+2, adr1, adr1+1, ...] shape (18,)
    self._contact_force_sensor_indices = jp.array(
        [adr + d for adr in sensor_adrs for d in range(3)],
        dtype=jp.int32,
    )

    # Fingertip body ids for perturbation forces.
    self._fingertip_body_ids = jp.array(
        [self._mj_model.body(n).id for n in consts.FINGERTIP_BODY_NAMES],
        dtype=jp.int32,
    )
    self._n_fingertips = len(consts.FINGERTIP_BODY_NAMES)

    # Per-finger-joint error weights for DeepMimic pose reward.
    self._finger_err_w = jp.array(
        self._config.reward_config.finger_err_w, dtype=jp.float32
    )

    # Actuator torque limits from ctrlrange.
    self._torque_low = jp.array(self._mj_model.actuator_ctrlrange[:, 0])
    self._torque_high = jp.array(self._mj_model.actuator_ctrlrange[:, 1])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Random step offset: start from a random point in the trajectory cycle.
    rng, phase_rng, pos_rng, vel_rng, kp_rng, kd_rng = jax.random.split(rng, 6)
    step_offset = jax.random.randint(
        phase_rng, (), minval=0, maxval=self._traj_len
    )

    # Get expert target at the random step.
    traj_idx = step_offset % self._traj_len
    target_qpos = self._traj_qpos[traj_idx]
    target_qvel = self._traj_qvel[traj_idx]

    # Initialize qpos from expert target + small perturbation, clipped to joint range.
    jnt_range_low = jp.array(self._mj_model.jnt_range[self._joint_ids, 0])
    jnt_range_high = jp.array(self._mj_model.jnt_range[self._joint_ids, 1])
    qpos = jp.array(self._mj_model.qpos0)
    qpos = qpos.at[self._joint_qids].set(
        jp.clip(
            target_qpos + 0.01 * jax.random.normal(pos_rng, (consts.NQ,)),
            jnt_range_low,
            jnt_range_high,
        )
    )
    qvel = jp.zeros(self._mj_model.nv)
    qvel = qvel.at[self._joint_dqids].set(
        target_qvel + 0.01 * jax.random.normal(vel_rng, (consts.NV,))
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
        "step_offset": step_offset,
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
    metrics["tracking_pos_error_per_step"] = jp.zeros(())
    metrics["tracking_vel_error_per_step"] = jp.zeros(())
    metrics["max_body_pos_error_per_step"] = jp.zeros(())
    metrics["term/nan"] = jp.zeros(())
    metrics["term/pose"] = jp.zeros(())

    obs = self._get_obs(data, info, traj_idx, target_qpos, target_qvel)
    reward, done = jp.zeros(2)

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # Apply perturbation forces to fingertips.
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)

    # Policy outputs target joint positions relative to default pose.
    wrist_action = (
        action[self._wrist_joint_ids] * self._config.wrist_action_scale
    )
    finger_action = (
        action[self._finger_joint_ids] * self._config.finger_action_scale
    )
    action = action.at[self._wrist_joint_ids].set(wrist_action)
    action = action.at[self._finger_joint_ids].set(finger_action)

    position_targets = self._default_pose + action
    # Clip to joint limits so PD controller never drives past range.
    position_targets = jp.clip(
        position_targets,
        self._mj_model.jnt_range[self._joint_ids, 0],
        self._mj_model.jnt_range[self._joint_ids, 1],
    )

    # PD torque: tau = kp * (target - q) - kd * qvel
    joint_pos = state.data.qpos[self._joint_qids]
    joint_vel = state.data.qvel[self._joint_dqids]
    torque = (
        state.info["kp"] * (position_targets - joint_pos)
        - state.info["kd"] * joint_vel
    )
    torque = jp.clip(torque, self._torque_low, self._torque_high)

    # Step physics.
    data = mjx_env.step(self.mjx_model, state.data, torque, self.n_substeps)

    # Compute trajectory index once, use everywhere.
    traj_idx = (
        (state.info["steps"] + state.info["step_offset"]) % self._traj_len
    ).astype(jp.int32)
    target_qpos = self._traj_qpos[traj_idx]
    target_qvel = self._traj_qvel[traj_idx]
    ref_body_pos = self._traj_xpos[traj_idx]
    ref_key_pos = self._traj_key_xpos[traj_idx]

    # Observations, termination, rewards.
    obs = self._get_obs(data, state.info, traj_idx, target_qpos, target_qvel)
    done, term_reasons = self._get_termination(data, state.info, ref_body_pos)
    rewards = self._get_reward(
        data,
        action,
        state.info,
        target_qpos,
        target_qvel,
    )
    rewards = {
        key: value * self._config.reward_config.scales[key] * self.dt
        for key, value in rewards.items()
    }
    sum_reward = sum(rewards.values()) if rewards else jp.zeros(())

    # Compute tracking errors for monitoring.
    joint_pos = data.qpos[self._joint_qids]
    joint_vel = data.qvel[self._joint_dqids]
    state.metrics["tracking_pos_error_per_step"] = jp.mean(
        jp.square(joint_pos - target_qpos)
    )
    state.metrics["tracking_vel_error_per_step"] = jp.mean(
        jp.square(joint_vel - target_qvel)
    )
    # Body cartesian position error for monitoring (wrist-local frame for fingers).
    cur_body_pos = self._get_tracked_body_pos(data)
    body_dist_sq = jp.sum(jp.square(cur_body_pos - ref_body_pos), axis=-1)
    state.metrics["max_body_pos_error_per_step"] = jp.sqrt(jp.max(body_dist_sq))

    # Termination reason diagnostics.
    for k, v in term_reasons.items():
      state.metrics[k] = v

    # Update info and metrics.
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(sum_reward.dtype)
    return state.replace(data=data, obs=obs, reward=sum_reward, done=done)

  # Helper methods. -----------------------------------------------------------

  def _get_obs(
      self,
      data: mjx.Data,
      info: dict[str, Any],
      traj_idx: jax.Array,
      target_qpos: jax.Array,
      target_qvel: jax.Array,
  ) -> mjx_env.Observation:
    # DeepMimic-style observation: wrist = root (floating base),
    # fingers = joints.  Each hand is treated independently.
    #
    # Joint layout (per hand, 14 DOF):
    #   [0:2]   wrist slide XY   (root position, 2 DOF)
    #   [2:5]   wrist hinge RPY  (root rotation, 3 DOF)
    #   [5:14]  finger hinges    (joint angles, 9 DOF)
    # Left hand = indices 0..13, Right hand = indices 14..27.

    joint_pos = data.qpos[self._joint_qids]  # (28,)
    joint_vel = data.qvel[self._joint_dqids]  # (28,)

    # --- Encoder noise (applied to joint positions only) ---
    info["rng"], noise_rng = jax.random.split(info["rng"])
    pos_noise = (
        (2 * jax.random.uniform(noise_rng, shape=joint_pos.shape) - 1)
        * self._config.obs_noise.level
        * self._config.obs_noise.scales.joint_pos
    )
    noisy_jpos = joint_pos + pos_noise
    # noisy_jpos = joint_pos

    # === Left hand (wrist = root) ===
    # Wrist position: XY from slide joints (2,)
    l_wrist_pos = noisy_jpos[consts.L_WRIST_SLIDE]
    # Wrist rotation: sin/cos encoding of RPY hinge joints (6,)
    l_wrist_rot_obs = jp.concatenate([
        jp.sin(noisy_jpos[consts.L_WRIST_HINGE]),
        jp.cos(noisy_jpos[consts.L_WRIST_HINGE]),
    ])
    # Wrist velocity: linear (2,) + angular (3,)
    l_wrist_lin_vel = joint_vel[consts.L_WRIST_SLIDE]
    l_wrist_ang_vel = joint_vel[consts.L_WRIST_HINGE]
    # Finger joint angles relative to default pose (9,)
    l_finger_qpos = (
        noisy_jpos[consts.L_FINGER_ALL]
        - self._default_pose[consts.L_FINGER_ALL]
    )
    # Finger joint velocities (9,)
    l_finger_qvel = joint_vel[consts.L_FINGER_ALL]

    # === Right hand (wrist = root) ===
    r_wrist_pos = noisy_jpos[consts.R_WRIST_SLIDE]
    r_wrist_rot_obs = jp.concatenate([
        jp.sin(noisy_jpos[consts.R_WRIST_HINGE]),
        jp.cos(noisy_jpos[consts.R_WRIST_HINGE]),
    ])
    r_wrist_lin_vel = joint_vel[consts.R_WRIST_SLIDE]
    r_wrist_ang_vel = joint_vel[consts.R_WRIST_HINGE]
    r_finger_qpos = (
        noisy_jpos[consts.R_FINGER_ALL]
        - self._default_pose[consts.R_FINGER_ALL]
    )
    r_finger_qvel = joint_vel[consts.R_FINGER_ALL]

    # === Future target observations (DeepMimic G1 style, global_obs=True) ===
    # Per target step per hand:
    #   wrist_pos: target - current (relative)
    #   wrist_rot: absolute sin/cos encoding
    #   finger_qpos: absolute joint angle relative to default pose
    # Plus: contact force reference for all 8 contact bodies.
    target_obs_list = []
    for step_offset in self._config.target_obs_steps:
      future_idx = (traj_idx + step_offset) % self._traj_len
      target_qpos_future = self._traj_qpos[future_idx]

      # Left hand target (16,): pos_diff(2) + rot_sincos(6) + finger(9) = 17
      left_target = jp.concatenate([
          target_qpos_future[consts.L_WRIST_SLIDE]
          - joint_pos[consts.L_WRIST_SLIDE],
          jp.sin(target_qpos_future[consts.L_WRIST_HINGE]),
          jp.cos(target_qpos_future[consts.L_WRIST_HINGE]),
          target_qpos_future[consts.L_FINGER_ALL]
          - self._default_pose[consts.L_FINGER_ALL],
      ])

      # Right hand target (17,)
      right_target = jp.concatenate([
          target_qpos_future[consts.R_WRIST_SLIDE]
          - joint_pos[consts.R_WRIST_SLIDE],
          jp.sin(target_qpos_future[consts.R_WRIST_HINGE]),
          jp.cos(target_qpos_future[consts.R_WRIST_HINGE]),
          target_qpos_future[consts.R_FINGER_ALL]
          - self._default_pose[consts.R_FINGER_ALL],
      ])

      # Contact force reference (24,): 8 bodies × 3.
      cf_ref = self._traj_contact_force[future_idx].flatten()

      target_obs_list.append(
          jp.concatenate([left_target, right_target, cf_ref])
      )

    target_obs = jp.concatenate(target_obs_list)

    # === Actuator force (all joints, 28) ===
    actuator_force = data.actuator_force

    # === Actor observation (state) ===
    # Per hand (31): wrist_pos(2) + wrist_rot_sincos(6) + wrist_lin_vel(2)
    #   + wrist_ang_vel(3) + finger_qpos(9) + finger_qvel(9)
    state_obs = jp.concatenate([
        # Left hand proprioception (31,)
        l_wrist_pos,
        # l_wrist_rot_obs,
        # l_wrist_lin_vel,
        # l_wrist_ang_vel,
        l_finger_qpos,
        l_finger_qvel,
        # Right hand proprioception (31,)
        r_wrist_pos,
        # r_wrist_rot_obs,
        # r_wrist_lin_vel,
        # r_wrist_ang_vel,
        r_finger_qpos,
        r_finger_qvel,
        # Future targets: (17+17+24) * num_target_steps
        target_obs,
        # Last action (28,)
        info["last_act"],
    ])

    # === Privileged critic observation ===
    # Additional: actual cfrc_ext for contact bodies (8 bodies × 3 = 24).
    contact_cfrc = data.cfrc_ext[self._contact_body_ids, 3:]  # (8, 3)
    privileged_state = jp.concatenate([
        state_obs,
        # Actuator force (28,)
        actuator_force,
        contact_cfrc.flatten(),
    ])

    return {
        "state": state_obs,
        "privileged_state": privileged_state,
    }

  def _get_tracked_body_pos(self, data: mjx.Data) -> jax.Array:
    """Get tracked body positions with finger bodies in wrist-local frame.

    Wrist positions are in world frame. Finger body positions are relative
    to their respective wrist's local coordinate frame:
      p_local = R_wrist^T @ (p_world - p_wrist)

    This matches the coordinate convention used in the expert trajectory data.
    """
    all_xpos = data.xpos[self._tracked_body_ids]  # (20, 3)

    # Wrist positions (world frame) and rotation matrices (3x3)
    l_wrist_pos = data.xpos[self._l_wrist_body_id]  # (3,)
    r_wrist_pos = data.xpos[self._r_wrist_body_id]  # (3,)
    l_wrist_rot = data.xmat[self._l_wrist_body_id].reshape(3, 3)  # (3, 3)
    r_wrist_rot = data.xmat[self._r_wrist_body_id].reshape(3, 3)  # (3, 3)

    # Left finger bodies: world -> left wrist local frame
    left_local = (all_xpos[self._left_finger_slice] - l_wrist_pos) @ l_wrist_rot
    # Right finger bodies: world -> right wrist local frame
    right_local = (
        all_xpos[self._right_finger_slice] - r_wrist_pos
    ) @ r_wrist_rot

    # Assemble: wrists in world frame, fingers in wrist-local frame
    result = all_xpos  # (20, 3)
    result = result.at[self._left_finger_slice].set(left_local)
    result = result.at[self._right_finger_slice].set(right_local)
    return result

  def _get_key_body_pos(self, data: mjx.Data) -> jax.Array:
    """Get key body positions (fingertips) in wrist-local frame."""
    all_key_xpos = data.xpos[self._key_body_ids]  # (N_key, 3)

    l_wrist_pos = data.xpos[self._l_wrist_body_id]
    r_wrist_pos = data.xpos[self._r_wrist_body_id]
    l_wrist_rot = data.xmat[self._l_wrist_body_id].reshape(3, 3)
    r_wrist_rot = data.xmat[self._r_wrist_body_id].reshape(3, 3)

    n = self._n_left_key
    left_local = (all_key_xpos[:n] - l_wrist_pos) @ l_wrist_rot
    right_local = (all_key_xpos[n:] - r_wrist_pos) @ r_wrist_rot
    return jp.concatenate([left_local, right_local], axis=0)

  def _get_termination(
      self,
      data: mjx.Data,
      info: dict[str, Any],
      ref_body_pos: jax.Array,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    # 1. NaN safety check (MJX can produce NaN on simulation divergence).
    nan_fail = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

    # 2. Pose termination: max tracked-body cartesian position error.
    #    Finger bodies are compared in wrist-local frame (matching expert data).
    cur_body_pos = self._get_tracked_body_pos(data)
    body_pos_diff = cur_body_pos - ref_body_pos
    body_pos_dist_sq = jp.sum(body_pos_diff * body_pos_diff, axis=-1)
    max_body_dist_sq = jp.max(body_pos_dist_sq)
    threshold_sq = self._config.pose_termination_dist**2
    pose_fail = max_body_dist_sq > threshold_sq

    # Only fail after first timestep (MimicKit: not_first_step guard).
    not_first_step = info["steps"] > 0

    done = jp.zeros((), dtype=jp.bool_)
    if self._config.terminate_on_nan:
      done = done | nan_fail
    if self._config.terminate_on_pose:
      done = done | (not_first_step & pose_fail)

    term_reasons = {
        "term/nan": nan_fail.astype(jp.float32),
        "term/pose": (not_first_step & pose_fail).astype(jp.float32),
    }
    return done.astype(jp.float32), term_reasons

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    """Apply periodic sinusoidal force perturbation to fingertip bodies."""
    info = state.info
    steps = info["steps"]
    last_pert_step = info["last_pert_step"]

    # Check if a new perturbation should start.
    start_pert = jp.mod(steps, info["pert_wait_steps"]) == 0
    start_pert &= steps != 0  # No perturbation at step 0.
    last_pert_step = jp.where(start_pert, steps, last_pert_step)
    duration = jp.clip(steps - last_pert_step, 0, 100_000)
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
      target_qpos: jax.Array,
      target_qvel: jax.Array,
  ) -> dict[str, jax.Array]:
    return {
        "pose": self._reward_pose(data, target_qpos),
        "vel": self._reward_vel(data, target_qvel),
        "root_pose": self._reward_root_pose(data, target_qpos),
        "root_vel": self._reward_root_vel(data, target_qvel),
        "key_pos": self._reward_key_pos(data, info),
        "contact_force": self._reward_contact_force(data, info),
        "action_rate": self._reward_action_rate(action, info["last_act"]),
        # "action_smooth": self._reward_action_smooth(
        #     action, info["last_act"], info["last_last_act"]
        # ),
        # "energy": self._reward_energy(data),
    }

  # Reward functions (DeepMimic style). ----------------------------------------
  #
  # The massage task maps onto DeepMimic as follows:
  #   DeepMimic root  -> wrist (5 DOF per hand: 2 slide + 3 hinge)
  #   DeepMimic joints -> finger joints (9 per hand)
  #   DeepMimic key_pos -> fingertip cartesian positions (wrist-local frame)
  #
  # Each sub-reward has the form: r = exp(-scale * err)

  def _reward_pose(self, data: mjx.Data, target_qpos: jax.Array) -> jax.Array:
    """Finger joint pose tracking (DeepMimic pose_r).

    err = sum(w_j * (q_j - q_j^*)²)  over all finger joints.
    r = exp(-pose_scale * err)
    """
    finger_ids = jp.array(consts.FINGER_INDICES)
    joint_pos = data.qpos[self._joint_qids]
    finger_pos = joint_pos[finger_ids]
    finger_tar = target_qpos[finger_ids]

    diff = finger_pos - finger_tar
    pose_err = jp.sum(self._finger_err_w * diff * diff)
    return jp.exp(-self._config.reward_config.pose_scale * pose_err)

  def _reward_vel(self, data: mjx.Data, target_qvel: jax.Array) -> jax.Array:
    """Finger joint velocity tracking (DeepMimic vel_r).

    err = sum(w_j * (dq_j - dq_j^*)²)  over all finger joints.
    r = exp(-vel_scale * err)
    """
    finger_ids = jp.array(consts.FINGER_INDICES)
    joint_vel = data.qvel[self._joint_dqids]
    finger_vel = joint_vel[finger_ids]
    finger_tar_vel = target_qvel[finger_ids]

    diff = finger_vel - finger_tar_vel
    vel_err = jp.sum(self._finger_err_w * diff * diff)
    return jp.exp(-self._config.reward_config.vel_scale * vel_err)

  def _reward_root_pose(
      self, data: mjx.Data, target_qpos: jax.Array
  ) -> jax.Array:
    """Wrist (root) pose tracking (DeepMimic root_pose_r).

    For each hand:
      pos_err = ||p - p*||²   (2D wrist slide joints)
      rot_err = ||θ - θ*||²   (3D wrist hinge joints, radian diff)
    r = exp(-root_pose_scale * (pos_err + rot_coeff * rot_err))
    """
    joint_pos = data.qpos[self._joint_qids]

    # Left wrist: slide [0:2], hinge [2:5]
    l_pos_err = jp.sum(
        jp.square(
            joint_pos[consts.L_WRIST_SLIDE] - target_qpos[consts.L_WRIST_SLIDE]
        )
    )
    l_rot_err = jp.sum(
        jp.square(
            joint_pos[consts.L_WRIST_HINGE] - target_qpos[consts.L_WRIST_HINGE]
        )
    )
    # Right wrist: slide [14:16], hinge [16:19]
    r_pos_err = jp.sum(
        jp.square(
            joint_pos[consts.R_WRIST_SLIDE] - target_qpos[consts.R_WRIST_SLIDE]
        )
    )
    r_rot_err = jp.sum(
        jp.square(
            joint_pos[consts.R_WRIST_HINGE] - target_qpos[consts.R_WRIST_HINGE]
        )
    )

    pos_err = (l_pos_err + r_pos_err) * 20.0  # normalize for cm-scale range
    rot_err = l_rot_err + r_rot_err
    rot_coeff = self._config.reward_config.root_pose_rot_coeff
    scale = self._config.reward_config.root_pose_scale
    return jp.exp(-scale * (pos_err + rot_coeff * rot_err))

  def _reward_root_vel(
      self, data: mjx.Data, target_qvel: jax.Array
  ) -> jax.Array:
    """Wrist (root) velocity tracking (DeepMimic root_vel_r).

    For each hand:
      lin_vel_err = ||v - v*||²   (wrist slide velocities)
      ang_vel_err = ||ω - ω*||²   (wrist hinge velocities)
    r = exp(-root_vel_scale * (lin_vel_err + rot_coeff * ang_vel_err))
    """
    joint_vel = data.qvel[self._joint_dqids]

    # Left wrist velocities
    l_lin_err = jp.sum(
        jp.square(
            joint_vel[consts.L_WRIST_SLIDE] - target_qvel[consts.L_WRIST_SLIDE]
        )
    )
    l_ang_err = jp.sum(
        jp.square(
            joint_vel[consts.L_WRIST_HINGE] - target_qvel[consts.L_WRIST_HINGE]
        )
    )
    # Right wrist velocities
    r_lin_err = jp.sum(
        jp.square(
            joint_vel[consts.R_WRIST_SLIDE] - target_qvel[consts.R_WRIST_SLIDE]
        )
    )
    r_ang_err = jp.sum(
        jp.square(
            joint_vel[consts.R_WRIST_HINGE] - target_qvel[consts.R_WRIST_HINGE]
        )
    )

    lin_err = l_lin_err + r_lin_err
    ang_err = l_ang_err + r_ang_err
    rot_coeff = self._config.reward_config.root_vel_rot_coeff
    scale = self._config.reward_config.root_vel_scale
    return jp.exp(-scale * (lin_err + rot_coeff * ang_err))

  def _reward_key_pos(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    """Key body (fingertip) cartesian position tracking (DeepMimic key_pos_r).

    Fingertip positions are in wrist-local frame (analogous to DeepMimic's
    key_pos relative to root).
    err = sum_k ||p_k - p_k*||²
    r = exp(-key_pos_scale * err)
    """
    traj_idx = ((info["steps"] + info["step_offset"]) % self._traj_len).astype(
        jp.int32
    )
    ref_key_pos = self._traj_key_xpos[traj_idx]  # (N_key, 3)

    cur_key_pos = self._get_key_body_pos(data)  # (N_key, 3) wrist-local
    key_pos_diff = cur_key_pos - ref_key_pos
    key_pos_err = jp.sum(jp.square(key_pos_diff))
    return jp.exp(-self._config.reward_config.key_pos_scale * key_pos_err)

  def _reward_contact_force(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    """Contact force tracking: r = exp(-scale * ||cfrc_ext - ref||²).

    Compares actual cfrc_ext linear force on contact bodies against the
    reference force trajectory synthesised from replay data.
    """
    traj_idx = ((info["steps"] + info["step_offset"]) % self._traj_len).astype(
        jp.int32
    )
    ref_force = self._traj_contact_force[traj_idx]  # (8, 3)
    cur_force = data.cfrc_ext[self._contact_body_ids, 3:]  # (8, 3)
    err = jp.sum(jp.square(cur_force - ref_force))
    return jp.exp(-self._config.reward_config.contact_force_scale * err)

  def _reward_action_rate(
      self, act: jax.Array, last_act: jax.Array
  ) -> jax.Array:
    """Action rate penalty (1st order): sum of squared first-order differences."""
    return jp.sum(jp.square(act - last_act))

  def _reward_action_smooth(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    """Action smoothness penalty (2nd order): sum of squared second-order differences."""
    return jp.sum(jp.square(act - 2 * last_act + last_last_act))

  # def _reward_energy(self, data: mjx.Data) -> jax.Array:
  #   """Energy consumption penalty: sum(|qvel * actuator_force|)."""
  #   joint_vel = data.qvel[self._joint_dqids]
  #   return jp.sum(jp.abs(joint_vel * data.actuator_force))

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
  metadata = _domain_randomization_metadata()
  collision_geom_ids = metadata["collision_geom_ids"]
  hand_body_ids = metadata["hand_body_ids"]
  joint_dof_ids = metadata["joint_dof_ids"]

  @jax.vmap
  def rand(rng):
    # 1. Contact friction: =U(0.3, 1.0) for collision-enabled hand geoms.
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
