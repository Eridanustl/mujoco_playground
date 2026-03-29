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
"""Deploy an MJX massage policy in ONNX format to C MuJoCo and play with it.

The observation and action processing must exactly match the training
environment (massage.py) to ensure correct sim2sim transfer.

Training observation (292-dim "state"):
  - Left hand proprioception (31):
      wrist_pos(2) + wrist_rot_sincos(6) + wrist_linvel(2)
      + wrist_angvel(3) + finger_qpos_rel(9) + finger_qvel(9)
  - Right hand proprioception (31): same structure.
  - Future trajectory targets (174):
      3 future steps x 58 dims each (left 17 + right 17 + cf_ref 24).
  - Last action (28)
  - Actuator force (28)

Action processing:
  1. Scale wrist actions by 0.05, finger actions by 0.5
  2. position_targets = default_pose + scaled_action
  3. Clip to joint limits
  4. PD torque = kp * (target - q) - kd * qvel
  5. Clip torque to actuator control range

After the viewer is closed, plots of reference vs actual trajectories
(joint positions and contact forces) are saved to the plots/ directory.
"""

import pickle
from pathlib import Path

from etils import epath
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

from mujoco_playground._src.manipulation.xleo_hand import constants as consts
from mujoco_playground._src.manipulation.xleo_hand.massage import get_assets
from mujoco_playground._src.mjx_env import get_qpos_ids
from mujoco_playground._src.mjx_env import get_qvel_ids

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"
_PLOT_DIR = Path(__file__).parent / "plots"

# Future target observation steps (must match training config).
_TARGET_OBS_STEPS = [1, 2, 3]


class OnnxController:
  """ONNX controller for the XleoMassage dual-hand massage task.

  Replicates the exact observation construction and action processing
  from the MJX training environment (massage.py).
  """

  def __init__(
      self,
      policy_path: str,
      model: mujoco.MjModel,
      joint_qids: np.ndarray,
      joint_dqids: np.ndarray,
      default_pose: np.ndarray,
      kp: np.ndarray,
      kd: np.ndarray,
      traj_qpos: np.ndarray,
      traj_contact_force: np.ndarray,
      traj_len: int,
      n_substeps: int,
      obs_mean: np.ndarray,
      obs_std: np.ndarray,
      contact_body_ids: np.ndarray,
      wrist_action_scale: float = 0.05,
      finger_action_scale: float = 0.5,
      action_ema_alpha: float = 0.3,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    self._model = model
    self._joint_qids = joint_qids
    self._joint_dqids = joint_dqids
    self._default_pose = default_pose
    self._kp = kp
    self._kd = kd
    self._traj_qpos = traj_qpos
    self._traj_contact_force = traj_contact_force  # (T, 8, 3)
    self._traj_len = traj_len
    self._wrist_action_scale = wrist_action_scale
    self._finger_action_scale = finger_action_scale
    self._contact_body_ids = contact_body_ids

    # Wrist/finger index masks into the 28-dim action/joint array.
    # Left: wrist=[0..4], finger=[5..13]; Right: wrist=[14..18], finger=[19..27]
    self._wrist_ids = np.array(consts.WRIST_INDICES, dtype=np.int32)
    self._finger_ids = np.array(consts.FINGER_INDICES, dtype=np.int32)

    # Joint limits for clipping position targets.
    joint_ids = np.array(
        [model.joint(n).id for n in consts.JOINT_NAMES], dtype=np.int32
    )
    self._jnt_range_low = model.jnt_range[joint_ids, 0]
    self._jnt_range_high = model.jnt_range[joint_ids, 1]

    # Torque limits from actuator control range.
    self._torque_low = model.actuator_ctrlrange[:, 0]
    self._torque_high = model.actuator_ctrlrange[:, 1]

    # Action clipping bounds derived from training normalizer statistics.
    act_dim = consts.NU
    la_mean = obs_mean[-act_dim:]
    la_std = obs_std[-act_dim:]
    self._action_lo = (la_mean - 4.0 * la_std).astype(np.float32)
    self._action_hi = (la_mean + 4.0 * la_std).astype(np.float32)

    self._last_action = np.zeros(consts.NU, dtype=np.float32)
    self._ema_action = np.zeros(consts.NU, dtype=np.float32)
    self._action_ema_alpha = action_ema_alpha  # 0→全平滑, 1→无滤波
    self._counter = 0
    self._n_substeps = n_substeps
    self._traj_idx = 0

    # --- Recording buffers for post-playback plotting ---
    self._rec_actual_qpos = []  # (N, 28) actual joint positions
    self._rec_ref_qpos = []  # (N, 28) reference joint positions
    self._rec_actual_cf = []  # (N, 8, 3) actual contact forces (cfrc_ext)
    self._rec_ref_cf = []  # (N, 8, 3) reference contact forces
    self._rec_actuator_force = []  # (N, 28) actuator forces

  def _get_obs(self, data: mujoco.MjData) -> np.ndarray:
    """Construct 292-dim observation matching the training environment.

    Replicates massage.py _get_obs() with correct 28-DOF index slices.
    """
    joint_pos = data.qpos[self._joint_qids]  # (28,)
    joint_vel = data.qvel[self._joint_dqids]  # (28,)

    # === Left hand proprioception (31 dims) ===
    l_wrist_pos = joint_pos[consts.L_WRIST_SLIDE]  # (2,)
    l_wrist_rot_obs = np.concatenate([
        np.sin(joint_pos[consts.L_WRIST_HINGE]),
        np.cos(joint_pos[consts.L_WRIST_HINGE]),
    ])  # (6,)
    l_wrist_lin_vel = joint_vel[consts.L_WRIST_SLIDE]  # (2,)
    l_wrist_ang_vel = joint_vel[consts.L_WRIST_HINGE]  # (3,)
    l_finger_qpos = (
        joint_pos[consts.L_FINGER_ALL] - self._default_pose[consts.L_FINGER_ALL]
    )  # (9,)
    l_finger_qvel = joint_vel[consts.L_FINGER_ALL]  # (9,)

    # === Right hand proprioception (31 dims) ===
    r_wrist_pos = joint_pos[consts.R_WRIST_SLIDE]  # (2,)
    r_wrist_rot_obs = np.concatenate([
        np.sin(joint_pos[consts.R_WRIST_HINGE]),
        np.cos(joint_pos[consts.R_WRIST_HINGE]),
    ])  # (6,)
    r_wrist_lin_vel = joint_vel[consts.R_WRIST_SLIDE]  # (2,)
    r_wrist_ang_vel = joint_vel[consts.R_WRIST_HINGE]  # (3,)
    r_finger_qpos = (
        joint_pos[consts.R_FINGER_ALL] - self._default_pose[consts.R_FINGER_ALL]
    )  # (9,)
    r_finger_qvel = joint_vel[consts.R_FINGER_ALL]  # (9,)

    # === Future target observations (174 dims) ===
    target_obs_list = []
    for step_offset in _TARGET_OBS_STEPS:
      future_idx = (self._traj_idx + step_offset) % self._traj_len
      tgt = self._traj_qpos[future_idx]  # (28,)

      # Left target (17 dims)
      left_target = np.concatenate([
          tgt[consts.L_WRIST_SLIDE] - joint_pos[consts.L_WRIST_SLIDE],  # (2,)
          np.sin(tgt[consts.L_WRIST_HINGE]),
          np.cos(tgt[consts.L_WRIST_HINGE]),  # (6,)
          tgt[consts.L_FINGER_ALL]
          - self._default_pose[consts.L_FINGER_ALL],  # (9,)
      ])

      # Right target (17 dims)
      right_target = np.concatenate([
          tgt[consts.R_WRIST_SLIDE] - joint_pos[consts.R_WRIST_SLIDE],  # (2,)
          np.sin(tgt[consts.R_WRIST_HINGE]),
          np.cos(tgt[consts.R_WRIST_HINGE]),  # (6,)
          tgt[consts.R_FINGER_ALL]
          - self._default_pose[consts.R_FINGER_ALL],  # (9,)
      ])

      # Contact force reference (24 dims): 8 bodies × 3
      cf_ref = self._traj_contact_force[future_idx].flatten()

      target_obs_list.append(
          np.concatenate([left_target, right_target, cf_ref])
      )

    target_obs = np.concatenate(target_obs_list)  # (174,)

    # === Assemble full 264-dim observation ===
    obs = np.concatenate([
        # Left hand (31)
        l_wrist_pos,
        l_wrist_rot_obs,
        l_wrist_lin_vel,
        l_wrist_ang_vel,
        l_finger_qpos,
        l_finger_qvel,
        # Right hand (31)
        r_wrist_pos,
        r_wrist_rot_obs,
        r_wrist_lin_vel,
        r_wrist_ang_vel,
        r_finger_qpos,
        r_finger_qvel,
        # Future targets (174)
        target_obs,
        # Last action (28)
        self._last_action,
    ])
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps != 0:
      return

    # --- Record actual vs reference data before computing new control ---
    joint_pos = data.qpos[self._joint_qids].copy()
    self._rec_actual_qpos.append(joint_pos)
    self._rec_ref_qpos.append(self._traj_qpos[self._traj_idx].copy())

    # Actual contact forces: cfrc_ext linear force (world frame).
    actual_cf = data.cfrc_ext[self._contact_body_ids, 3:].copy()  # (8, 3)
    self._rec_actual_cf.append(actual_cf)
    self._rec_ref_cf.append(self._traj_contact_force[self._traj_idx].copy())

    self._rec_actuator_force.append(data.actuator_force.copy())

    # --- Policy inference ---
    obs = self._get_obs(data)
    onnx_input = {"obs": obs.reshape(1, -1)}
    onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]

    # Clip raw action to prevent feedback divergence.
    onnx_pred = np.clip(onnx_pred, self._action_lo, self._action_hi)

    # EMA low-pass filter: smoothed = alpha * new + (1-alpha) * prev.
    # alpha=1.0 means no filtering; smaller alpha = more smoothing.
    self._ema_action = (
        self._action_ema_alpha * onnx_pred
        + (1.0 - self._action_ema_alpha) * self._ema_action
    )
    smoothed_action = self._ema_action

    # Apply per-joint action scales (must match training).
    scaled_action = smoothed_action.copy()
    scaled_action[self._wrist_ids] *= self._wrist_action_scale
    scaled_action[self._finger_ids] *= self._finger_action_scale

    # Compute position targets and clip to joint limits.
    position_targets = self._default_pose + scaled_action
    position_targets = np.clip(
        position_targets, self._jnt_range_low, self._jnt_range_high
    )

    # PD torque control: tau = kp * (target - q) - kd * qvel
    joint_vel = data.qvel[self._joint_dqids]
    torque = self._kp * (position_targets - joint_pos) - self._kd * joint_vel
    torque = np.clip(torque, self._torque_low, self._torque_high)

    data.ctrl[:] = torque
    self._last_action = onnx_pred.copy()

    # Advance trajectory index.
    self._traj_idx = (self._traj_idx + 1) % self._traj_len

  def get_recorded_data(self) -> dict:
    """Return recorded trajectory data as numpy arrays."""
    n = len(self._rec_actual_qpos)
    if n == 0:
      return {}
    return {
        "actual_qpos": np.array(self._rec_actual_qpos),  # (N, 28)
        "ref_qpos": np.array(self._rec_ref_qpos),  # (N, 28)
        "actual_cf": np.array(self._rec_actual_cf),  # (N, 8, 3)
        "ref_cf": np.array(self._rec_ref_cf),  # (N, 8, 3)
        "actuator_force": np.array(self._rec_actuator_force),  # (N, 28)
        "n_steps": n,
    }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def _plot_qpos_tracking(
    ref: np.ndarray,
    actual: np.ndarray,
    times: np.ndarray,
    save_dir: Path,
) -> None:
  """Plot reference vs actual joint positions, grouped by joint group."""
  n_groups = len(consts.JOINT_GROUPS)
  fig, axes = plt.subplots(
      n_groups, 1, figsize=(16, 3.2 * n_groups), squeeze=False, sharex=True
  )
  for g, (group_title, joints) in enumerate(consts.JOINT_GROUPS):
    ax = axes[g, 0]
    for idx, label in joints:
      (line,) = ax.plot(times, actual[:, idx], linewidth=1.5, label=f"{label}")
      ax.plot(
          times,
          ref[:, idx],
          linewidth=1.2,
          linestyle="--",
          color=line.get_color(),
          alpha=0.6,
      )
    ax.set_ylabel("Position (rad/m)", fontsize=11)
    ax.set_title(group_title, fontsize=13, loc="left", fontweight="bold")
    ax.legend(fontsize=9, ncol=len(joints), loc="upper right")
    ax.grid(True, alpha=0.3)
  axes[-1, 0].set_xlabel("Time (s)", fontsize=12)
  fig.suptitle(
      "Joint Position Tracking (solid=actual, dashed=reference)",
      fontsize=16,
      fontweight="bold",
  )
  fig.tight_layout(rect=[0, 0, 1, 0.97])
  path = save_dir / "sim2sim_qpos_tracking.png"
  fig.savefig(str(path), dpi=150)
  plt.close(fig)
  print(f"  Saved: {path}")


def _plot_contact_force_tracking(
    ref: np.ndarray,
    actual: np.ndarray,
    times: np.ndarray,
    save_dir: Path,
) -> None:
  """Plot reference vs actual contact forces for each tracked body."""
  body_names = list(consts.CONTACT_FORCE_BODY_NAMES)
  n_bodies = len(body_names)
  dim_labels = ["fx", "fy", "fz"]
  colors = ["red", "green", "blue"]

  fig, axes = plt.subplots(
      n_bodies, 1, figsize=(16, 3.0 * n_bodies), squeeze=False, sharex=True
  )
  for b in range(n_bodies):
    ax = axes[b, 0]
    for d in range(3):
      ax.plot(
          times,
          actual[:, b, d],
          linewidth=1.5,
          color=colors[d],
          label=f"{dim_labels[d]}",
      )
      ax.plot(
          times,
          ref[:, b, d],
          linewidth=1.2,
          linestyle="--",
          color=colors[d],
          alpha=0.6,
      )
    ax.set_ylabel("Force (N)", fontsize=11)
    ax.set_title(body_names[b], fontsize=13, loc="left", fontweight="bold")
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)
  axes[-1, 0].set_xlabel("Time (s)", fontsize=12)
  fig.suptitle(
      "Contact Force Tracking (solid=actual, dashed=reference)",
      fontsize=16,
      fontweight="bold",
  )
  fig.tight_layout(rect=[0, 0, 1, 0.97])
  path = save_dir / "sim2sim_contact_force_tracking.png"
  fig.savefig(str(path), dpi=150)
  plt.close(fig)
  print(f"  Saved: {path}")


def _plot_actuator_force(
    force: np.ndarray,
    times: np.ndarray,
    save_dir: Path,
) -> None:
  """Plot actuator forces, grouped by joint group."""
  n_groups = len(consts.JOINT_GROUPS)
  fig, axes = plt.subplots(
      n_groups, 1, figsize=(16, 3.2 * n_groups), squeeze=False, sharex=True
  )
  for g, (group_title, joints) in enumerate(consts.JOINT_GROUPS):
    ax = axes[g, 0]
    for idx, label in joints:
      ax.plot(times, force[:, idx], linewidth=1.5, label=f"{label}")
    ax.set_ylabel("Torque (N·m)", fontsize=11)
    ax.set_title(group_title, fontsize=13, loc="left", fontweight="bold")
    ax.legend(fontsize=9, ncol=len(joints), loc="upper right")
    ax.grid(True, alpha=0.3)
  axes[-1, 0].set_xlabel("Time (s)", fontsize=12)
  fig.suptitle("Actuator Forces", fontsize=16, fontweight="bold")
  fig.tight_layout(rect=[0, 0, 1, 0.97])
  path = save_dir / "sim2sim_actuator_force.png"
  fig.savefig(str(path), dpi=150)
  plt.close(fig)
  print(f"  Saved: {path}")


def plot_sim2sim_results(recorded: dict, ctrl_dt: float, save_dir: Path):
  """Generate all sim2sim tracking plots from recorded data."""
  if not recorded:
    print("No data recorded — skipping plots.")
    return

  save_dir.mkdir(parents=True, exist_ok=True)
  n = recorded["n_steps"]
  times = np.arange(n) * ctrl_dt
  print(f"\nGenerating sim2sim plots ({n} steps, {times[-1]:.2f}s) ...")

  _plot_qpos_tracking(
      recorded["ref_qpos"],
      recorded["actual_qpos"],
      times,
      save_dir,
  )
  _plot_contact_force_tracking(
      recorded["ref_cf"],
      recorded["actual_cf"],
      times,
      save_dir,
  )
  _plot_actuator_force(
      recorded["actuator_force"],
      times,
      save_dir,
  )
  print(f"All sim2sim plots saved to {save_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Global reference so we can access recorded data after viewer closes.
_controller: OnnxController | None = None


def load_callback(model=None, data=None):
  global _controller
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_string(
      epath.Path(consts.SCENE_XML).read_text(),
      assets=get_assets(),
  )
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 0)

  # Timing must match training: ctrl_dt=0.02, sim_dt=0.005 → 50 Hz policy.
  sim_dt = 0.001
  n_substeps = 20  # ctrl_dt / sim_dt = 0.02 / 0.005 = 4
  model.opt.timestep = sim_dt

  # Joint IDs for qpos and qvel indexing.
  joint_qids = get_qpos_ids(model, consts.JOINT_NAMES)
  joint_dqids = get_qvel_ids(model, consts.JOINT_NAMES)

  # Default pose from keyframe.
  default_pose = model.qpos0[joint_qids].copy()

  # Build per-joint PD gains (must match training config).
  # Left: wrist=[0..4], finger=[5..13]; Right: wrist=[14..18], finger=[19..27]
  wrist_ids = list(range(0, 5)) + list(range(14, 19))
  finger_ids = list(range(5, 14)) + list(range(19, 28))
  kp = np.zeros(consts.NU, dtype=np.float64)
  kd = np.zeros(consts.NU, dtype=np.float64)
  kp[wrist_ids] = 10.0
  kd[wrist_ids] = 0.5
  kp[finger_ids] = 5.0
  kd[finger_ids] = 0.1

  # Load expert massage trajectory.
  traj_path = consts.DATA_PATH / "massage_traj.pkl"
  with open(epath.Path(traj_path), "rb") as f:
    traj = pickle.load(f)
  traj_qpos = np.array(traj["qpos"])  # (T, 28)
  traj_len = traj_qpos.shape[0]

  # Contact force reference trajectory.
  if "tracked_contact_force" in traj:
    traj_contact_force = np.array(traj["tracked_contact_force"])  # (T, 8, 3)
  else:
    traj_contact_force = np.zeros((traj_len, 8, 3))

  # Load normalizer statistics (saved by export script alongside ONNX).
  norm_path = _ONNX_DIR / "xleo_massage_policy_norm.npz"
  norm_data = np.load(epath.Path(norm_path).as_posix())
  obs_mean = norm_data["obs_mean"]
  obs_std = norm_data["obs_std"]

  # Contact body IDs for cfrc_ext recording.
  contact_body_ids = np.array(
      [model.body(n).id for n in consts.CONTACT_FORCE_BODY_NAMES],
      dtype=np.int32,
  )

  _controller = OnnxController(
      policy_path=(_ONNX_DIR / "xleo_massage_policy.onnx").as_posix(),
      model=model,
      joint_qids=joint_qids,
      joint_dqids=joint_dqids,
      default_pose=default_pose,
      kp=kp,
      kd=kd,
      traj_qpos=traj_qpos,
      traj_contact_force=traj_contact_force,
      traj_len=traj_len,
      n_substeps=n_substeps,
      obs_mean=obs_mean,
      obs_std=obs_std,
      contact_body_ids=contact_body_ids,
      wrist_action_scale=0.05,
      finger_action_scale=0.5,
  )

  mujoco.set_mjcb_control(_controller.get_control)

  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)

  # After viewer closes, plot reference vs actual trajectories.
  if _controller is not None:
    recorded = _controller.get_recorded_data()
    ctrl_dt = 0.02  # Must match training: ctrl_dt = n_substeps * sim_dt
    plot_sim2sim_results(recorded, ctrl_dt, _PLOT_DIR)
