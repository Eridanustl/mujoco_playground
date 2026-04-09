"""XleoMassage2 ONNX policy sim2sim deployment.

Loads a trained ONNX policy and runs it in C MuJoCo viewer with PD torque
control. The environment starts from the XML "home" keyframe. After the
viewer is closed, tracking plots are saved automatically.

Usage:
    # Use default ONNX dir (sim2sim/onnx/):
    python play_xleo_massage.py

    # Specify ONNX model directory:
    python play_xleo_massage.py --model_dir /path/to/onnx_dir

    # The onnx_dir should contain:
    #   - xleo_massage_policy.onnx       (policy network)
    #   - xleo_massage_policy_norm.npz   (observation normalizer stats)

Observation layout (state, input to policy network):
    Left hand proprio (31) + Right hand proprio (31) + Future targets (174)
    + Last action (28) = 264 dims

Action processing pipeline:
    1. Scale: wrist *= WRIST_ACTION_SCALE, finger *= FINGER_ACTION_SCALE
    2. Position targets = default_pose + scaled_action
    3. Clip to joint limits
    4. PD torque: tau = kp * (target - q) - kd * qvel
    5. Clip torque to actuator ctrlrange
"""

import argparse
import pickle
from pathlib import Path

from etils import epath
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

from mujoco_playground._src.manipulation.xleo_hand import constants as consts
from mujoco_playground._src.manipulation.xleo_hand.massage2 import get_assets
from mujoco_playground._src.mjx_env import get_qpos_ids, get_qvel_ids

# ============================================================================
# Configuration — must match massage2.py training config
# ============================================================================

# --- Timing ---
SIM_DT = 0.001  # MuJoCo simulation timestep (s)
CTRL_DT = 0.02  # Policy control timestep (s)
N_SUBSTEPS = int(CTRL_DT / SIM_DT)  # = 4

# --- Action scales ---
WRIST_ACTION_SCALE = 0.05
FINGER_ACTION_SCALE = 0.5

# --- PD gains (per joint type) ---
WRIST_SLIDE_KP = 100.0  # Wrist linear X, Y
WRIST_SLIDE_KD = 20.0
WRIST_HINGE_KP = 10.0  # Wrist rotation ROLL, PITCH, YAW
WRIST_HINGE_KD = 0.5
FINGER_KP = 3.0  # Finger joints
FINGER_KD = 0.1

# --- Expert trajectory ---
TRAJ_PATH = consts.DATA_PATH / "massage_data_mink.pkl"

# --- Future target observation steps ---
TARGET_OBS_STEPS = [1, 2, 3]

# --- ONNX file names ---
ONNX_POLICY_NAME = "xleo_massage_policy.onnx"
ONNX_NORM_NAME = "xleo_massage_policy_norm.npz"

# --- Action EMA smoothing (1.0 = no filter, smaller = more smooth) ---
ACTION_EMA_ALPHA = 1.0

# --- Output ---
PLOT_DIR = Path(__file__).parent / "plots"

# ============================================================================
# Internal constants
# ============================================================================

_HERE = epath.Path(__file__).parent
_DEFAULT_ONNX_DIR = _HERE / "onnx"


# ============================================================================
# Helpers
# ============================================================================


def _build_kp_kd() -> tuple[np.ndarray, np.ndarray]:
  """Build per-joint kp/kd arrays from config constants."""
  kp = np.zeros(consts.NU, dtype=np.float64)
  kd = np.zeros(consts.NU, dtype=np.float64)
  kp[consts.WRIST_SLIDE_INDICES] = WRIST_SLIDE_KP
  kd[consts.WRIST_SLIDE_INDICES] = WRIST_SLIDE_KD
  kp[consts.WRIST_HINGE_INDICES] = WRIST_HINGE_KP
  kd[consts.WRIST_HINGE_INDICES] = WRIST_HINGE_KD
  kp[consts.FINGER_INDICES] = FINGER_KP
  kd[consts.FINGER_INDICES] = FINGER_KD
  return kp, kd


def _load_trajectory() -> dict:
  """Load expert trajectory. Returns dict with 'qpos', 'contact_force', 'traj_len'."""
  with open(epath.Path(TRAJ_PATH), "rb") as f:
    traj = pickle.load(f)
  traj_len = traj["qpos"].shape[0]
  contact_force = (
      np.array(traj["tracked_contact_force"])
      if "tracked_contact_force" in traj
      else np.zeros((traj_len, len(consts.CONTACT_FORCE_BODY_NAMES), 3))
  )
  return {
      "qpos": np.array(traj["qpos"]),
      "contact_force": contact_force,
      "traj_len": traj_len,
  }


# ============================================================================
# ONNX Controller
# ============================================================================


class OnnxController:
  """ONNX-based PD torque controller for XleoMassage2.

  Replicates the exact observation construction and action processing
  from massage2.py to ensure correct sim2sim transfer.
  """

  def __init__(
      self,
      policy_path: str,
      model: mujoco.MjModel,
      traj: dict,
      obs_mean: np.ndarray,
      obs_std: np.ndarray,
  ):
    # ONNX session
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    # Model and joint indexing
    self._model = model
    self._joint_qids = get_qpos_ids(model, consts.JOINT_NAMES)
    self._joint_dqids = get_qvel_ids(model, consts.JOINT_NAMES)
    self._joint_ids = np.array(
        [model.joint(n).id for n in consts.JOINT_NAMES], dtype=np.int32
    )
    self._default_pose = model.qpos0[self._joint_qids].copy()

    # PD gains and limits
    self._kp, self._kd = _build_kp_kd()
    self._jnt_range_low = model.jnt_range[self._joint_ids, 0]
    self._jnt_range_high = model.jnt_range[self._joint_ids, 1]
    self._torque_low = model.actuator_ctrlrange[:, 0]
    self._torque_high = model.actuator_ctrlrange[:, 1]

    # Wrist / finger index masks
    self._wrist_ids = np.array(consts.WRIST_INDICES, dtype=np.int32)
    self._finger_ids = np.array(consts.FINGER_INDICES, dtype=np.int32)

    # Expert trajectory
    self._traj_qpos = traj["qpos"]
    self._traj_contact_force = traj["contact_force"]
    self._traj_len = traj["traj_len"]

    # Contact body IDs for cfrc_ext recording
    self._contact_body_ids = np.array(
        [model.body(n).id for n in consts.CONTACT_FORCE_BODY_NAMES],
        dtype=np.int32,
    )

    # Runtime state
    self._last_action = np.zeros(consts.NU, dtype=np.float32)
    self._ema_action = np.zeros(consts.NU, dtype=np.float32)
    self._counter = 0
    self._traj_idx = 0
    self._n_record_cycles = 3  # Only record the first N trajectory cycles
    self._record_limit = self._traj_len * self._n_record_cycles

    # Recording buffers for post-playback plotting
    self._rec_actual_qpos: list[np.ndarray] = []
    self._rec_ref_qpos: list[np.ndarray] = []
    self._rec_actual_cf: list[np.ndarray] = []
    self._rec_ref_cf: list[np.ndarray] = []
    self._rec_actuator_force: list[np.ndarray] = []

  # ------------------------------------------------------------------
  # Observation construction
  # ------------------------------------------------------------------

  def _get_obs(self, data: mujoco.MjData) -> np.ndarray:
    """Build observation vector matching massage2.py _get_obs().

    Structure kept 1:1 with massage2.py so you can comment out the
    same lines in both files to keep them in sync.
    """
    joint_pos = data.qpos[self._joint_qids]  # (28,)
    joint_vel = data.qvel[self._joint_dqids]  # (28,)

    # === Left hand (wrist = root) ===
    # Wrist position: XY from slide joints (2,)
    l_wrist_pos = joint_pos[consts.L_WRIST_SLIDE]
    # Wrist rotation: sin/cos encoding of RPY hinge joints (6,)
    l_wrist_rot_obs = np.concatenate([
        np.sin(joint_pos[consts.L_WRIST_HINGE]),
        np.cos(joint_pos[consts.L_WRIST_HINGE]),
    ])
    # Wrist velocity: linear (2,) + angular (3,)
    l_wrist_lin_vel = joint_vel[consts.L_WRIST_SLIDE]
    l_wrist_ang_vel = joint_vel[consts.L_WRIST_HINGE]
    # Finger joint angles relative to default pose (9,)
    l_finger_qpos = (
        joint_pos[consts.L_FINGER_ALL] - self._default_pose[consts.L_FINGER_ALL]
    )
    # Finger joint velocities (9,)
    l_finger_qvel = joint_vel[consts.L_FINGER_ALL]

    # === Right hand (wrist = root) ===
    r_wrist_pos = joint_pos[consts.R_WRIST_SLIDE]
    r_wrist_rot_obs = np.concatenate([
        np.sin(joint_pos[consts.R_WRIST_HINGE]),
        np.cos(joint_pos[consts.R_WRIST_HINGE]),
    ])
    r_wrist_lin_vel = joint_vel[consts.R_WRIST_SLIDE]
    r_wrist_ang_vel = joint_vel[consts.R_WRIST_HINGE]
    r_finger_qpos = (
        joint_pos[consts.R_FINGER_ALL] - self._default_pose[consts.R_FINGER_ALL]
    )
    r_finger_qvel = joint_vel[consts.R_FINGER_ALL]

    # === Future target observations (DeepMimic G1 style, global_obs=True) ===
    # Per target step per hand:
    #   wrist_pos: target - current (relative)
    #   wrist_rot: absolute sin/cos encoding
    #   finger_qpos: absolute joint angle relative to default pose
    # Plus: contact force reference for all 8 contact bodies.
    target_obs_list = []
    for step_offset in TARGET_OBS_STEPS:
      future_idx = (self._traj_idx + step_offset) % self._traj_len
      target_qpos_future = self._traj_qpos[future_idx]

      # Left hand target (17,): pos_diff(2) + rot_sincos(6) + finger(9)
      left_target = np.concatenate([
          target_qpos_future[consts.L_WRIST_SLIDE]
          - joint_pos[consts.L_WRIST_SLIDE],
          np.sin(target_qpos_future[consts.L_WRIST_HINGE]),
          np.cos(target_qpos_future[consts.L_WRIST_HINGE]),
          target_qpos_future[consts.L_FINGER_ALL]
          - self._default_pose[consts.L_FINGER_ALL],
      ])

      # Right hand target (17,)
      right_target = np.concatenate([
          target_qpos_future[consts.R_WRIST_SLIDE]
          - joint_pos[consts.R_WRIST_SLIDE],
          np.sin(target_qpos_future[consts.R_WRIST_HINGE]),
          np.cos(target_qpos_future[consts.R_WRIST_HINGE]),
          target_qpos_future[consts.R_FINGER_ALL]
          - self._default_pose[consts.R_FINGER_ALL],
      ])

      # Contact force reference (24,): 8 bodies × 3.
      cf_ref = self._traj_contact_force[future_idx].flatten()

      target_obs_list.append(
          np.concatenate([left_target, right_target, cf_ref])
      )

    target_obs = np.concatenate(target_obs_list)

    # === Actor observation (state) ===
    # Per hand (31): wrist_pos(2) + wrist_rot_sincos(6) + wrist_lin_vel(2)
    #   + wrist_ang_vel(3) + finger_qpos(9) + finger_qvel(9)
    state_obs = np.concatenate([
        # Left hand proprioception (31,)
        l_wrist_pos,
        l_wrist_rot_obs,
        l_wrist_lin_vel,
        l_wrist_ang_vel,
        l_finger_qpos,
        l_finger_qvel,
        # Right hand proprioception (31,)
        r_wrist_pos,
        r_wrist_rot_obs,
        r_wrist_lin_vel,
        r_wrist_ang_vel,
        r_finger_qpos,
        r_finger_qvel,
        # Future targets: (17+17+24) * num_target_steps
        target_obs,
        # Last action (28,)
        self._last_action,
    ])

    return state_obs.astype(np.float32)

  # ------------------------------------------------------------------
  # Control callback
  # ------------------------------------------------------------------

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """MuJoCo control callback, called every simulation step."""
    self._counter += 1
    if self._counter % N_SUBSTEPS != 0:
      return

    joint_pos = data.qpos[self._joint_qids].copy()
    joint_vel = data.qvel[self._joint_dqids]

    # Record before computing new control (only first N cycles)
    if len(self._rec_actual_qpos) < self._record_limit:
      self._rec_actual_qpos.append(joint_pos)
      self._rec_ref_qpos.append(self._traj_qpos[self._traj_idx].copy())
      self._rec_actual_cf.append(
          data.cfrc_ext[self._contact_body_ids, 3:].copy()
      )
      self._rec_ref_cf.append(self._traj_contact_force[self._traj_idx].copy())
      self._rec_actuator_force.append(data.actuator_force.copy())

    # Policy inference
    obs = self._get_obs(data)
    raw_action = self._policy.run(["actions"], {"obs": obs.reshape(1, -1)})[0][
        0
    ]

    # EMA smoothing
    self._ema_action = (
        ACTION_EMA_ALPHA * raw_action
        + (1.0 - ACTION_EMA_ALPHA) * self._ema_action
    )

    # Scale action per joint type
    action = self._ema_action.copy()
    action[self._wrist_ids] *= WRIST_ACTION_SCALE
    action[self._finger_ids] *= FINGER_ACTION_SCALE

    # Position targets, clipped to joint limits
    targets = np.clip(
        self._default_pose + action,
        self._jnt_range_low,
        self._jnt_range_high,
    )

    # PD torque control
    torque = self._kp * (targets - joint_pos) - self._kd * joint_vel
    data.ctrl[:] = np.clip(torque, self._torque_low, self._torque_high)

    self._last_action = (
        action.copy()
    )  # Must be SCALED action to match training env
    self._traj_idx = (self._traj_idx + 1) % self._traj_len

  # ------------------------------------------------------------------
  # Data export
  # ------------------------------------------------------------------

  def get_recorded_data(self) -> dict:
    """Return recorded trajectory data as numpy arrays."""
    if not self._rec_actual_qpos:
      return {}
    return {
        "actual_qpos": np.array(self._rec_actual_qpos),
        "ref_qpos": np.array(self._rec_ref_qpos),
        "actual_cf": np.array(self._rec_actual_cf),
        "ref_cf": np.array(self._rec_ref_cf),
        "actuator_force": np.array(self._rec_actuator_force),
        "n_steps": len(self._rec_actual_qpos),
    }


# ============================================================================
# Plotting
# ============================================================================


def _plot_qpos_tracking(
    ref: np.ndarray, actual: np.ndarray, times: np.ndarray, save_dir: Path
) -> None:
  """Plot reference vs actual joint positions by group."""
  n_groups = len(consts.JOINT_GROUPS)
  fig, axes = plt.subplots(
      n_groups, 1, figsize=(16, 3.2 * n_groups), squeeze=False, sharex=True
  )
  for g, (title, joints) in enumerate(consts.JOINT_GROUPS):
    ax = axes[g, 0]
    for idx, label in joints:
      (line,) = ax.plot(times, actual[:, idx], lw=1.5, label=label)
      ax.plot(
          times,
          ref[:, idx],
          lw=1.2,
          ls="--",
          color=line.get_color(),
          alpha=0.6,
      )
    ax.set_ylabel("Position (rad/m)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(fontsize=9, ncol=len(joints), loc="upper right")
    ax.grid(True, alpha=0.3)
  axes[-1, 0].set_xlabel("Time (s)")
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
    ref: np.ndarray, actual: np.ndarray, times: np.ndarray, save_dir: Path
) -> None:
  """Plot reference vs actual contact forces per body."""
  body_names = list(consts.CONTACT_FORCE_BODY_NAMES)
  n = len(body_names)
  colors, labels = ["red", "green", "blue"], ["fx", "fy", "fz"]

  fig, axes = plt.subplots(
      n, 1, figsize=(16, 3.0 * n), squeeze=False, sharex=True
  )
  for b in range(n):
    ax = axes[b, 0]
    for d in range(3):
      ax.plot(
          times,
          actual[:, b, d],
          lw=1.5,
          color=colors[d],
          label=labels[d],
      )
      ax.plot(
          times,
          ref[:, b, d],
          lw=1.2,
          ls="--",
          color=colors[d],
          alpha=0.6,
      )
    ax.set_ylabel("Force (N)")
    ax.set_title(body_names[b], loc="left", fontweight="bold")
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)
  axes[-1, 0].set_xlabel("Time (s)")
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


def _plot_contact_force(
    actual: np.ndarray, times: np.ndarray, save_dir: Path
) -> None:
  """Plot actual contact forces per body (no reference)."""
  body_names = list(consts.CONTACT_FORCE_BODY_NAMES)
  n = len(body_names)
  colors, labels = ["red", "green", "blue"], ["fx", "fy", "fz"]

  fig, axes = plt.subplots(
      n, 1, figsize=(16, 3.0 * n), squeeze=False, sharex=True
  )
  for b in range(n):
    ax = axes[b, 0]
    for d in range(3):
      ax.plot(
          times,
          actual[:, b, d],
          lw=1.5,
          color=colors[d],
          label=labels[d],
      )
    ax.set_ylabel("Force (N)")
    ax.set_title(body_names[b], loc="left", fontweight="bold")
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)
  axes[-1, 0].set_xlabel("Time (s)")
  fig.suptitle(
      "Contact Force",
      fontsize=16,
      fontweight="bold",
  )
  fig.tight_layout(rect=[0, 0, 1, 0.97])
  path = save_dir / "sim2sim_contact_force.png"
  fig.savefig(str(path), dpi=150)
  plt.close(fig)
  print(f"  Saved: {path}")


def _plot_actuator_force(
    force: np.ndarray, times: np.ndarray, save_dir: Path
) -> None:
  """Plot actuator forces by joint group."""
  n_groups = len(consts.JOINT_GROUPS)
  fig, axes = plt.subplots(
      n_groups, 1, figsize=(16, 3.2 * n_groups), squeeze=False, sharex=True
  )
  for g, (title, joints) in enumerate(consts.JOINT_GROUPS):
    ax = axes[g, 0]
    for idx, label in joints:
      ax.plot(times, force[:, idx], lw=1.5, label=label)
    ax.set_ylabel("Torque (Nm)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(fontsize=9, ncol=len(joints), loc="upper right")
    ax.grid(True, alpha=0.3)
  axes[-1, 0].set_xlabel("Time (s)")
  fig.suptitle("Actuator Forces", fontsize=16, fontweight="bold")
  fig.tight_layout(rect=[0, 0, 1, 0.97])
  path = save_dir / "sim2sim_actuator_force.png"
  fig.savefig(str(path), dpi=150)
  plt.close(fig)
  print(f"  Saved: {path}")


def plot_sim2sim_results(recorded: dict, save_dir: Path) -> None:
  """Generate all sim2sim tracking plots."""
  if not recorded:
    print("No data recorded, skipping plots.")
    return
  save_dir.mkdir(parents=True, exist_ok=True)
  n = recorded["n_steps"]
  times = np.arange(n) * CTRL_DT
  print(f"\nGenerating sim2sim plots ({n} steps, {times[-1]:.2f}s) ...")
  _plot_qpos_tracking(
      recorded["ref_qpos"], recorded["actual_qpos"], times, save_dir
  )
  _plot_contact_force_tracking(
      recorded["ref_cf"], recorded["actual_cf"], times, save_dir
  )
  _plot_contact_force(recorded["actual_cf"], times, save_dir)
  _plot_actuator_force(recorded["actuator_force"], times, save_dir)
  print(f"All plots saved to {save_dir}/")


# ============================================================================
# Main
# ============================================================================

_controller: OnnxController | None = None


def load_callback(model=None, data=None):
  """Viewer loader: build model, reset to home keyframe, attach controller."""
  global _controller
  mujoco.set_mjcb_control(None)

  # Load model and set timestep
  model = mujoco.MjModel.from_xml_string(
      epath.Path(consts.SCENE_XML).read_text(), assets=get_assets()
  )
  model.opt.timestep = SIM_DT
  data = mujoco.MjData(model)

  # Reset to XML "home" keyframe as initial pose
  mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

  # Load trajectory and normalizer
  traj = _load_trajectory()
  norm = np.load(epath.Path(_onnx_dir / ONNX_NORM_NAME).as_posix())

  _controller = OnnxController(
      policy_path=(_onnx_dir / ONNX_POLICY_NAME).as_posix(),
      model=model,
      traj=traj,
      obs_mean=norm["obs_mean"],
      obs_std=norm["obs_std"],
  )
  mujoco.set_mjcb_control(_controller.get_control)
  return model, data


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Deploy XleoMassage2 ONNX policy in C MuJoCo viewer"
  )
  parser.add_argument(
      "--model_dir",
      type=str,
      default=None,
      help=(
          f"Directory containing {ONNX_POLICY_NAME} and {ONNX_NORM_NAME}. "
          "Default: sim2sim/onnx/"
      ),
  )
  args = parser.parse_args()

  _onnx_dir = _DEFAULT_ONNX_DIR
  if args.model_dir is not None:
    _onnx_dir = epath.Path(Path(args.model_dir).resolve())

  viewer.launch(loader=load_callback)

  # After viewer closes, generate tracking plots
  if _controller is not None:
    plot_sim2sim_results(_controller.get_recorded_data(), PLOT_DIR)
