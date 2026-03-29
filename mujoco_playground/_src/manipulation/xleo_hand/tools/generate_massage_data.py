"""Generate expert trajectory data for dual hand massage motion.

Produces (1-cos)/2 driven qpos/qvel for all joints at a configurable
sampling frequency and saves to a pickle file.

Usage:
    python generate_massage_data.py --duration 10.0
    python generate_massage_data.py --duration 20.0 --data_freq 240.0 --output my_data.pkl
"""

import argparse
import pickle
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import mujoco
import numpy as np
from etils import epath

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.xleo_hand import constants as consts

# ---------------------------------------------------------------------------
# Joint trajectory configs: (joint_name, amplitude, sign)
#   All use (1-cos)/2 waveform: 0 -> amplitude -> 0
#   sign: +1 or -1 controls the direction
# ---------------------------------------------------------------------------

# Wrist Y-axis (use wrist omega/phase, amplitude from config)
#   J_LINK_HAND_BASE_L_Y  -> -Y is inward squeeze
#   J_LINK_HAND_BASE_R_Y  -> +Y is inward squeeze
WRIST_JOINTS = [
    # (joint_name, sign)
    ("J_LINK_HAND_BASE_L_Y", -1),  # left wrist Y: negative = inward
    ("J_LINK_HAND_BASE_R_Y", +1),  # right wrist Y: positive = inward
]

# Finger joints (use finger omega/phase)
#   Each entry: (joint_name, amplitude)
FINGER_JOINTS = [
    # F0 (thumb) - separate amplitudes
    ("J_F0_L0", 0.3),
    ("J_F0_L1", 0.4),
    ("J_F0_R0", 0.3),
    ("J_F0_R1", 0.4),
    # F1
    ("J_F1_L0", 0.7),
    ("J_F1_R0", 0.7),
    # F2
    ("J_F2_L0", 0.7),
    ("J_F2_R0", 0.7),
]


# ---------------------------------------------------------------------------
# Project root helper
# ---------------------------------------------------------------------------


def _project_root() -> Path:
  """Walk up from this file to find the project root (directory containing .git)."""
  p = Path(__file__).resolve().parent
  while p != p.parent:
    if (p / ".git").exists():
      return p
    p = p.parent
  raise RuntimeError("Cannot find project root (no .git found)")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MassageConfig:
  # Wrist Y-axis
  wrist_amplitude: float = 0.04
  wrist_period: float = 2.0
  wrist_phase: float = 0.0
  # Finger base joints
  finger_amplitude: float = 0.7  # default for non-thumb fingers
  finger_period: float = 2.0
  finger_phase: float = 0.0
  # Duration & sampling
  duration: float = 2.0
  data_freq: float = 120.0


# ---------------------------------------------------------------------------
# (1-cos)/2 waveform helpers
# ---------------------------------------------------------------------------


def _halfcos_pos(
    amp: float, omega: float, t: np.ndarray, phase: float
) -> np.ndarray:
  """qpos = amp * (1 - cos(omega*t + phase)) / 2"""
  return amp * (1.0 - np.cos(omega * t + phase)) / 2.0


def _halfcos_vel(
    amp: float, omega: float, t: np.ndarray, phase: float
) -> np.ndarray:
  """qvel = d/dt qpos = amp * omega * sin(omega*t + phase) / 2"""
  return amp * omega * np.sin(omega * t + phase) / 2.0


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------


def generate(cfg: MassageConfig) -> dict:
  t = np.arange(0, cfg.duration, 1.0 / cfg.data_freq)
  T = len(t)
  qpos = np.zeros((T, consts.NQ), dtype=np.float64)
  qvel = np.zeros((T, consts.NQ), dtype=np.float64)

  omega_w = 2.0 * np.pi / cfg.wrist_period
  omega_f = 2.0 * np.pi / cfg.finger_period

  # Wrist joints
  for name, sign in WRIST_JOINTS:
    idx = consts.joint_index(name)
    qpos[:, idx] = sign * _halfcos_pos(
        cfg.wrist_amplitude, omega_w, t, cfg.wrist_phase
    )
    qvel[:, idx] = sign * _halfcos_vel(
        cfg.wrist_amplitude, omega_w, t, cfg.wrist_phase
    )

  # Finger joints
  for name, amp in FINGER_JOINTS:
    idx = consts.joint_index(name)
    qpos[:, idx] = _halfcos_pos(amp, omega_f, t, cfg.finger_phase)
    qvel[:, idx] = _halfcos_vel(amp, omega_f, t, cfg.finger_phase)

  return {
      "qpos": qpos,
      "qvel": qvel,
      "data_freq": cfg.data_freq,
      "duration": cfg.duration,
      "config": asdict(cfg),
  }


# ---------------------------------------------------------------------------
# FK precomputation
# ---------------------------------------------------------------------------


def _get_assets():
  """Load model assets (meshes, XMLs) for MuJoCo model construction."""
  assets = {}
  mjx_env.update_assets(
      assets, consts.ROOT_PATH / "models" / "ftl_meshes", "*.stl"
  )
  mjx_env.update_assets(
      assets, consts.ROOT_PATH / "models" / "ftl_meshes", "*.STL"
  )
  mjx_env.update_assets(assets, consts.ROOT_PATH / "models" / "xmls", "*.xml")
  convex_dir = epath.Path(
      consts.ROOT_PATH / "models" / "ftl_meshes" / "convex_new"
  )
  for f in convex_dir.glob("*.stl"):
    assets[f"convex_new/{f.name}"] = f.read_bytes()
  return assets


def precompute_body_xpos(data: dict) -> dict:
  """Run MuJoCo CPU FK on each trajectory frame to get body xpos.

  Wrist positions (L_WRIST, R_WRIST) are stored in the world frame.
  Finger body positions are stored relative to their respective wrist's
  local coordinate frame:  p_local = R_wrist^T @ (p_world - p_wrist).

  Adds tracked_body_xpos, key_body_xpos, and body name lists to the data dict.

  Args:
    data: dict with "qpos" (T, 28), "data_freq", etc.

  Returns:
    The same dict with new keys added.
  """
  assets = _get_assets()
  mj_model = mujoco.MjModel.from_xml_string(
      epath.Path(consts.SCENE_XML.as_posix()).read_text(), assets=assets
  )
  mj_data = mujoco.MjData(mj_model)

  joint_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
  tracked_body_ids = np.array(
      [mj_model.body(n).id for n in consts.TRACKED_BODY_NAMES]
  )
  key_body_ids = np.array([mj_model.body(n).id for n in consts.KEY_BODY_NAMES])

  # Wrist body IDs for coordinate transformation
  l_wrist_body_id = mj_model.body("L_WRIST").id
  r_wrist_body_id = mj_model.body("R_WRIST").id

  # Index mapping within TRACKED_BODY_NAMES:
  #   LEFT_BODY_NAMES:  [0]=L_WRIST, [1..9]=left finger bodies
  #   RIGHT_BODY_NAMES: [10]=R_WRIST, [11..19]=right finger bodies
  n_left = len(consts.LEFT_BODY_NAMES)  # 10
  left_wrist_idx = 0
  right_wrist_idx = n_left  # 10
  left_finger_slice = slice(1, n_left)  # 1..9
  right_finger_slice = slice(
      n_left + 1, len(consts.TRACKED_BODY_NAMES)
  )  # 11..19

  # KEY_BODY_NAMES: first 3 are left fingertips, last 3 are right fingertips
  n_left_key = len([n for n in consts.KEY_BODY_NAMES if "_L" in n])

  qpos_data = data["qpos"]
  T = qpos_data.shape[0]
  tracked_xpos = np.zeros((T, len(tracked_body_ids), 3))
  key_xpos = np.zeros((T, len(key_body_ids), 3))

  for t in range(T):
    mj_data.qpos[:] = mj_model.qpos0
    mj_data.qpos[joint_qids] = qpos_data[t]
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    # World-frame positions for all tracked bodies
    all_xpos = mj_data.xpos[tracked_body_ids].copy()

    # Wrist positions (world frame) and rotation matrices (3x3)
    l_wrist_pos = mj_data.xpos[l_wrist_body_id].copy()
    r_wrist_pos = mj_data.xpos[r_wrist_body_id].copy()
    l_wrist_rot = mj_data.xmat[l_wrist_body_id].reshape(3, 3)
    r_wrist_rot = mj_data.xmat[r_wrist_body_id].reshape(3, 3)

    # Keep wrist positions in world frame
    tracked_xpos[t, left_wrist_idx] = l_wrist_pos
    tracked_xpos[t, right_wrist_idx] = r_wrist_pos

    # Left finger bodies: transform to left wrist local frame
    left_world = all_xpos[left_finger_slice]  # (9, 3)
    tracked_xpos[t, left_finger_slice] = (
        left_world - l_wrist_pos
    ) @ l_wrist_rot  # R^T @ delta = delta @ R

    # Right finger bodies: transform to right wrist local frame
    right_world = all_xpos[right_finger_slice]  # (9, 3)
    tracked_xpos[t, right_finger_slice] = (
        right_world - r_wrist_pos
    ) @ r_wrist_rot

    # Key body xpos (fingertips): also in wrist-local frame
    all_key_xpos = mj_data.xpos[key_body_ids].copy()
    key_xpos[t, :n_left_key] = (
        all_key_xpos[:n_left_key] - l_wrist_pos
    ) @ l_wrist_rot
    key_xpos[t, n_left_key:] = (
        all_key_xpos[n_left_key:] - r_wrist_pos
    ) @ r_wrist_rot

  data["tracked_body_xpos"] = tracked_xpos
  data["key_body_xpos"] = key_xpos
  data["tracked_body_names"] = list(consts.TRACKED_BODY_NAMES)
  data["key_body_names"] = list(consts.KEY_BODY_NAMES)
  return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
  parser = argparse.ArgumentParser(
      description="Generate massage expert trajectory data"
  )
  for f in fields(MassageConfig):
    parser.add_argument(
        f"--{f.name}",
        type=float,
        default=f.default,
        help=f"{f.name} (default: {f.default})",
    )
  default_output = _project_root() / "data" / "massage_data.pkl"
  parser.add_argument(
      "--output",
      type=str,
      default=str(default_output),
      help="Output pkl path (default: data/massage_data.pkl)",
  )
  args = parser.parse_args()

  cfg = MassageConfig(
      **{f.name: getattr(args, f.name) for f in fields(MassageConfig)}
  )
  data = generate(cfg)

  # Pre-compute body positions via CPU FK.
  print("Running FK precomputation for body positions...")
  data = precompute_body_xpos(data)
  print(
      f"  tracked_body_xpos: {data['tracked_body_xpos'].shape}, "
      f"key_body_xpos: {data['key_body_xpos'].shape}"
  )

  with open(args.output, "wb") as fout:
    pickle.dump(data, fout)

  T = data["qpos"].shape[0]
  print(f"Saved {args.output}: {T} frames, {cfg.data_freq} Hz, {cfg.duration}s")

  # Plot trajectories and save to data/ folder
  plot_dir = _project_root() / "data" / "plots"
  print(f"Plotting joint trajectories to {plot_dir} ...")
  from mujoco_playground._src.manipulation.xleo_hand.tools.plot_utils import plot_all

  freq = data["data_freq"]
  plot_all(data, plot_dir, prefix=f"{int(freq)}hz_origin_")


if __name__ == "__main__":
  main()
