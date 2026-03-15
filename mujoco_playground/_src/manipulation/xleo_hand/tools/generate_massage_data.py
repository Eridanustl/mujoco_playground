"""Generate expert trajectory data for dual hand massage motion.

Produces (1-cos)/2 driven qpos/qvel for all 30 joints at a configurable
sampling frequency and saves to a pickle file.

Usage:
    python generate_massage_data.py --duration 10.0
    python generate_massage_data.py --duration 20.0 --data_freq 240.0 --output my_data.pkl
"""

import argparse
import pickle
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NUM_JOINTS = 30

# ---------------------------------------------------------------------------
# Joint trajectory configs: (joint_index, amplitude, sign)
#   All use (1-cos)/2 waveform: 0 -> amplitude -> 0
#   sign: +1 or -1 controls the direction
# ---------------------------------------------------------------------------

# Wrist Y-axis (use wrist omega/phase, amplitude from config)
#   idx  1: J_LINK_HAND_BASE_L_Y  -> -Y is inward squeeze
#   idx 16: J_LINK_HAND_BASE_R_Y  -> +Y is inward squeeze
WRIST_JOINTS = [
    # (joint_idx, sign)
    (1, -1),  # left wrist Y: negative = inward
    (16, +1),  # right wrist Y: positive = inward
]

# Finger joints (use finger omega/phase)
#   Each entry: (joint_idx, amplitude)
FINGER_JOINTS = [
    # F0 (thumb) - separate amplitudes
    (6, 0.3),  # J_F0_L0
    (7, 0.4),  # J_F0_L1
    (21, 0.3),  # J_F0_R0
    (22, 0.4),  # J_F0_R1
    # F1
    (9, 0.7),  # J_F1_L0
    (24, 0.7),  # J_F1_R0
    # F2
    (12, 0.7),  # J_F2_L0
    (27, 0.7),  # J_F2_R0
]

# Joint name mapping for plotting
JOINT_NAMES = {
    1: "wrist_L_Y",
    16: "wrist_R_Y",
    6: "J_F0_L0",
    7: "J_F0_L1",
    9: "J_F1_L0",
    12: "J_F2_L0",
    21: "J_F0_R0",
    22: "J_F0_R1",
    24: "J_F1_R0",
    27: "J_F2_R0",
}


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
  wrist_amplitude: float = 0.1
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
  qpos = np.zeros((T, NUM_JOINTS), dtype=np.float64)
  qvel = np.zeros((T, NUM_JOINTS), dtype=np.float64)

  omega_w = 2.0 * np.pi / cfg.wrist_period
  omega_f = 2.0 * np.pi / cfg.finger_period

  # Wrist joints
  for idx, sign in WRIST_JOINTS:
    qpos[:, idx] = sign * _halfcos_pos(
        cfg.wrist_amplitude, omega_w, t, cfg.wrist_phase
    )
    qvel[:, idx] = sign * _halfcos_vel(
        cfg.wrist_amplitude, omega_w, t, cfg.wrist_phase
    )

  # Finger joints
  for idx, amp in FINGER_JOINTS:
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
# Plotting
# ---------------------------------------------------------------------------


def _joint_name(idx: int) -> str:
  return JOINT_NAMES.get(idx, f"joint_{idx}")


def plot_trajectories(data: dict, output_dir: Path) -> None:
  """Plot qpos and qvel curves for each active joint and save to output_dir."""
  output_dir.mkdir(parents=True, exist_ok=True)
  qpos = data["qpos"]
  qvel = data["qvel"]
  freq = data["data_freq"]
  T = qpos.shape[0]
  t = np.arange(T) / freq

  # Identify active joints (any non-zero qpos or qvel)
  active_indices = [
      j
      for j in range(NUM_JOINTS)
      if np.any(qpos[:, j] != 0) or np.any(qvel[:, j] != 0)
  ]

  # Plot each active joint individually
  for j in active_indices:
    name = _joint_name(j)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t, qpos[:, j], color="tab:blue", linewidth=1.5)
    ax1.set_ylabel("Position (rad)", fontsize=12)
    ax1.set_title(f"Joint {j}: {name} - Position", fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, qvel[:, j], color="tab:orange", linewidth=1.5)
    ax2.set_ylabel("Velocity (rad/s)", fontsize=12)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_title(f"Joint {j}: {name} - Velocity", fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"joint_{j}_{name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved joint_{j}_{name}.png")

  # Plot all active joints together in one overview figure
  n = len(active_indices)
  fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n), sharex=True)
  if n == 1:
    axes = axes.reshape(1, -1)

  for row, j in enumerate(active_indices):
    name = _joint_name(j)
    axes[row, 0].plot(t, qpos[:, j], color="tab:blue", linewidth=1.2)
    axes[row, 0].set_ylabel("rad", fontsize=10)
    axes[row, 0].set_title(f"{name} - Position", fontsize=11)
    axes[row, 0].grid(True, alpha=0.3)

    axes[row, 1].plot(t, qvel[:, j], color="tab:orange", linewidth=1.2)
    axes[row, 1].set_ylabel("rad/s", fontsize=10)
    axes[row, 1].set_title(f"{name} - Velocity", fontsize=11)
    axes[row, 1].grid(True, alpha=0.3)

  axes[-1, 0].set_xlabel("Time (s)", fontsize=11)
  axes[-1, 1].set_xlabel("Time (s)", fontsize=11)
  fig.suptitle("Massage Trajectory - All Active Joints", fontsize=14, y=1.0)
  fig.tight_layout()
  fig.savefig(output_dir / "all_joints_overview.png", dpi=150)
  plt.close(fig)
  print(f"  Saved all_joints_overview.png")


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

  with open(args.output, "wb") as fout:
    pickle.dump(data, fout)

  T = data["qpos"].shape[0]
  print(f"Saved {args.output}: {T} frames, {cfg.data_freq} Hz, {cfg.duration}s")

  # Plot trajectories and save to data/ folder
  plot_dir = _project_root() / "data"
  print(f"Plotting joint trajectories to {plot_dir} ...")
  plot_trajectories(data, plot_dir)


if __name__ == "__main__":
  main()
