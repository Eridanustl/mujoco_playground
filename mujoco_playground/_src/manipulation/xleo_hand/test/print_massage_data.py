"""Print summary and sample data from massage_data.pkl."""

import pickle
import sys
from pathlib import Path

import numpy as np
from mujoco_playground._src.manipulation.xleo_hand import constants as consts

# Joint index -> name mapping (from generate_massage_data.py)
JOINT_NAMES = {
    1: "wrist_L_Y",
    6: "J_F0_L0 (thumb_L)",
    9: "J_F1_L0 (index_L)",
    12: "J_F2_L0 (middle_L)",
    16: "wrist_R_Y",
    21: "J_F0_R0 (thumb_R)",
    24: "J_F1_R0 (index_R)",
    27: "J_F2_R0 (middle_R)",
}


def _jname(j: int) -> str:
  return JOINT_NAMES.get(j, f"joint_{j}")


def _find_data_file() -> Path:
  """Walk up from this file to find project root, then locate data/massage_data.pkl."""
  p = Path(__file__).resolve().parent
  while p != p.parent:
    candidate = p / "data" / "massage_data_mink.pkl"
    if candidate.exists():
      return candidate
    p = p.parent
  return None


def main():
  path = None
  if len(sys.argv) > 1:
    path = Path(sys.argv[1])
  else:
    path = _find_data_file()

  if path is None or not path.exists():
    print(f"Error: data file not found: {path}")
    sys.exit(1)

  with open(path, "rb") as f:
    data = pickle.load(f)

  print(f"=== Massage Data: {path.name} ===\n")

  # Basic info
  print(f"Keys: {list(data.keys())}")
  print(f"data_freq: {data.get('data_freq')} Hz")
  print(f"duration:  {data.get('duration')} s")

  # Config
  cfg = data.get("config", {})
  if cfg:
    print(f"\nConfig:")
    for k, v in cfg.items():
      print(f"  {k}: {v}")

  # qpos / qvel shape and stats
  for name in ("qpos", "qvel"):
    arr = data.get(name)
    if arr is None:
      continue
    print(f"\n--- {name} ---")
    print(f"  shape: {arr.shape}  dtype: {arr.dtype}")
    print(f"  min:   {arr.min():.6f}")
    print(f"  max:   {arr.max():.6f}")
    print(f"  mean:  {arr.mean():.6f}")

    # Per-joint stats for non-zero joints
    print(f"\n  Non-zero joints:")
    for j in range(arr.shape[1]):
      col = arr[:, j]
      if np.any(col != 0):
        print(
            f"    joint {j:2d} ({_jname(j):20s}): min={col.min():+.4f} "
            f" max={col.max():+.4f}  mean={col.mean():+.4f}"
        )

  # Print first and last few frames
  qpos = data.get("qpos")
  if qpos is not None:
    n = min(5, qpos.shape[0])
    print(f"\n--- First {n} frames (qpos, non-zero joints only) ---")
    nz_joints = [j for j in range(qpos.shape[1]) if np.any(qpos[:, j] != 0)]
    header = "frame  " + "  ".join(f"{_jname(j):>8s}" for j in nz_joints)
    print(f"  {header}")
    for i in range(n):
      vals = "  ".join(f"{qpos[i, j]:>+8.4f}" for j in nz_joints)
      print(f"  {i:5d}  {vals}")

    print(f"\n--- Last {n} frames (qpos, non-zero joints only) ---")
    print(f"  {header}")
    for i in range(qpos.shape[0] - n, qpos.shape[0]):
      vals = "  ".join(f"{qpos[i, j]:>+8.4f}" for j in nz_joints)
      print(f"  {i:5d}  {vals}")


if __name__ == "__main__":
  main()
