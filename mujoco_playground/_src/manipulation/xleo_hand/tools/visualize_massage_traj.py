"""Kinematic trajectory visualizer for massage trajectories.

Directly writes joint positions from a trajectory pkl into data.qpos,
bypassing physics simulation and actuator control. Useful for quickly
checking whether a generated trajectory is correct.

Usage:
    python visualize_massage_traj.py
    python visualize_massage_traj.py --pkl_path path/to/traj.pkl
    python visualize_massage_traj.py --speed 2.0
"""

import argparse
import pickle
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_playground._src.manipulation.xleo_hand import constants as consts


def _project_root() -> Path:
  """Walk up from this file to find the project root (directory containing .git)."""
  p = Path(__file__).resolve().parent
  while p != p.parent:
    if (p / ".git").exists():
      return p
    p = p.parent
  raise RuntimeError("Cannot find project root (no .git found)")


def _interpolate(data: np.ndarray, t: float, data_freq: float) -> np.ndarray:
  """Linearly interpolate a (T, D) array at continuous time t."""
  idx = t * data_freq
  T = data.shape[0]
  i0 = int(idx)
  if i0 >= T - 1:
    return data[-1]
  i1 = i0 + 1
  alpha = idx - i0
  return (1.0 - alpha) * data[i0] + alpha * data[i1]


def run(pkl_path: str, speed: float):
  # Load trajectory data.
  with open(pkl_path, "rb") as f:
    traj = pickle.load(f)

  qpos_data = traj["qpos"]  # (T, NQ)
  data_freq = traj["data_freq"]
  duration = traj["duration"]

  print(f"Trajectory: {qpos_data.shape[0]} frames, "
        f"data_freq={data_freq} Hz, duration={duration}s")
  print(f"Playback speed: {speed}x")

  # Load scene model.
  scene_xml = str(
      _project_root()
      / "mujoco_playground"
      / "_src"
      / "manipulation"
      / "xleo_hand"
      / "models"
      / "xmls"
      / "ftl_xleo_dual_hand.scene.xml"
  )
  model = mujoco.MjModel.from_xml_path(scene_xml)
  data = mujoco.MjData(model)

  # Resolve joint qpos/qvel addresses.
  qpos_adrs = []
  qvel_adrs = []
  for jname in consts.JOINT_NAMES:
    jid = model.joint(jname).id
    qpos_adrs.append(model.jnt_qposadr[jid])
    qvel_adrs.append(model.jnt_dofadr[jid])

  # Use launch_passive so we control the loop ourselves (no mj_step).
  print("Launching viewer (close window or Ctrl+C to exit)...")
  wall_start = time.monotonic()

  with mujoco.viewer.launch_passive(model, data) as viewer:
    # Match simulate's initial view: azimuth/elevation from <global>.
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -40
    viewer.cam.distance = 0.6
    viewer.cam.lookat[:] = [0.05, 0, 0]

    # Only show group 0 (visual) and 1 (floor), hide collision groups.
    viewer.opt.geomgroup[0] = True
    viewer.opt.geomgroup[1] = True
    viewer.opt.geomgroup[2] = False
    viewer.opt.geomgroup[3] = False

    while viewer.is_running():
      wall_now = time.monotonic() - wall_start
      t = (wall_now * speed) % duration
      qpos_target = _interpolate(qpos_data, t, data_freq)

      # Write joint positions directly into qpos.
      for i in range(consts.NQ):
        data.qpos[qpos_adrs[i]] = qpos_target[i]
        # Zero out velocities to prevent physics integration drift.
        data.qvel[qvel_adrs[i]] = 0.0

      # Forward kinematics only (no physics step).
      mujoco.mj_forward(model, data)

      # Sync viewer and sleep to roughly 60 fps.
      viewer.sync()
      time.sleep(max(0, 1.0 / 60.0 - (time.monotonic() - wall_start - wall_now)))


def main():
  parser = argparse.ArgumentParser(
      description="Kinematic visualizer for massage trajectories"
  )
  default_pkl = str(_project_root() / "data" / "massage_traj.pkl")
  parser.add_argument(
      "--pkl_path",
      type=str,
      default=default_pkl,
      help="Path to trajectory pkl (default: data/massage_traj.pkl)",
  )
  parser.add_argument(
      "--speed",
      type=float,
      default=1.0,
      help="Playback speed multiplier (default: 1.0)",
  )
  args = parser.parse_args()
  run(args.pkl_path, args.speed)


if __name__ == "__main__":
  main()
