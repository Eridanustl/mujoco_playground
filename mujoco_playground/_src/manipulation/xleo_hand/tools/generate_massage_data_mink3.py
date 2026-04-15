"""Generate expert trajectory data for dual hand massage using mink IK (v2).

Trajectory description:
  - Left wrist Y:  0.06 → 0.04   (squeeze inward)
  - Right wrist Y: -0.06 → -0.04 (squeeze inward)
  - Wrist orientation (RPY) interpolated between init and end targets.
  - Left index (lf1) X:  0.1136 → 0.0836  (pull back)
  - Right middle (rf2) X: 0.1136 → 0.0836  (pull back)
  - All other fingertip targets stay fixed.

Initial pose matches mink_test.py TARGET_POSITIONS.

Usage:
    python generate_massage_data_mink2.py
    python generate_massage_data_mink2.py --duration 10.0 --data_freq 50.0
    python generate_massage_data_mink2.py --visualize
"""

import argparse
import logging
import pickle
import signal
from dataclasses import dataclass, fields
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from etils import epath
from scipy.spatial.transform import Rotation

import mink

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.xleo_hand import constants as consts

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent.parent
_XML = _HERE / "models" / "xmls" / "ftl_xleo_dual_hand_position.scene.xml"


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration (edit here)
# ═══════════════════════════════════════════════════════════════════════════════

# IK solver parameters.
SOLVER = "daqp"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 20

# Feature switches.
ENABLE_THUMB = False  # 是否启用大拇指 (lf0/rf0) 轨迹规划

# Initial target positions (from mink_test.py).
# Finger targets: [x, y, z] (position only)
# Wrist targets:  [x, y, z, roll, pitch, yaw] (position + euler angles in rad)
INIT_TARGET_POSITIONS = {
    "lf0": np.array([0.0215, 0.0391, 0.1267]),
    "lf1": np.array([0.1136, -0.0347, 0.0393]),
    "lf2": np.array([0.1145, -0.0362, -0.0225]),
    "rf0": np.array([0.0215, -0.0391, 0.1067]),
    "rf1": np.array([0.1145, 0.0362, 0.0025]),
    "rf2": np.array([0.1136, 0.0347, -0.0593]),
    "lw": np.array([0.0, 0.06, 0.0, 0.0, 0.0, 0.0]),
    "rw": np.array([0.0, -0.06, -0.02, 0.0, 0.0, 0.0]),
}

END_TARGET_POSITIONS = {
    "lf0": np.array([0.0215, 0.0391, 0.1267]),
    "lf1": np.array([0.0736, -0.0547, 0.0393]),
    "lf2": np.array([0.0745, -0.0562, -0.0225]),
    "rf0": np.array([0.0215, -0.0391, 0.1067]),
    "rf1": np.array([0.0745, 0.0562, 0.0025]),
    "rf2": np.array([0.0736, 0.0547, -0.0593]),
    "lw": np.array([0.0, 0.04, 0.0, 0.0, 0.0, 0.0]),
    "rw": np.array([0.0, -0.04, -0.02, 0.0, 0.0, 0.0]),
}

# IK task definitions: (key, site_name, position_cost, orientation_cost, is_thumb)
TASK_DEFS = [
    ("lf0", "site_left_f0_tip", 1.0, 0.0, True),
    ("lf1", "site_left_f1_tip", 1.0, 0.0, False),
    ("lf2", "site_left_f2_tip", 1.0, 0.0, False),
    ("rf0", "site_right_f0_tip", 1.0, 0.0, True),
    ("rf1", "site_right_f1_tip", 1.0, 0.0, False),
    ("rf2", "site_right_f2_tip", 1.0, 0.0, False),
    ("lw", "site_L_WRIST", 1.0, [1, 1, 1], False),
    ("rw", "site_R_WRIST", 1.0, [1, 1, 1], False),
]

# RGBA colors for viewer visualization.
TARGET_COLORS = {
    "lf0": [1, 0.5, 0, 0.2],
    "lf1": [1, 0, 0, 0.2],
    "lf2": [1, 0, 0, 0.2],
    "rf0": [0, 0.5, 1, 0.2],
    "rf1": [0, 0, 1, 0.2],
    "rf2": [0, 0, 1, 0.2],
    "lw": [0, 1, 0, 0.2],
    "rw": [0, 1, 0, 0.2],
}

# Keys that have 6D targets (position + rpy).
WRIST_KEYS = {"lw", "rw"}

# Keys for thumb targets.
THUMB_KEYS = {"lf0", "rf0"}

# Joints whose lower bound is overridden to 0 (positive angles only).
POSITIVE_JOINTS = ["J_F1_L0", "J_F2_L0", "J_F1_R0", "J_F2_R0"]

# F1/F2 L2 joints need a positive initial value to avoid singularity.
L2_JOINTS = ["J_F1_L2", "J_F2_L2", "J_F1_R2", "J_F2_R2"]
L2_INIT_ANGLE = np.deg2rad(30)


@dataclass
class MinkMassageConfig:
  period: float = 2.0  # oscillation period (seconds)
  duration: float = 2.0  # recording duration
  data_freq: float = 50.0  # Hz
  warmup_periods: int = 5  # warmup before recording


# ═══════════════════════════════════════════════════════════════════════════════
# IK solver context
# ═══════════════════════════════════════════════════════════════════════════════


class IKContext:
  """Bundles MuJoCo model/data with mink IK solver state."""

  def __init__(self, cfg: MinkMassageConfig):
    self.cfg = cfg
    self.model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    self.data = mujoco.MjData(self.model)

    # Override joint limits.
    for jname in POSITIVE_JOINTS:
      jid = self.model.joint(jname).id
      self.model.jnt_range[jid][0] = 0.0

    # Build mink configuration and tasks.
    self.configuration = mink.Configuration(self.model)
    self.tasks, self.task_map = self._build_tasks()
    self.posture_task = self.tasks["posture"]
    self.limits = [mink.ConfigurationLimit(model=self.model)]
    self.dt = 1.0 / cfg.data_freq

    # Joint index mapping: IK model → consts.JOINT_NAMES.
    self.ik_to_consts_idx = []
    for i in range(self.model.njnt):
      jname = self.model.joint(i).name
      if jname in consts.JOINT_NAME_TO_INDEX:
        self.ik_to_consts_idx.append(
            (self.model.jnt_qposadr[i], consts.JOINT_NAME_TO_INDEX[jname])
        )

  def _build_tasks(self):
    """Create IK tasks from TASK_DEFS table."""
    task_map = {}
    for key, site, pos_cost, ori_cost, is_thumb in TASK_DEFS:
      if is_thumb and not ENABLE_THUMB:
        continue
      task_map[key] = mink.FrameTask(
          frame_name=site,
          frame_type="site",
          position_cost=pos_cost,
          orientation_cost=ori_cost,
          lm_damping=0.1,
      )
    posture_task = mink.PostureTask(model=self.model, cost=1e-3)
    all_tasks = {**task_map, "posture": posture_task}
    return all_tasks, task_map

  def reset(self):
    """Reset to home keyframe and seed L2 joints."""
    mujoco.mj_resetDataKeyframe(
        self.model, self.data, self.model.key("home").id
    )
    for jname in L2_JOINTS:
      jid = self.model.joint(jname).id
      self.data.qpos[self.model.jnt_qposadr[jid]] = L2_INIT_ANGLE
    self.configuration.update(self.data.qpos)
    self.posture_task.set_target_from_configuration(self.configuration)
    mujoco.mj_forward(self.model, self.data)

  def step(self, t: float):
    """Set targets for time t, solve IK, integrate, and step physics."""
    targets = compute_targets(t, self.cfg)
    for key, task in self.task_map.items():
      _set_task_target(task, targets[key], key)

    vel = mink.solve_ik(
        self.configuration,
        self.tasks.values(),
        self.dt,
        SOLVER,
        damping=1e-3,
        limits=self.limits,
    )
    self.configuration.integrate_inplace(vel, self.dt)
    self.data.ctrl = self.configuration.q
    mujoco.mj_step(self.model, self.data)

  def extract_q(self) -> np.ndarray:
    """Extract the 28 joint values from the IK configuration."""
    q = np.zeros(consts.NQ)
    for ik_idx, consts_idx in self.ik_to_consts_idx:
      q[consts_idx] = self.configuration.q[ik_idx]
    return q


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _halfcos(vstart: float, vend: float, omega: float, t: float) -> float:
  """(1-cos)/2 waveform: oscillates between vstart and vend."""
  return vstart + (vend - vstart) * (1.0 - np.cos(omega * t)) / 2.0


def _rpy_to_so3(rpy: np.ndarray) -> "mink.SO3":
  """Convert roll-pitch-yaw (xyz extrinsic) to mink.SO3 quaternion."""
  quat_xyzw = Rotation.from_euler("xyz", rpy).as_quat()  # [x,y,z,w]
  quat_wxyz = np.array(
      [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
  )
  return mink.SO3(wxyz=quat_wxyz)


def _set_task_target(task, target: np.ndarray, key: str):
  """Set IK task target with rotation for wrists, translation-only for fingers."""
  if key in WRIST_KEYS:
    so3 = _rpy_to_so3(target[3:6])
    task.set_target(mink.SE3.from_rotation_and_translation(so3, target[:3]))
  else:
    task.set_target(mink.SE3.from_translation(target))


def _project_root() -> Path:
  p = Path(__file__).resolve().parent
  while p != p.parent:
    if (p / ".git").exists():
      return p
    p = p.parent
  raise RuntimeError("Cannot find project root (no .git found)")


def compute_targets(t: float, cfg: MinkMassageConfig) -> dict:
  """Return interpolated target positions for all tasks at time t.

  Each channel oscillates between INIT and END via half-cosine waveform.
  Finger targets are 3D (xyz). Wrist targets are 6D (xyz + rpy).
  Thumb targets (lf0/rf0) are skipped when ENABLE_THUMB is False.
  """
  omega = 2.0 * np.pi / cfg.period
  targets = {}
  for key, start in INIT_TARGET_POSITIONS.items():
    if not ENABLE_THUMB and key in THUMB_KEYS:
      continue
    end = END_TARGET_POSITIONS[key]
    targets[key] = np.array(
        [_halfcos(start[i], end[i], omega, t) for i in range(len(start))]
    )
  return targets


# ═══════════════════════════════════════════════════════════════════════════════
# FK precomputation
# ═══════════════════════════════════════════════════════════════════════════════


def _get_assets():
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
  """Run MuJoCo CPU FK on each trajectory frame to get body xpos."""
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

  l_wrist_body_id = mj_model.body("L_WRIST").id
  r_wrist_body_id = mj_model.body("R_WRIST").id

  n_left = len(consts.LEFT_BODY_NAMES)
  left_wrist_idx = 0
  right_wrist_idx = n_left
  left_finger_slice = slice(1, n_left)
  right_finger_slice = slice(n_left + 1, len(consts.TRACKED_BODY_NAMES))

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

    all_xpos = mj_data.xpos[tracked_body_ids].copy()

    l_wrist_pos = mj_data.xpos[l_wrist_body_id].copy()
    r_wrist_pos = mj_data.xpos[r_wrist_body_id].copy()
    l_wrist_rot = mj_data.xmat[l_wrist_body_id].reshape(3, 3)
    r_wrist_rot = mj_data.xmat[r_wrist_body_id].reshape(3, 3)

    tracked_xpos[t, left_wrist_idx] = l_wrist_pos
    tracked_xpos[t, right_wrist_idx] = r_wrist_pos

    tracked_xpos[t, left_finger_slice] = (
        all_xpos[left_finger_slice] - l_wrist_pos
    ) @ l_wrist_rot
    tracked_xpos[t, right_finger_slice] = (
        all_xpos[right_finger_slice] - r_wrist_pos
    ) @ r_wrist_rot

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


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory generation (headless)
# ═══════════════════════════════════════════════════════════════════════════════


def generate(cfg: MinkMassageConfig) -> dict:
  """Solve IK at each timestep and record qpos/qvel trajectories."""
  ctx = IKContext(cfg)
  ctx.reset()

  # Warmup phase.
  warmup_duration = cfg.warmup_periods * cfg.period
  warmup_steps = int(warmup_duration * cfg.data_freq)
  print(
      f"  Warmup: {cfg.warmup_periods} periods "
      f"({warmup_steps} steps, {warmup_duration:.1f}s) ..."
  )
  for step in range(warmup_steps):
    ctx.step(step * ctx.dt)

  # Recording phase.
  t_array = np.arange(0, cfg.duration, ctx.dt)
  T = len(t_array)
  qpos_traj = np.zeros((T, consts.NQ), dtype=np.float64)
  qvel_traj = np.zeros((T, consts.NQ), dtype=np.float64)

  prev_q = ctx.extract_q()
  for step, t in enumerate(t_array):
    ctx.step(t)

    q_frame = ctx.extract_q()
    qpos_traj[step] = q_frame
    qvel_traj[step] = (q_frame - prev_q) / ctx.dt
    prev_q = q_frame.copy()

    if (step + 1) % int(cfg.data_freq) == 0:
      print(f"  IK solved: {step + 1}/{T} frames ({t:.1f}s)")

  loop_err = np.max(np.abs(qpos_traj[0] - qpos_traj[-1]))
  print(f"  Loop quality: max |qpos[0] - qpos[-1]| = {loop_err:.6f} rad")

  return {
      "qpos": qpos_traj,
      "qvel": qvel_traj,
      "data_freq": cfg.data_freq,
      "duration": cfg.duration,
  }


# ═══════════════════════════════════════════════════════════════════════════════
# Visualize mode (interactive viewer)
# ═══════════════════════════════════════════════════════════════════════════════


def visualize(cfg: MinkMassageConfig):
  """Run IK in an interactive viewer for previewing the trajectory."""
  from loop_rate_limiters import RateLimiter

  ctx = IKContext(cfg)

  paused = False
  should_reset = False
  step_once = False
  running = True
  sim_time = 0.0

  def reset():
    nonlocal sim_time
    ctx.reset()
    sim_time = 0.0

  def key_callback(keycode):
    nonlocal paused, should_reset, step_once, running
    if keycode == 32:  # Space
      paused = not paused
      print(f"{'Paused' if paused else 'Resumed'}")
    elif keycode == 259:  # Backspace
      should_reset = True
      print("Reset")
    elif keycode == 262:  # Right arrow
      if paused:
        step_once = True
    elif keycode in (256, 81):  # Escape / Q
      running = False

  reset()

  with mujoco.viewer.launch_passive(
      model=ctx.model, data=ctx.data, key_callback=key_callback
  ) as viewer:
    mujoco.mjv_defaultFreeCamera(ctx.model, viewer.cam)
    rate = RateLimiter(frequency=cfg.data_freq, warn=False)

    def sigint_handler(sig, frame):
      nonlocal running
      running = False

    signal.signal(signal.SIGINT, sigint_handler)

    while viewer.is_running() and running:
      if should_reset:
        reset()
        should_reset = False

      if not paused or step_once:
        step_once = False
        ctx.step(sim_time)
        sim_time += ctx.dt

      # Draw target spheres.
      viewer.user_scn.ngeom = 0
      targets = compute_targets(sim_time, cfg)
      geom_count = 0
      for key, val in targets.items():
        if key not in TARGET_COLORS:
          continue
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_count],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.008, 0, 0],
            pos=val[:3],
            mat=np.eye(3).flatten(),
            rgba=np.array(TARGET_COLORS[key], dtype=np.float32),
        )
        geom_count += 1
      viewer.user_scn.ngeom = geom_count

      viewer.sync()
      rate.sleep()

    print("\nShutting down...")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
  parser = argparse.ArgumentParser(
      description="Generate massage expert trajectory data via mink IK (v2)"
  )
  for f in fields(MinkMassageConfig):
    ftype = float if f.type == "float" else int
    parser.add_argument(
        f"--{f.name}",
        type=ftype,
        default=f.default,
        help=f"{f.name} (default: {f.default})",
    )
  default_output = _project_root() / "data" / "massage_data_mink2.pkl"
  parser.add_argument(
      "--output",
      type=str,
      default=str(default_output),
      help="Output pkl path (default: data/massage_data_mink2.pkl)",
  )
  parser.add_argument(
      "--visualize",
      action="store_true",
      help="Preview trajectory in interactive viewer (no file output)",
  )
  args = parser.parse_args()

  cfg = MinkMassageConfig(
      **{f.name: getattr(args, f.name) for f in fields(MinkMassageConfig)}
  )

  if args.visualize:
    visualize(cfg)
    return

  print(f"Generating mink IK trajectory (v2): {cfg}")
  data = generate(cfg)

  print("Running FK precomputation for body positions...")
  data = precompute_body_xpos(data)
  print(
      f"  tracked_body_xpos: {data['tracked_body_xpos'].shape}, "
      f"key_body_xpos: {data['key_body_xpos'].shape}"
  )

  T = data["qpos"].shape[0]
  n_contact_bodies = len(consts.CONTACT_FORCE_BODY_NAMES)
  data["tracked_contact_force"] = np.zeros((T, n_contact_bodies, 3))
  data["contact_body_names"] = list(consts.CONTACT_FORCE_BODY_NAMES)

  out_path = Path(args.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, "wb") as fout:
    pickle.dump(data, fout)

  print(f"Saved {out_path}: {T} frames, {cfg.data_freq} Hz, {cfg.duration}s")

  try:
    plot_dir = _project_root() / "data" / "plots"
    print(f"Plotting joint trajectories to {plot_dir} ...")
    from mujoco_playground._src.manipulation.xleo_hand.tools.plot_utils import (
        plot_all,
    )

    plot_all(data, plot_dir, prefix=f"{int(cfg.data_freq)}hz_mink2_")
  except ImportError:
    print("plot_utils not available, skipping plots.")


if __name__ == "__main__":
  main()
