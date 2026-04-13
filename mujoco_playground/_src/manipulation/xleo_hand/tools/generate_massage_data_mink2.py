"""Generate expert trajectory data for dual hand massage using mink IK (v2).

Trajectory description:
  - Left wrist Y:  0.06 → 0.04   (squeeze inward)
  - Right wrist Y: -0.06 → -0.04 (squeeze inward)
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

import mink

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.xleo_hand import constants as consts

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent.parent
_XML = _HERE / "models" / "xmls" / "ftl_xleo_dual_hand_position.scene.xml"

# IK solver parameters.
SOLVER = "daqp"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 20

# ── Initial target positions (from mink_test.py) ─────────────────────────────
INIT_TARGET_POSITIONS = {
    "lf1": np.array([0.1136, -0.0347, 0.0393]),
    "lf2": np.array([0.1145, -0.0362, -0.0225]),
    "rf1": np.array([0.1145, 0.0362, 0.0025]),
    "rf2": np.array([0.1136, 0.0347, -0.0593]),
    "lw": np.array([0.0, 0.06, 0.0]),
    "rw": np.array([0.0, -0.06, -0.02]),
}

# ── End target positions ──────────────────────────────────────────────────────
END_TARGET_POSITIONS = {
    "lf1": np.array([0.0836, -0.0347, 0.0393]),  # X: 0.1136 → 0.0836
    "lf2": np.array([0.1145, -0.0362, -0.0225]),  # unchanged
    "rf1": np.array([0.1145, 0.0362, 0.0025]),  # unchanged
    "rf2": np.array([0.0836, 0.0347, -0.0593]),  # X: 0.1136 → 0.0836
    "lw": np.array([0.0, 0.04, 0.0]),  # Y: 0.06 → 0.04
    "rw": np.array([0.0, -0.04, -0.02]),  # Y: -0.06 → -0.04
}

# RGBA colors for viewer visualization.
TARGET_COLORS = {
    "lf1": [1, 0, 0, 0.2],
    "lf2": [1, 0, 0, 0.2],
    "rf1": [0, 0, 1, 0.2],
    "rf2": [0, 0, 1, 0.2],
    "lw": [0, 1, 0, 0.2],
    "rw": [0, 1, 0, 0.2],
}

# Joints whose lower bound is overridden to 0 (positive angles only).
POSITIVE_JOINTS = ["J_F1_L0", "J_F2_L0", "J_F1_R0", "J_F2_R0"]

# F1/F2 L2 joints need a positive initial value to avoid singularity.
L2_JOINTS = ["J_F1_L2", "J_F2_L2", "J_F1_R2", "J_F2_R2"]
L2_INIT_ANGLE = np.deg2rad(30)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MinkMassageConfig:
  period: float = 2.0  # oscillation period (seconds)
  duration: float = 2.0  # recording duration
  data_freq: float = 50.0  # Hz
  warmup_periods: int = 5  # warmup before recording


# ---------------------------------------------------------------------------
# Waveform helper
# ---------------------------------------------------------------------------


def _halfcos(vstart: float, vend: float, omega: float, t: float) -> float:
  """(1-cos)/2 waveform: oscillates between vstart and vend."""
  return vstart + (vend - vstart) * (1.0 - np.cos(omega * t)) / 2.0


# ---------------------------------------------------------------------------
# Project root helper
# ---------------------------------------------------------------------------


def _project_root() -> Path:
  p = Path(__file__).resolve().parent
  while p != p.parent:
    if (p / ".git").exists():
      return p
    p = p.parent
  raise RuntimeError("Cannot find project root (no .git found)")


# ---------------------------------------------------------------------------
# Build mink tasks
# ---------------------------------------------------------------------------


def _build_tasks(model, configuration):
  """Create all IK tasks and return (tasks_dict, task_name_to_task, posture_task, limits)."""
  config_limit = mink.ConfigurationLimit(model=model)

  left_finger1_tip_task = mink.FrameTask(
      frame_name="site_left_f1_tip",
      frame_type="site",
      position_cost=1.0,
      orientation_cost=0.0,
      lm_damping=0.1,
  )
  left_finger2_tip_task = mink.FrameTask(
      frame_name="site_left_f2_tip",
      frame_type="site",
      position_cost=1.0,
      orientation_cost=0.0,
      lm_damping=0.1,
  )
  right_finger1_tip_task = mink.FrameTask(
      frame_name="site_right_f1_tip",
      frame_type="site",
      position_cost=1.0,
      orientation_cost=0.0,
      lm_damping=0.1,
  )
  right_finger2_tip_task = mink.FrameTask(
      frame_name="site_right_f2_tip",
      frame_type="site",
      position_cost=1.0,
      orientation_cost=0.0,
      lm_damping=0.1,
  )
  left_wrist_task = mink.FrameTask(
      frame_name="site_L_WRIST",
      frame_type="site",
      position_cost=1.0,
      orientation_cost=[1, 1, 1],
      lm_damping=0.1,
  )
  right_wrist_task = mink.FrameTask(
      frame_name="site_R_WRIST",
      frame_type="site",
      position_cost=1.0,
      orientation_cost=[1, 1, 1],
      lm_damping=0.1,
  )
  posture_task = mink.PostureTask(model=model, cost=1e-3)

  tasks = {
      "lf1": left_finger1_tip_task,
      "lf2": left_finger2_tip_task,
      "rf1": right_finger1_tip_task,
      "rf2": right_finger2_tip_task,
      "lw": left_wrist_task,
      "rw": right_wrist_task,
      "posture": posture_task,
  }
  task_name_to_task = {
      "lf1": left_finger1_tip_task,
      "lf2": left_finger2_tip_task,
      "rf1": right_finger1_tip_task,
      "rf2": right_finger2_tip_task,
      "lw": left_wrist_task,
      "rw": right_wrist_task,
  }
  return tasks, task_name_to_task, posture_task, [config_limit]


# ---------------------------------------------------------------------------
# Compute target positions at time t
# ---------------------------------------------------------------------------


def _compute_targets(t: float, cfg: MinkMassageConfig) -> dict:
  """Return interpolated target positions for all tasks at time t.

  Each target channel oscillates between its INIT and END value using
  a half-cosine waveform with the configured period.
  """
  omega = 2.0 * np.pi / cfg.period
  targets = {}
  for key in INIT_TARGET_POSITIONS:
    start = INIT_TARGET_POSITIONS[key]
    end = END_TARGET_POSITIONS[key]
    # Per-component half-cosine interpolation.
    targets[key] = np.array(
        [_halfcos(start[i], end[i], omega, t) for i in range(3)]
    )
  return targets


# ---------------------------------------------------------------------------
# FK precomputation (reused from generate_massage_data_mink.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# IK trajectory generation (headless)
# ---------------------------------------------------------------------------


def _extract_q(configuration, ik_to_consts_idx) -> np.ndarray:
  """Extract the 28 joint values from the IK configuration."""
  q = np.zeros(consts.NQ)
  for ik_idx, consts_idx in ik_to_consts_idx:
    q[consts_idx] = configuration.q[ik_idx]
  return q


def _reset_sim(model, data, configuration, posture_task):
  """Reset to home keyframe and seed L2 joints."""
  mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
  for jname in L2_JOINTS:
    jid = model.joint(jname).id
    data.qpos[model.jnt_qposadr[jid]] = L2_INIT_ANGLE
  configuration.update(data.qpos)
  posture_task.set_target_from_configuration(configuration)
  mujoco.mj_forward(model, data)


def _step_ik(
    model, configuration, tasks, task_name_to_task, limits, data, t, dt, cfg
):
  """Set targets for time t, solve IK, integrate, and step physics."""
  targets = _compute_targets(t, cfg)
  for key, task in task_name_to_task.items():
    task.set_target(mink.SE3.from_translation(targets[key]))

  vel = mink.solve_ik(
      configuration,
      tasks.values(),
      dt,
      SOLVER,
      damping=1e-3,
      limits=limits,
  )
  configuration.integrate_inplace(vel, dt)

  data.ctrl = configuration.q
  mujoco.mj_step(model, data)


def generate(cfg: MinkMassageConfig) -> dict:
  """Solve IK at each timestep and record qpos/qvel trajectories."""
  model = mujoco.MjModel.from_xml_path(_XML.as_posix())
  data = mujoco.MjData(model)

  for jname in POSITIVE_JOINTS:
    jid = model.joint(jname).id
    model.jnt_range[jid][0] = 0.0

  configuration = mink.Configuration(model)
  tasks, task_name_to_task, posture_task, limits = _build_tasks(
      model, configuration
  )

  _reset_sim(model, data, configuration, posture_task)

  dt = 1.0 / cfg.data_freq

  # Map from IK model joint names to consts.JOINT_NAMES indices.
  ik_to_consts_idx = []
  for i in range(model.njnt):
    jname = model.joint(i).name
    if jname in consts.JOINT_NAME_TO_INDEX:
      ik_to_consts_idx.append(
          (model.jnt_qposadr[i], consts.JOINT_NAME_TO_INDEX[jname])
      )

  # --- Warmup phase ---
  warmup_duration = cfg.warmup_periods * cfg.period
  warmup_steps = int(warmup_duration * cfg.data_freq)
  print(
      f"  Warmup: {cfg.warmup_periods} periods "
      f"({warmup_steps} steps, {warmup_duration:.1f}s) ..."
  )
  for step in range(warmup_steps):
    t = step * dt
    _step_ik(
        model,
        configuration,
        tasks,
        task_name_to_task,
        limits,
        data,
        t,
        dt,
        cfg,
    )

  warmup_last_q = _extract_q(configuration, ik_to_consts_idx)

  # --- Recording phase ---
  t_array = np.arange(0, cfg.duration, dt)
  T = len(t_array)
  qpos_traj = np.zeros((T, consts.NQ), dtype=np.float64)
  qvel_traj = np.zeros((T, consts.NQ), dtype=np.float64)

  prev_q = warmup_last_q
  for step, t in enumerate(t_array):
    _step_ik(
        model,
        configuration,
        tasks,
        task_name_to_task,
        limits,
        data,
        t,
        dt,
        cfg,
    )

    q_frame = _extract_q(configuration, ik_to_consts_idx)
    qpos_traj[step] = q_frame
    qvel_traj[step] = (q_frame - prev_q) / dt
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


# ---------------------------------------------------------------------------
# Visualize mode (interactive viewer)
# ---------------------------------------------------------------------------


def visualize(cfg: MinkMassageConfig):
  """Run IK in an interactive viewer for previewing the trajectory."""
  from loop_rate_limiters import RateLimiter

  model = mujoco.MjModel.from_xml_path(_XML.as_posix())
  data = mujoco.MjData(model)

  for jname in POSITIVE_JOINTS:
    jid = model.joint(jname).id
    model.jnt_range[jid][0] = 0.0

  configuration = mink.Configuration(model)
  tasks, task_name_to_task, posture_task, limits = _build_tasks(
      model, configuration
  )

  paused = False
  should_reset = False
  step_once = False
  running = True
  sim_time = 0.0

  def reset_sim():
    nonlocal sim_time
    _reset_sim(model, data, configuration, posture_task)
    sim_time = 0.0

  def key_callback(keycode):
    nonlocal paused, should_reset, step_once, running
    if keycode == 32:
      paused = not paused
      print(f"{'Paused' if paused else 'Resumed'}")
    elif keycode == 259:
      should_reset = True
      print("Reset")
    elif keycode == 262:
      if paused:
        step_once = True
    elif keycode in (256, 81):
      running = False

  reset_sim()
  dt = 1.0 / cfg.data_freq

  with mujoco.viewer.launch_passive(
      model=model, data=data, key_callback=key_callback
  ) as viewer:
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    rate = RateLimiter(frequency=cfg.data_freq, warn=False)

    def sigint_handler(sig, frame):
      nonlocal running
      running = False

    signal.signal(signal.SIGINT, sigint_handler)

    while viewer.is_running() and running:
      if should_reset:
        reset_sim()
        should_reset = False

      if not paused or step_once:
        step_once = False

        targets = _compute_targets(sim_time, cfg)
        for key, task in task_name_to_task.items():
          task.set_target(mink.SE3.from_translation(targets[key]))

        vel = mink.solve_ik(
            configuration,
            tasks.values(),
            dt,
            SOLVER,
            damping=1e-3,
            limits=limits,
        )
        configuration.integrate_inplace(vel, dt)

        data.ctrl = configuration.q
        mujoco.mj_step(model, data)
        sim_time += dt

      # Draw target spheres.
      viewer.user_scn.ngeom = 0
      targets = _compute_targets(sim_time, cfg)
      for i, (key, pos) in enumerate(targets.items()):
        if key not in TARGET_COLORS:
          continue
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.008, 0, 0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=np.array(TARGET_COLORS[key], dtype=np.float32),
        )
      viewer.user_scn.ngeom = len([k for k in targets if k in TARGET_COLORS])

      viewer.sync()
      rate.sleep()

    print("\nShutting down...")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
