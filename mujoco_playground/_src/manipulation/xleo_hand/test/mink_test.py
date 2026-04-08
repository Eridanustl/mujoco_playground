import logging
import signal
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent.parent
_XML = _HERE / "models" / "xmls" / "ftl_xleo_dual_hand_position.scene.xml"

# IK parameters
SOLVER = "daqp"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 20

# Target positions for each task.
TARGET_POSITIONS = {
    "lf1": np.array([0.151, 0.005, 0.022]),
    "lf2": np.array([0.151, 0.005, -0.023]),
    "rf1": np.array([0.151, -0.005, 0.022]),
    "rf2": np.array([0.151, -0.005, -0.023]),
    "lw": np.array([0.0, 0.10, 0.0]),
    "rw": np.array([0.0, -0.10, 0.0]),
}

# RGBA colors: left=red, right=blue, wrist=green.
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
POSITIVE_JOINT_INIT_ANGLE = np.deg2rad(15)


def main():
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Force R0/L0 joints to only allow positive angles.
    for jname in POSITIVE_JOINTS:
        jid = model.joint(jname).id
        model.jnt_range[jid][0] = 0.0

    configuration = mink.Configuration(model)

    # Configuration limit (hard inequality constraint on joint ranges).
    config_limit = mink.ConfigurationLimit(model=model)

    # Tasks
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
        orientation_cost=[1, 1, 0],
        lm_damping=0.1,
    )
    right_wrist_task = mink.FrameTask(
        frame_name="site_R_WRIST",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=[1, 1, 0],
        lm_damping=0.1,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-3)

    # Single-joint angle tasks for specific finger joints.
    joint_cost = 1.0
    joint_targets = {
        "J_F1_L2": 0.0,
        "J_F2_L2": 0.0,
        "J_F1_R2": 0.0,
        "J_F2_R2": 0.0,
    }
    joint_cost_vec = np.zeros(model.nv)
    joint_target_q = configuration.q.copy()
    for jname, angle in joint_targets.items():
        jid = model.joint(jname).id
        joint_cost_vec[model.jnt_dofadr[jid]] = joint_cost
        joint_target_q[model.jnt_qposadr[jid]] = angle
    joint_angle_task = mink.PostureTask(model=model, cost=joint_cost_vec)
    joint_angle_task.set_target(joint_target_q)

    tasks = {
        "lf1": left_finger1_tip_task,
        "lf2": left_finger2_tip_task,
        "rf1": right_finger1_tip_task,
        "rf2": right_finger2_tip_task,
        "lw": left_wrist_task,
        "rw": right_wrist_task,
        "posture": posture_task,
        "joint_angles": joint_angle_task,
    }
    limits = [config_limit]

    task_name_to_task = {
        "lf1": left_finger1_tip_task,
        "lf2": left_finger2_tip_task,
        "rf1": right_finger1_tip_task,
        "rf2": right_finger2_tip_task,
        "lw": left_wrist_task,
        "rw": right_wrist_task,
    }

    # --- Viewer state ---
    paused = False
    should_reset = False
    step_once = False
    running = True

    def reset_sim():
        """Reset to home keyframe and seed positive joints."""
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        # for jname in POSITIVE_JOINTS:
        #     jid = model.joint(jname).id
        #     data.qpos[model.jnt_qposadr[jid]] = POSITIVE_JOINT_INIT_ANGLE
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

    def key_callback(keycode):
        nonlocal paused, should_reset, step_once, running
        # Space = pause/resume
        if keycode == 32:
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        # Backspace = reset
        elif keycode == 259:
            should_reset = True
            print("Reset")
        # Right arrow = step once (only while paused)
        elif keycode == 262:
            if paused:
                step_once = True
        # Escape or Q = quit
        elif keycode in (256, 81):
            running = False

    # Initialize
    reset_sim()

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        rate = RateLimiter(frequency=200.0, warn=False)

        # Ctrl+C handler: set running flag to False.
        def sigint_handler(sig, frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGINT, sigint_handler)

        while viewer.is_running() and running:
            dt = rate.dt

            # Handle reset request.
            if should_reset:
                reset_sim()
                should_reset = False

            # Step simulation only when not paused (or single-step requested).
            if not paused or step_once:
                step_once = False

                # Set IK targets.
                for key, task in task_name_to_task.items():
                    task.set_target(mink.SE3.from_translation(TARGET_POSITIONS[key]))

                # Solve IK.
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
                # print(
                #     f"data.qpos: {np.array2string(data.qpos, precision=3, suppress_small=True)}"
                # )

            # Visualize target positions as spheres (always, even when paused).
            viewer.user_scn.ngeom = 0
            for i, (key, pos) in enumerate(TARGET_POSITIONS.items()):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.008, 0, 0],
                    pos=pos,
                    mat=np.eye(3).flatten(),
                    rgba=np.array(TARGET_COLORS[key], dtype=np.float32),
                )
            viewer.user_scn.ngeom = len(TARGET_POSITIONS)

            viewer.sync()
            rate.sleep()

        print("\nShutting down...")


if __name__ == "__main__":
    main()
