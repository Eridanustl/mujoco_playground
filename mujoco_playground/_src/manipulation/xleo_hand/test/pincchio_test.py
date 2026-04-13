"""Forward-kinematics test for the XLEO dual hand using MuJoCo.

Computes fingertip positions in world frame for a given qpos.
"""
from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

# ── XML path ─────────────────────────────────────────────────────────────────
XML_PATH = (
    Path(__file__).resolve().parent.parent
    / "models" / "xmls" / "ftl_xleo_dual_hand.xml"
)

# ── Fingertip definitions ────────────────────────────────────────────────────
# All fingertips use tip sites defined in the MuJoCo XML with a 5 cm offset
# from the last link frame.
#
# Format: (human-readable name, site_or_body_name, is_site)
FINGERTIP_DEFS: list[tuple[str, str, bool]] = [
    ("left_hand_base",   "L_WRIST",            False),  # hand base (body)
    ("left_thumb_tip",   "site_left_f0_tip",   True),   # tip site
    ("left_index_tip",   "site_left_f1_tip",   True),
    ("left_middle_tip",  "site_left_f2_tip",   True),
    ("right_hand_base",  "R_WRIST",            False),  # hand base (body)
    ("right_thumb_tip",  "site_right_f0_tip",  True),   # tip site
    ("right_index_tip",  "site_right_f1_tip",  True),
    ("right_middle_tip", "site_right_f2_tip",  True),
]


def _load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the MuJoCo model and create data."""
    xml = str(XML_PATH)
    print(f"XML: {xml}")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    return model, data


def compute_fingertip_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
) -> dict[str, np.ndarray]:
    """Set qpos, run forward kinematics, and return fingertip world positions.

    Args:
        model: MuJoCo model.
        data:  MuJoCo data.
        qpos:  Joint configuration vector (length 28).

    Returns:
        Dict mapping fingertip name -> (3,) world-frame position.
    """
    assert qpos.shape[0] == model.nq, (
        f"qpos length {qpos.shape[0]} != model.nq {model.nq}"
    )
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)  # full forward pass (kinematics + dynamics)

    results: dict[str, np.ndarray] = {}
    for tip_name, mj_name, is_site in FINGERTIP_DEFS:
        if is_site:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, mj_name)
            pos = data.site_xpos[site_id].copy()
        else:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mj_name)
            pos = data.xpos[body_id].copy()
        results[tip_name] = pos

    return results


# ── Model inspection (original functionality) ────────────────────────────────

def inspect_model():
    """Print model info: joints, bodies, mass, joint positions at q=0."""
    model, data = _load_model()

    print(f"model name : {model.opt.timestep}")
    print(f"nq         : {model.nq}")
    print(f"nv         : {model.nv}")
    print(f"njnt       : {model.njnt}")
    print(f"nbody      : {model.nbody}")

    # Set to zero config and run FK
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)

    # Total mass
    total_mass = sum(model.body_mass)
    print(f"Total Mass : {total_mass:.4f}")

    # Print joint names + qpos index
    print("\n--- Joints ---")
    for i in range(model.njnt):
        jnt = model.joint(i)
        print(f"  joint[{i:>2}] qpos[{model.jnt_qposadr[i]:>2}]  {jnt.name}")

    # Print body positions at zero config
    print("\n--- Body positions (q=0) ---")
    for i in range(model.nbody):
        name = model.body(i).name
        pos = data.xpos[i]
        print(f"  body[{i:>2}] {name:<24} x={pos[0]:>8.4f}  y={pos[1]:>8.4f}  z={pos[2]:>8.4f}")


# ── Fingertip FK demo ────────────────────────────────────────────────────────

def demo_fingertip_fk():
    """Compute and print fingertip positions for a given qpos."""
    model, data = _load_model()

    # 28-DOF qpos (fixed base)
    # Left hand  [0..13]:  wrist(5) + F0(3) + F1(3) + F2(3)
    # Right hand [14..27]: wrist(5) + F0(3) + F1(3) + F2(3)
    qpos = np.array([
        # ── left hand ──
        0.0, -0.02, 0, 0, 0,           # left wrist (5 DOF)
        0, 0, 0,                        # left thumb  F0 (3 DOF)
        1.05, 0.2, 0.76,               # left index  F1 (3 DOF)
        1.05, 0, 0.76,                 # left middle F2 (3 DOF)
        # ── right hand ──
        0, 0.02, 0, 0, 0,              # right wrist (5 DOF)
        0, 0, 0,                        # right thumb  F0 (3 DOF)
        1.05, 0, 0.76,                 # right index  F1 (3 DOF)
        1.05, -0.2, 0.76,              # right middle F2 (3 DOF)
    ])

    start = time.perf_counter()
    tips = compute_fingertip_positions(model, data, qpos)
    elapsed = time.perf_counter() - start

    print(f"\n{'=' * 60}")
    print(f"  指尖位置（世界坐标系） — 计算耗时: {elapsed * 1000:.3f} ms")
    print(f"{'=' * 60}")
    print(f"  {'名称':<24} {'X':>10} {'Y':>10} {'Z':>10}")
    print(f"  {'-' * 56}")
    for name, pos in tips.items():
        print(f"  {name:<24} {pos[0]:>10.4f} {pos[1]:>10.4f} {pos[2]:>10.4f}")
    print(f"{'=' * 60}")

    return tips


def main():
    inspect_model()
    demo_fingertip_fk()


if __name__ == "__main__":
    main()
