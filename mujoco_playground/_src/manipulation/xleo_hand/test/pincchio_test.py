import os
import sys

# ROS Jazzy pollutes PYTHONPATH and LD_LIBRARY_PATH with paths that contain
# an old pinocchio/eigenpy compiled against NumPy 1.x, causing crashes.
# We clean them and re-exec if needed, so the dynamic linker sees only the
# uv-installed libraries.
_ROS_FILTER = "/opt/ros"

def _needs_clean() -> bool:
    for var in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        val = os.environ.get(var, "")
        if _ROS_FILTER in val:
            return True
    return any(_ROS_FILTER in p for p in sys.path)

if _needs_clean():
    for var in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        val = os.environ.get(var, "")
        cleaned = ":".join(p for p in val.split(":") if p and _ROS_FILTER not in p)
        os.environ[var] = cleaned
    os.execv(sys.executable, [sys.executable] + sys.argv)

import time
from pathlib import Path

import numpy as np
import pinocchio

def main():
    # URDF file path
    urdf_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "urdf"
        / "ftl_xleo_dual_hand.urdf"
    )
    urdf_filename = str(urdf_path)
    print(f"URDF: {urdf_filename}")

    # Build model with floating base (free-flyer)
    model = pinocchio.buildModelFromUrdf(urdf_filename, pinocchio.JointModelFreeFlyer())
    data = model.createData()

    # Build model with fixed base
    model_fixed = pinocchio.buildModelFromUrdf(urdf_filename)
    data_fixed = model_fixed.createData()

    # Print model info
    print(f"model name: {model.name}")
    print(f"pino_model.nq: {model.nq}")
    print(f"pino_model.nv: {model.nv}")
    print(f"pino_model.njoints: {model.njoints}")
    print(f"pino_model.nbodies: {model.nbodies}")

    # Calculate the Mass Matrix
    q = pinocchio.neutral(model)
    q_fixed = pinocchio.neutral(model_fixed)

    start = time.perf_counter()
    pinocchio.crba(model, data, q)
    pinocchio.crba(model_fixed, data_fixed, q_fixed)
    pinocchio.forwardKinematics(model, data, q)
    pinocchio.forwardKinematics(model_fixed, data_fixed, q_fixed)
    pinocchio.centerOfMass(model, data, q)
    end = time.perf_counter()

    print(f"Total Mass: {data.mass[0]}")
    print(f"运行时间：{(end - start) * 1000:.3f}毫秒")

    # Print initial q for floating base model
    print("\nInitial q for floating base model:")
    for i in range(model.nq):
        print(f"{i}\t :\t{q[i]}")

    # Print initial q for fixed base model
    print("Initial q for fixed base model:")
    for i in range(model_fixed.nq):
        print(f"{i} :\t{q_fixed[i]}")

    # Print joint placements for floating base model
    print("\n--- Floating base model joints ---")
    for joint_id in range(model.njoints):
        name = model.names[joint_id]
        trans = data.oMi[joint_id].translation
        print(f"{joint_id:<24} {name} {trans[0]:.3f} {trans[1]:.3f} {trans[2]:.3f}")

    for joint_id in range(model.njoints):
        print(f"{model.names[joint_id]},")

    # Print joint placements for fixed base model
    print("\n--- Fixed base model joints ---")
    for joint_id in range(model_fixed.njoints):
        name = model_fixed.names[joint_id]
        trans = data_fixed.oMi[joint_id].translation
        print(f"{joint_id:<24} {name} {trans[0]:.3f} {trans[1]:.3f} {trans[2]:.3f}")

    for joint_id in range(model_fixed.njoints):
        print(f"{model_fixed.names[joint_id]},")


if __name__ == "__main__":
    main()
