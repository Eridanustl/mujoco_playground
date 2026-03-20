import argparse
import pickle
import signal
from pathlib import Path

import mujoco
import mujoco.viewer


def _project_root() -> Path:
  """Walk up from this file to find the project root (directory containing .git)."""
  p = Path(__file__).resolve().parent
  while p != p.parent:
    if (p / ".git").exists():
      return p
    p = p.parent
  raise RuntimeError("Cannot find project root (no .git found)")


def main():
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

  print("ID and Body names")
  for i in range(model.nbody):
    print(f"id: {i:>2}: {model.body(i).name:<15}")

  print("\nID and Joint names")
  for i in range(model.nbody):
    print(f"id: {i:>2}: {model.joint(i).name:<25}")

  mujoco.mj_forward(model, data)
  print("\nID Body names and xpos")
  for i in range(model.nbody):
    print(
        f"id: {i:>2}: {model.body(i).name:<15} xpos: [{data.xpos[i][0]:6.4f},"
        f" {data.xpos[i][1]:7.4f}, {data.xpos[i][2]:6.4f}]"
    )


if __name__ == "__main__":
  main()
