"""Constants for xleo dual hand."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "xleo_hand"
DATA_PATH = mjx_env.ROOT_PATH / ".." / ".." / "data"
SCENE_XML = ROOT_PATH / "models" / "xmls" / "ftl_xleo_dual_hand.scene.xml"

NQ = 28
NV = 28
NU = 28

# Left hand wrist joints (5)
LEFT_WRIST_JOINT_NAMES = [
    "J_LINK_HAND_BASE_L_X",
    "J_LINK_HAND_BASE_L_Y",
    "J_LINK_HAND_BASE_L_ROLL",
    "J_LINK_HAND_BASE_L_PITCH",
    "J_LINK_HAND_BASE_L_YAW",
]

# Left hand finger joints (9)
LEFT_FINGER_JOINT_NAMES = [
    "J_F0_L0",
    "J_F0_L1",
    "J_F0_L2",
    "J_F1_L0",
    "J_F1_L1",
    "J_F1_L2",
    "J_F2_L0",
    "J_F2_L1",
    "J_F2_L2",
]

# Right hand wrist joints (5)
RIGHT_WRIST_JOINT_NAMES = [
    "J_LINK_HAND_BASE_R_X",
    "J_LINK_HAND_BASE_R_Y",
    "J_LINK_HAND_BASE_R_ROLL",
    "J_LINK_HAND_BASE_R_PITCH",
    "J_LINK_HAND_BASE_R_YAW",
]

# Right hand finger joints (9)
RIGHT_FINGER_JOINT_NAMES = [
    "J_F0_R0",
    "J_F0_R1",
    "J_F0_R2",
    "J_F1_R0",
    "J_F1_R1",
    "J_F1_R2",
    "J_F2_R0",
    "J_F2_R1",
    "J_F2_R2",
]


WRIST_NAMES = LEFT_WRIST_JOINT_NAMES + RIGHT_WRIST_JOINT_NAMES
FINGER_NAMES = LEFT_FINGER_JOINT_NAMES + RIGHT_FINGER_JOINT_NAMES
# All joint names (28 total: 5+9 per hand)
JOINT_NAMES = (
    LEFT_WRIST_JOINT_NAMES
    + LEFT_FINGER_JOINT_NAMES
    + RIGHT_WRIST_JOINT_NAMES
    + RIGHT_FINGER_JOINT_NAMES
)

# Left hand actuators (5 wrist + 9 finger)
LEFT_ACTUATOR_NAMES = [
    "M_LINK_HAND_BASE_L_X",
    "M_LINK_HAND_BASE_L_Y",
    "M_LINK_HAND_BASE_L_ROLL",
    "M_LINK_HAND_BASE_L_PITCH",
    "M_LINK_HAND_BASE_L_YAW",
    "M_F0_L0",
    "M_F0_L1",
    "M_F0_L2",
    "M_F1_L0",
    "M_F1_L1",
    "M_F1_L2",
    "M_F2_L0",
    "M_F2_L1",
    "M_F2_L2",
]

# Right hand actuators (5 wrist + 9 finger)
RIGHT_ACTUATOR_NAMES = [
    "M_LINK_HAND_BASE_R_X",
    "M_LINK_HAND_BASE_R_Y",
    "M_LINK_HAND_BASE_R_ROLL",
    "M_LINK_HAND_BASE_R_PITCH",
    "M_LINK_HAND_BASE_R_YAW",
    "M_F0_R0",
    "M_F0_R1",
    "M_F0_R2",
    "M_F1_R0",
    "M_F1_R1",
    "M_F1_R2",
    "M_F2_R0",
    "M_F2_R1",
    "M_F2_R2",
]

# All actuator names (28 total)
ACTUATOR_NAMES = LEFT_ACTUATOR_NAMES + RIGHT_ACTUATOR_NAMES

# Body names for cartesian position tracking (termination).
# Excludes "world" (static).
LEFT_BODY_NAMES = [
    "L_WRIST",
    "LINK_F0_L0",
    "LINK_F0_L1",
    "LINK_F0_L2",
    "LINK_F1_L0",
    "LINK_F1_L1",
    "LINK_F1_L2",
    "LINK_F2_L0",
    "LINK_F2_L1",
    "LINK_F2_L2",
]

RIGHT_BODY_NAMES = [
    "R_WRIST",
    "LINK_F0_R0",
    "LINK_F0_R1",
    "LINK_F0_R2",
    "LINK_F1_R0",
    "LINK_F1_R1",
    "LINK_F1_R2",
    "LINK_F2_R0",
    "LINK_F2_R1",
    "LINK_F2_R2",
]

# All tracked body names (20 hand bodies)
TRACKED_BODY_NAMES = LEFT_BODY_NAMES + RIGHT_BODY_NAMES

# Key body names for key position reward (fingertips + wrists).
KEY_BODY_NAMES = [
    "LINK_F0_L2",
    "LINK_F1_L2",
    "LINK_F2_L2",
    "LINK_F0_R2",
    "LINK_F1_R2",
    "LINK_F2_R2",
]

# Fingertip body names for perturbation forces.
FINGERTIP_BODY_NAMES = [
    "LINK_F0_L2",
    "LINK_F1_L2",
    "LINK_F2_L2",
    "LINK_F0_R2",
    "LINK_F1_R2",
    "LINK_F2_R2",
]

# Body names for contact force tracking reward.
CONTACT_FORCE_BODY_NAMES = [
    "L_WRIST",
    "LINK_F1_L0",
    "LINK_F2_L0",
    "R_WRIST",
    "LINK_F1_R0",
    "LINK_F2_R0",
]

# Sensor names for contact force tracking (force sensors in XML).
CONTACT_FORCE_SENSOR_NAMES = [
    "S_FORCE_L_WRIST",
    "S_FORCE_F1_L0",
    "S_FORCE_F2_L0",
    "S_FORCE_R_WRIST",
    "S_FORCE_F1_R0",
    "S_FORCE_F2_R0",
]

# --- Derived index helpers (auto-computed from name lists above) ---

JOINT_NAME_TO_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}

_n_lw = len(LEFT_WRIST_JOINT_NAMES)
_n_lf = len(LEFT_FINGER_JOINT_NAMES)
_n_rw = len(RIGHT_WRIST_JOINT_NAMES)

LEFT_WRIST_INDICES = list(range(0, _n_lw))
LEFT_FINGER_INDICES = list(range(_n_lw, _n_lw + _n_lf))
RIGHT_WRIST_INDICES = list(range(_n_lw + _n_lf, _n_lw + _n_lf + _n_rw))
RIGHT_FINGER_INDICES = list(range(_n_lw + _n_lf + _n_rw, NQ))

WRIST_INDICES = LEFT_WRIST_INDICES + RIGHT_WRIST_INDICES
FINGER_INDICES = LEFT_FINGER_INDICES + RIGHT_FINGER_INDICES


def joint_index(name: str) -> int:
  return JOINT_NAME_TO_INDEX[name]


def _build_joint_groups():
  """自动构建关节分组，用于绘图。返回 [(group_name, [(index, short_label), ...]), ...]"""
  def _short(name):
    if "HAND_BASE" in name:
      return name.split("_")[-1]  # X, Y, ROLL, PITCH, YAW
    return name.replace("J_", "")

  groups = []
  for side, wrist_names, finger_names, side_label in [
      ("Left", LEFT_WRIST_JOINT_NAMES, LEFT_FINGER_JOINT_NAMES, "L"),
      ("Right", RIGHT_WRIST_JOINT_NAMES, RIGHT_FINGER_JOINT_NAMES, "R"),
  ]:
    groups.append((f"{side} Wrist", [(joint_index(n), _short(n)) for n in wrist_names]))
    for fid in ["F0", "F1", "F2"]:
      fnames = [n for n in finger_names if fid in n]
      groups.append((f"{side} Finger {fid[1]}", [(joint_index(n), _short(n)) for n in fnames]))
  return groups


JOINT_GROUPS = _build_joint_groups()
