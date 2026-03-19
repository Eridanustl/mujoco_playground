# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deploy an MJX massage policy in ONNX format to C MuJoCo and play with it."""

import pickle

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

from mujoco_playground._src.manipulation.xleo_hand import constants as consts
from mujoco_playground._src.manipulation.xleo_hand.massage import get_assets
from mujoco_playground._src.mjx_env import get_qpos_ids
from mujoco_playground._src.mjx_env import get_qvel_ids

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"


class OnnxController:
  """ONNX controller for the XleoMassage dual-hand massage task."""

  def __init__(
      self,
      policy_path: str,
      joint_qids: np.ndarray,
      joint_dqids: np.ndarray,
      default_pose: np.ndarray,
      kp: np.ndarray,
      kd: np.ndarray,
      traj_qpos: np.ndarray,
      traj_len: int,
      n_substeps: int,
      action_scale: float = 0.5,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    self._joint_qids = joint_qids
    self._joint_dqids = joint_dqids
    self._default_pose = default_pose
    self._kp = kp
    self._kd = kd
    self._traj_qpos = traj_qpos
    self._traj_len = traj_len
    self._action_scale = action_scale

    self._last_action = np.zeros(consts.NU, dtype=np.float32)
    self._counter = 0
    self._n_substeps = n_substeps
    self._traj_idx = 0

  def get_obs(self, model, data) -> np.ndarray:  # pylint: disable=unused-argument
    joint_pos = data.qpos[self._joint_qids]
    joint_vel = data.qvel[self._joint_dqids]

    # Relative joint positions (no noise in deployment).
    joint_pos_rel = joint_pos - self._default_pose

    # Phase encoding from trajectory index.
    phase_angle = 2.0 * np.pi * self._traj_idx / self._traj_len
    phase = np.array([np.sin(phase_angle), np.cos(phase_angle)])

    # 92-dim state observation.
    obs = np.concatenate([
        joint_pos_rel,      # 30: relative joint positions
        joint_vel,          # 30: joint velocities
        phase,              # 2: [sin(phase), cos(phase)]
        self._last_action,  # 30: last action
    ])
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs = self.get_obs(model, data)
      onnx_input = {"obs": obs.reshape(1, -1)}
      onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]

      # Compute position targets from action.
      position_targets = self._default_pose + onnx_pred * self._action_scale

      # PD torque control: tau = kp * (target - q) - kd * qvel
      joint_pos = data.qpos[self._joint_qids]
      joint_vel = data.qvel[self._joint_dqids]
      torque = (
          self._kp * (position_targets - joint_pos)
          - self._kd * joint_vel
      )

      data.ctrl[:] = torque
      self._last_action = onnx_pred.copy()

      # Advance trajectory index.
      self._traj_idx = (self._traj_idx + 1) % self._traj_len


def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_string(
      epath.Path(consts.SCENE_XML).read_text(),
      assets=get_assets(),
  )
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 0)

  # Timing: policy at 100 Hz (action_repeat=5 × sim_dt=0.002).
  sim_dt = 0.002
  n_substeps = 5
  model.opt.timestep = sim_dt

  # Joint IDs for qpos and qvel indexing.
  joint_qids = get_qpos_ids(model, consts.JOINT_NAMES)
  joint_dqids = get_qvel_ids(model, consts.JOINT_NAMES)

  # Default pose from keyframe.
  default_pose = model.qpos0[joint_qids].copy()

  # Build per-joint PD gains.
  # Wrist: indices 0..5 (left) and 15..20 (right).
  # Fingers: indices 6..14 (left) and 21..29 (right).
  wrist_ids = list(range(0, 6)) + list(range(15, 21))
  finger_ids = list(range(6, 15)) + list(range(21, 30))
  kp = np.zeros(consts.NU, dtype=np.float64)
  kd = np.zeros(consts.NU, dtype=np.float64)
  kp[wrist_ids] = 100.0
  kd[wrist_ids] = 5.0
  kp[finger_ids] = 5.0
  kd[finger_ids] = 0.1

  # Load expert massage trajectory.
  traj_path = consts.DATA_PATH / "massage_traj.pkl"
  with open(epath.Path(traj_path), "rb") as f:
    traj = pickle.load(f)
  traj_qpos = np.array(traj["qpos"])  # (T, 30)
  traj_len = traj_qpos.shape[0]

  policy = OnnxController(
      policy_path=(_ONNX_DIR / "xleo_massage_policy.onnx").as_posix(),
      joint_qids=joint_qids,
      joint_dqids=joint_dqids,
      default_pose=default_pose,
      kp=kp,
      kd=kd,
      traj_qpos=traj_qpos,
      traj_len=traj_len,
      n_substeps=n_substeps,
      action_scale=0.5,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
