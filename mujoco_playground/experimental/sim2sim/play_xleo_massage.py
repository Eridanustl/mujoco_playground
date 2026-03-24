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
"""Deploy an MJX massage policy in ONNX format to C MuJoCo and play with it.

The observation and action processing must exactly match the training
environment (massage.py) to ensure correct sim2sim transfer.

Training observation (192-dim "state"):
  - Left hand proprioception (33):
      wrist_pos(3) + wrist_rot_sincos(6) + wrist_linvel(3)
      + wrist_angvel(3) + finger_qpos_rel(9) + finger_qvel(9)
  - Right hand proprioception (29):
      Same structure, but right finger slices yield 7 dims due to
      28-joint array with hardcoded [21:30] indexing.
  - Future trajectory targets (102):
      3 future steps x ~34 dims each (left 18 + right 16).
  - Last action (28)

Action processing:
  1. Scale wrist actions by 0.05, finger actions by 0.5
  2. position_targets = default_pose + scaled_action
  3. Clip to joint limits
  4. PD torque = kp * (target - q) - kd * qvel
  5. Clip torque to actuator control range
"""

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

# Future target observation steps (must match training config).
_TARGET_OBS_STEPS = [1, 2, 3]


class OnnxController:
  """ONNX controller for the XleoMassage dual-hand massage task.

  Replicates the exact observation construction and action processing
  from the MJX training environment (massage.py).
  """

  def __init__(
      self,
      policy_path: str,
      model: mujoco.MjModel,
      joint_qids: np.ndarray,
      joint_dqids: np.ndarray,
      default_pose: np.ndarray,
      kp: np.ndarray,
      kd: np.ndarray,
      traj_qpos: np.ndarray,
      traj_len: int,
      n_substeps: int,
      obs_mean: np.ndarray,
      obs_std: np.ndarray,
      wrist_action_scale: float = 0.05,
      finger_action_scale: float = 0.5,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    self._model = model
    self._joint_qids = joint_qids
    self._joint_dqids = joint_dqids
    self._default_pose = default_pose
    self._kp = kp
    self._kd = kd
    self._traj_qpos = traj_qpos
    self._traj_len = traj_len
    self._wrist_action_scale = wrist_action_scale
    self._finger_action_scale = finger_action_scale

    # Wrist/finger index masks into the 28-dim action/joint array.
    # Left: wrist=[0..4], finger=[5..13]; Right: wrist=[14..18], finger=[19..27]
    self._wrist_ids = np.array(
        list(range(0, 5)) + list(range(14, 19)), dtype=np.int32
    )
    self._finger_ids = np.array(
        list(range(5, 14)) + list(range(19, 28)), dtype=np.int32
    )

    # Joint limits for clipping position targets.
    joint_ids = np.array(
        [model.joint(n).id for n in consts.JOINT_NAMES], dtype=np.int32
    )
    self._jnt_range_low = model.jnt_range[joint_ids, 0]
    self._jnt_range_high = model.jnt_range[joint_ids, 1]

    # Torque limits from actuator control range.
    self._torque_low = model.actuator_ctrlrange[:, 0]
    self._torque_high = model.actuator_ctrlrange[:, 1]

    # Action clipping bounds derived from training normalizer statistics.
    # The obs includes last_action; its normalizer has small std for wrist
    # joints (~0.06-0.09). Out-of-distribution actions cause feedback
    # divergence via (action - mean) / tiny_std → huge normalized values.
    # Clip to mean ± 4*std of the last_action obs slice.
    act_dim = consts.NU
    la_mean = obs_mean[-act_dim:]
    la_std = obs_std[-act_dim:]
    self._action_lo = (la_mean - 4.0 * la_std).astype(np.float32)
    self._action_hi = (la_mean + 4.0 * la_std).astype(np.float32)

    self._last_action = np.zeros(consts.NU, dtype=np.float32)
    self._counter = 0
    self._n_substeps = n_substeps
    self._traj_idx = 0

  def _get_obs(self, data: mujoco.MjData) -> np.ndarray:
    """Construct 192-dim observation matching the training environment.

    Replicates massage.py _get_obs() with the same hardcoded index slices.
    Note: the training code uses [0:3], [3:6], [6:15], [15:18], [18:21],
    [21:30] on a 28-element array, so the right-hand finger slices yield
    7 elements instead of 9.
    """
    joint_pos = data.qpos[self._joint_qids]  # (28,)
    joint_vel = data.qvel[self._joint_dqids]  # (28,)

    # === Left hand proprioception (33 dims) ===
    l_wrist_pos = joint_pos[0:3]  # (3,)
    l_wrist_rot_obs = np.concatenate([
        np.sin(joint_pos[3:6]),
        np.cos(joint_pos[3:6]),
    ])  # (6,)
    l_wrist_lin_vel = joint_vel[0:3]  # (3,)
    l_wrist_ang_vel = joint_vel[3:6]  # (3,)
    l_finger_qpos = joint_pos[6:15] - self._default_pose[6:15]  # (9,)
    l_finger_qvel = joint_vel[6:15]  # (9,)

    # === Right hand proprioception (29 dims) ===
    r_wrist_pos = joint_pos[15:18]  # (3,)
    r_wrist_rot_obs = np.concatenate([
        np.sin(joint_pos[18:21]),
        np.cos(joint_pos[18:21]),
    ])  # (6,)
    r_wrist_lin_vel = joint_vel[15:18]  # (3,)
    r_wrist_ang_vel = joint_vel[18:21]  # (3,)
    r_finger_qpos = joint_pos[21:30] - self._default_pose[21:30]  # (7,)
    r_finger_qvel = joint_vel[21:30]  # (7,)

    # === Future target observations (102 dims) ===
    target_obs_list = []
    for step_offset in _TARGET_OBS_STEPS:
      future_idx = (self._traj_idx + step_offset) % self._traj_len
      tgt = self._traj_qpos[future_idx]  # (28,)

      # Left target (18 dims)
      left_target = np.concatenate([
          tgt[0:3] - joint_pos[0:3],  # wrist pos diff (3,)
          np.sin(tgt[3:6]),
          np.cos(tgt[3:6]),  # wrist rot sin/cos (6,)
          tgt[6:15] - self._default_pose[6:15],  # finger qpos rel (9,)
      ])

      # Right target (16 dims — [21:30] on 28-elem array yields 7)
      right_target = np.concatenate([
          tgt[15:18] - joint_pos[15:18],  # (3,)
          np.sin(tgt[18:21]),
          np.cos(tgt[18:21]),  # (6,)
          tgt[21:30] - self._default_pose[21:30],  # (7,)
      ])

      target_obs_list.append(np.concatenate([left_target, right_target]))

    target_obs = np.concatenate(target_obs_list)  # (102,)

    # === Assemble full 192-dim observation ===
    obs = np.concatenate([
        # Left hand (33)
        l_wrist_pos,
        l_wrist_rot_obs,
        l_wrist_lin_vel,
        l_wrist_ang_vel,
        l_finger_qpos,
        l_finger_qvel,
        # Right hand (29)
        r_wrist_pos,
        r_wrist_rot_obs,
        r_wrist_lin_vel,
        r_wrist_ang_vel,
        r_finger_qpos,
        r_finger_qvel,
        # Future targets (102)
        target_obs,
        # Last action (28)
        self._last_action,
    ])
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps != 0:
      return

    obs = self._get_obs(data)
    onnx_input = {"obs": obs.reshape(1, -1)}
    onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]

    # Clip raw action to prevent feedback divergence.
    # Training normalizer has very small std for wrist actions (~0.06-0.09),
    # so out-of-distribution actions feed back through last_action obs and
    # get amplified by normalization, causing exponential blowup.
    onnx_pred = np.clip(onnx_pred, self._action_lo, self._action_hi)

    # Apply per-joint action scales (must match training).
    scaled_action = onnx_pred.copy()
    scaled_action[self._wrist_ids] *= self._wrist_action_scale
    scaled_action[self._finger_ids] *= self._finger_action_scale

    # Compute position targets and clip to joint limits.
    position_targets = self._default_pose + scaled_action
    position_targets = np.clip(
        position_targets, self._jnt_range_low, self._jnt_range_high
    )

    # PD torque control: tau = kp * (target - q) - kd * qvel
    joint_pos = data.qpos[self._joint_qids]
    joint_vel = data.qvel[self._joint_dqids]
    torque = self._kp * (position_targets - joint_pos) - self._kd * joint_vel
    torque = np.clip(torque, self._torque_low, self._torque_high)

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

  # Timing must match training: ctrl_dt=0.02, sim_dt=0.005 → 50 Hz policy.
  sim_dt = 0.005
  n_substeps = 4  # ctrl_dt / sim_dt = 0.02 / 0.005 = 4
  model.opt.timestep = sim_dt

  # Joint IDs for qpos and qvel indexing.
  joint_qids = get_qpos_ids(model, consts.JOINT_NAMES)
  joint_dqids = get_qvel_ids(model, consts.JOINT_NAMES)

  # Default pose from keyframe.
  default_pose = model.qpos0[joint_qids].copy()

  # Build per-joint PD gains (must match training config).
  # Left: wrist=[0..4], finger=[5..13]; Right: wrist=[14..18], finger=[19..27]
  wrist_ids = list(range(0, 5)) + list(range(14, 19))
  finger_ids = list(range(5, 14)) + list(range(19, 28))
  kp = np.zeros(consts.NU, dtype=np.float64)
  kd = np.zeros(consts.NU, dtype=np.float64)
  kp[wrist_ids] = 10.0
  kd[wrist_ids] = 0.5
  kp[finger_ids] = 5.0
  kd[finger_ids] = 0.1

  # Load expert massage trajectory.
  traj_path = consts.DATA_PATH / "massage_traj.pkl"
  with open(epath.Path(traj_path), "rb") as f:
    traj = pickle.load(f)
  traj_qpos = np.array(traj["qpos"])  # (T, 28)
  traj_len = traj_qpos.shape[0]

  # Load normalizer statistics (saved by export script alongside ONNX).
  norm_path = _ONNX_DIR / "xleo_massage_policy_norm.npz"
  norm_data = np.load(epath.Path(norm_path).as_posix())
  obs_mean = norm_data["obs_mean"]
  obs_std = norm_data["obs_std"]

  policy = OnnxController(
      policy_path=(_ONNX_DIR / "xleo_massage_policy.onnx").as_posix(),
      model=model,
      joint_qids=joint_qids,
      joint_dqids=joint_dqids,
      default_pose=default_pose,
      kp=kp,
      kd=kd,
      traj_qpos=traj_qpos,
      traj_len=traj_len,
      n_substeps=n_substeps,
      obs_mean=obs_mean,
      obs_std=obs_std,
      wrist_action_scale=0.05,
      finger_action_scale=0.5,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
