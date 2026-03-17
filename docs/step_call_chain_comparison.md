# Step Function Call Chain Comparison: Massage vs Reorient

本文档对比 `xleo_hand/massage.py` (Massage) 和 `leap_hand_study/reorient.py` (CubeReorientStudy) 两个任务中，调用一次 `step()` 函数时分别会调用哪些函数及调用次数，以便理解 JIT 编译时的差异。

> **约定**: `N×` 表示调用 N 次；缩进表示调用层级；`(conditional)` 表示条件分支。

---

## 1. Massage `step()` 完整调用树

```
step(state, action)
│
├── [conditional] _maybe_apply_perturbation(state)          ×1  (pert_config.enable=True 时)
│   ├── jax.random.split                                    ×1
│   ├── jax.random.normal                                   ×1
│   └── data.replace(xfrc_applied=...)                      ×1
│
├── ── PD torque computation (inline) ──
│   ├── data.qpos[_joint_qids]                              ×1
│   └── data.qvel[_joint_dqids]                             ×1
│
├── mjx_env.step(model, data, torque, n_substeps=1)         ×1
│   └── jax.lax.scan(single_step, data, (), n_substeps=1)
│       └── mjx.step(model, data)                           ×1  (n_substeps=1, action_repeat=5 → 总共 5 次物理步)
│
├── _get_obs(data, info)                                    ×1
│   ├── data.qpos[_joint_qids]                              ×1
│   ├── data.qvel[_joint_dqids]                             ×1
│   ├── jax.random.split + jax.random.uniform (noise)       ×1
│   ├── _encode_joint_pos(noisy_joint_pos_rel)              ×1  ← 当前帧编码
│   │   └── _hinge_to_tan_norm(angles, axes)                ×1
│   │       ├── _axis_angle_to_quat(axes, angles)           ×1
│   │       └── _quat_to_tan_norm(q)                        ×1
│   │           ├── _quat_rotate(q, ref_tan)                ×1
│   │           └── _quat_rotate(q, ref_norm)               ×1
│   ├── [loop] 对 tar_obs_steps 中每个 offset (共 3 个):
│   │   ├── _get_traj_target(tar_time)                      ×3
│   │   │   ├── _interp_traj(time, _traj_qpos)             ×3
│   │   │   └── _interp_traj(time, _traj_qvel)             ×3
│   │   └── _encode_joint_pos(tar_pos_rel)                  ×3  ← 未来目标编码
│   │       └── _hinge_to_tan_norm → ... (同上)             ×3
│   └── _get_traj_target(sim_time)                          ×1  ← critic 用当前帧目标
│       ├── _interp_traj(time, _traj_qpos)                  ×1
│       └── _interp_traj(time, _traj_qvel)                  ×1
│
├── _get_termination(data, info)                            ×1
│   ├── jp.isnan(data.qpos), jp.isnan(data.qvel)           ×1
│   ├── _get_traj_body_pos(sim_time)                        ×1
│   │   └── _interp_traj(time, _traj_xpos)                 ×1
│   ├── data.xpos[_tracked_body_ids]                        ×1
│   └── data.qpos[_joint_qids] (joint limit check)         ×1
│
├── _get_reward(data, action, info)                         ×1
│   ├── _reward_pose(data, sim_time)                        ×1
│   │   ├── _get_traj_target(sim_time)                      ×1
│   │   │   ├── _interp_traj(time, _traj_qpos)             ×1
│   │   │   └── _interp_traj(time, _traj_qvel)             ×1
│   │   ├── _encode_joint_pos(cur)                          ×1
│   │   │   └── _hinge_to_tan_norm → ...                    ×1
│   │   └── _encode_joint_pos(tar)                          ×1
│   │       └── _hinge_to_tan_norm → ...                    ×1
│   ├── _reward_vel(data, sim_time)                         ×1
│   │   └── _get_traj_target(sim_time)                      ×1
│   │       ├── _interp_traj(time, _traj_qpos)             ×1
│   │       └── _interp_traj(time, _traj_qvel)             ×1
│   ├── _reward_key_pos(data, sim_time)                     ×1
│   │   └── _get_traj_key_pos(sim_time)                     ×1
│   │       └── _interp_traj(time, _traj_key_xpos)         ×1
│   ├── _reward_action_rate(act, last_act, last_last_act)   ×1
│   └── _reward_energy(data)                                ×1
│
├── ── Metrics computation (inline) ──
│   ├── _get_traj_target(sim_time)                          ×1  ← tracking error 监控
│   │   ├── _interp_traj(time, _traj_qpos)                 ×1
│   │   └── _interp_traj(time, _traj_qvel)                 ×1
│   └── _get_traj_body_pos(sim_time)                        ×1  ← body error 监控
│       └── _interp_traj(time, _traj_xpos)                 ×1
│
└── state.replace(data, obs, reward, done)                  ×1
```

### Massage 调用次数统计

| 函数 | 调用次数 | 说明 |
|------|---------|------|
| `mjx_env.step` | 1 | 含 `jax.lax.scan` |
| `mjx.step` | 1 (scan 内) | `n_substeps=1`，但 `action_repeat=5` 由外部训练循环控制 |
| `_get_obs` | 1 | |
| `_get_termination` | 1 | |
| `_get_reward` | 1 | |
| `_get_traj_target` | **7** | obs: 3(future)+1(critic) = 4; reward: 2(pose+vel); metrics: 1 |
| `_interp_traj` | **17** | `_get_traj_target` ×7 (每次调用 2 次) = 14; `_get_traj_body_pos` ×2 = 2; `_get_traj_key_pos` ×1 = 1 |
| `_encode_joint_pos` | **6** | obs: 1(cur)+3(future) = 4; reward/pose: 2(cur+tar) |
| `_hinge_to_tan_norm` | **6** | 同 `_encode_joint_pos` |
| `_axis_angle_to_quat` | **6** | 同上 |
| `_quat_to_tan_norm` | **6** | 同上 |
| `_quat_rotate` | **12** | 每次 `_quat_to_tan_norm` 调用 2 次 |
| `_get_traj_body_pos` | **2** | termination: 1; metrics: 1 |
| `_get_traj_key_pos` | **1** | reward/key_pos: 1 |
| `_reward_pose` | 1 | |
| `_reward_vel` | 1 | |
| `_reward_key_pos` | 1 | |
| `_reward_action_rate` | 1 | |
| `_reward_energy` | 1 | |

---

## 2. Reorient `step()` 完整调用树

```
step(state, action)
│
├── [conditional] _maybe_apply_perturbation(state, rng)     ×1  (pert_config.enable=True 时)
│   ├── gen_dir(rng)                                        ×1
│   │   └── jax.random.normal                               ×1
│   ├── get_xfrc(state, pert_dir, duration)                 ×1
│   └── data.replace(xfrc_applied=...)                      ×1
│
├── ── Position target computation (inline) ──
│   ├── EMA smoothing: clip + alpha blending                ×1
│   └── jp.clip(position_targets, lowers, uppers)           ×1
│
├── mjx_env.step(model, data, position_targets, n_substeps=5)  ×1
│   └── jax.lax.scan(single_step, data, (), n_substeps=5)
│       └── mjx.step(model, data)                           ×5  (ctrl_dt=0.05, sim_dt=0.01 → 5 substeps)
│
├── _cube_orientation_error(data)                           ×1  ← 用于 success 判定
│   ├── get_cube_orientation(data)                          ×1
│   │   └── mjx_env.get_sensor_data("cube_orientation")    ×1
│   ├── get_cube_goal_orientation(data)                     ×1
│   │   └── mjx_env.get_sensor_data("cube_goal_orientation") ×1
│   ├── math.quat_mul                                       ×1
│   ├── math.quat_inv                                       ×1
│   └── math.normalize                                      ×1
│
├── _get_termination(data, info)                            ×1
│   ├── get_cube_position(data)                             ×1
│   │   └── mjx_env.get_sensor_data("cube_position")       ×1
│   └── jp.isnan(data.qpos), jp.isnan(data.qvel)           ×1
│
├── _get_obs(data, info)                                    ×1
│   ├── data.qpos[_hand_qids]                               ×1
│   ├── jax.random.split + uniform (joint noise)            ×1
│   ├── jp.roll (qpos_error_history)                        ×1
│   ├── _get_cube_pose (内联闭包):                           ×1
│   │   ├── get_cube_position(data)                         ×1
│   │   │   └── mjx_env.get_sensor_data("cube_position")   ×1
│   │   ├── get_cube_orientation(data)                      ×1
│   │   │   └── mjx_env.get_sensor_data("cube_orientation") ×1
│   │   └── math.normalize (noisy quat)                     ×1
│   ├── uniform_quat(key1) (random injection)               ×1
│   ├── jax.random.bernoulli (injection prob)               ×1
│   ├── get_palm_position(data)                             ×1
│   │   └── mjx_env.get_sensor_data("palm_position")       ×1
│   ├── jp.roll (cube_pos_error_history)                    ×1
│   ├── get_cube_goal_orientation(data)                     ×1
│   │   └── mjx_env.get_sensor_data("cube_goal_orientation") ×1
│   ├── math.quat_mul + math.quat_inv (noisy diff)         ×1
│   ├── math.quat_to_mat (noisy xmat_diff)                 ×1
│   ├── jp.roll (cube_ori_error_history)                    ×1
│   ├── ── Privileged state (critic) ──
│   ├── get_cube_position(data)                             ×1  (uncorrupted)
│   │   └── mjx_env.get_sensor_data("cube_position")       ×1
│   ├── get_cube_orientation(data)                          ×1  (uncorrupted)
│   │   └── mjx_env.get_sensor_data("cube_orientation")    ×1
│   ├── math.quat_mul + math.quat_inv (uncorrupted diff)   ×1
│   ├── math.quat_to_mat (uncorrupted xmat_diff)           ×1
│   ├── get_fingertip_positions(data)                       ×1
│   │   └── mjx_env.get_sensor_data(f"{name}_position")    ×4  (th, if, mf, rf)
│   ├── get_cube_linvel(data)                               ×1
│   │   └── mjx_env.get_sensor_data("cube_linvel")         ×1
│   └── get_cube_angvel(data)                               ×1
│       └── mjx_env.get_sensor_data("cube_angvel")         ×1
│
├── _get_reward(data, action, info, metrics, done)          ×1
│   ├── _reward_cube_orientation(data)                      ×1
│   │   ├── _cube_orientation_error(data)                   ×1  ← 第 2 次调用
│   │   │   ├── get_cube_orientation(data)                  ×1
│   │   │   ├── get_cube_goal_orientation(data)             ×1
│   │   │   ├── math.quat_mul + math.quat_inv              ×1
│   │   │   └── math.normalize                              ×1
│   │   └── reward.tolerance(...)                           ×1
│   ├── _reward_cube_position(data)                         ×1
│   │   ├── get_cube_position(data)                         ×1
│   │   ├── get_palm_position(data)                         ×1
│   │   └── reward.tolerance(...)                           ×1
│   ├── _reward_termination(data, info)                     ×1
│   │   └── _get_termination(data, info)                    ×1  ← 第 2 次调用
│   │       ├── get_cube_position(data)                     ×1
│   │       └── jp.isnan checks                             ×1
│   ├── _reward_hand_pose(data)                             ×1
│   ├── _cost_action_rate(act, last_act, last_last_act)     ×1
│   ├── _cost_joint_vel(data)                               ×1
│   └── _cost_energy(qvel, actuator_force)                  ×1
│
├── ── Goal update (inline) ──
│   ├── jax.random.split                                    ×1
│   ├── jax.random.uniform                                  ×1
│   └── math.quat_integrate(mocap_quat, dquat, dt)         ×1
│
└── state.replace(data, obs, reward, done)                  ×1
```

### Reorient 调用次数统计

| 函数 | 调用次数 | 说明 |
|------|---------|------|
| `mjx_env.step` | 1 | 含 `jax.lax.scan` |
| `mjx.step` | **5** (scan 内) | `ctrl_dt/sim_dt = 0.05/0.01 = 5` substeps |
| `_get_obs` | 1 | |
| `_get_termination` | **2** | 直接调用 1 + `_reward_termination` 内 1 |
| `_get_reward` | 1 | |
| `_cube_orientation_error` | **2** | success 判定 1 + `_reward_cube_orientation` 内 1 |
| `mjx_env.get_sensor_data` | **20** | 详见下表 |
| `get_cube_position` | **5** | termination: 1; obs: 2(noisy+uncorrupted); reward/position: 1; reward/termination 内 `_get_termination`: 1 |
| `get_cube_orientation` | **4** | ori_error: 1; obs: 2(noisy+uncorrupted); reward/ori 内的 ori_error: 1 |
| `get_cube_goal_orientation` | **3** | ori_error: 1; obs: 1; reward/ori 内的 ori_error: 1 |
| `get_palm_position` | **2** | obs: 1; reward/position: 1 |
| `get_fingertip_positions` | **1** | obs (privileged) |
| `get_cube_linvel` | **1** | obs (privileged) |
| `get_cube_angvel` | **1** | obs (privileged) |
| `math.quat_mul` | **4** | ori_error: 1; obs noisy: 1; obs uncorrupted: 1; reward/ori 内的 ori_error: 1 |
| `math.quat_inv` | **4** | 同 `quat_mul` |
| `math.normalize` | **2** | ori_error: 1; reward/ori 内的 ori_error: 1 |
| `math.quat_to_mat` | **2** | obs noisy: 1; obs uncorrupted: 1 |
| `math.quat_integrate` | **1** | goal 更新 |
| `reward.tolerance` | **2** | orientation: 1; position: 1 |
| `uniform_quat` | **1** | obs 中 random injection |

**`get_sensor_data` 详细统计:**

| 调用位置 | 传感器名称 | 次数 |
|----------|-----------|------|
| `_cube_orientation_error` (×2) | `cube_orientation` | 2 |
| `_cube_orientation_error` (×2) | `cube_goal_orientation` | 2 |
| `_get_termination` (×2) | `cube_position` | 2 |
| `_get_obs` | `cube_position` (noisy) | 1 |
| `_get_obs` | `cube_orientation` (noisy) | 1 |
| `_get_obs` | `palm_position` | 1 |
| `_get_obs` | `cube_goal_orientation` | 1 |
| `_get_obs` | `cube_position` (uncorrupted) | 1 |
| `_get_obs` | `cube_orientation` (uncorrupted) | 1 |
| `_get_obs` | `{name}_position` (×4 fingertips) | 4 |
| `_get_obs` | `cube_linvel` | 1 |
| `_get_obs` | `cube_angvel` | 1 |
| `_reward_cube_position` | `cube_position` | 1 |
| `_reward_cube_position` | `palm_position` | 1 |
| **合计** | | **20** |

---

## 3. 关键差异对比表

| 维度 | Massage (xleo_hand) | Reorient (leap_hand_study) |
|------|--------------------|-----------------------------|
| **控制方式** | PD 力矩控制 (手动计算 `kp*(target-q) - kd*qvel`) | 位置控制 (直接设 ctrl，含 EMA 平滑) |
| **`mjx.step` 次数/step** | 1 (`n_substeps=1`, `ctrl_dt=sim_dt=0.002`) | **5** (`ctrl_dt=0.05`, `sim_dt=0.01`) |
| **`action_repeat`** | 5 (外部循环) | 1 |
| **观测编码** | 6D 旋转编码 (hinge → quat → tan_norm) | 原始关节角 + 四元数差异 + 旋转矩阵 |
| **`_encode_joint_pos` 调用** | **6** 次 (1 cur + 3 future + 2 reward) | **0** 次 (无此函数) |
| **`_quat_rotate` 调用** | **12** 次 (6 × 2) | **0** 次 (使用 `math.quat_to_mat` 代替) |
| **`_interp_traj` 调用** | **17** 次 | **0** 次 (无轨迹追踪) |
| **`get_sensor_data` 调用** | **0** 次 (直接读 `data.qpos/xpos`) | **20** 次 |
| **`reward.tolerance` 调用** | **0** 次 (手动 Gaussian kernel) | **2** 次 |
| **`math.quat_*` 调用** | **0** 次 | **~12** 次 (`quat_mul`×4 + `quat_inv`×4 + `normalize`×2 + `quat_to_mat`×2) |
| **`math.quat_integrate`** | **0** 次 | **1** 次 (goal 更新) |
| **`_get_termination` 调用** | **1** 次 | **2** 次 (reward 中重复调用) |
| **`_cube_orientation_error`** | N/A | **2** 次 (success + reward 重复调用) |
| **Reward 子函数数量** | **5** (pose, vel, key_pos, action_rate, energy) | **7** (orientation, position, termination, hand_pose, action_rate, joint_vel, energy) |
| **历史缓冲区** | 无 | 有 (`jp.roll` × 3: qpos_error, cube_pos_error, cube_ori_error) |
| **随机采样** | 无 (obs 中只有噪声) | `uniform_quat` + `bernoulli` (random injection) |
| **扰动目标** | 指尖 bodies (多体) | 立方体 body (单体) |

---

## 4. JIT 编译影响分析

### 4.1 Tracing 复杂度

**Massage** 的 JIT tracing 复杂度主要来自:
1. **大量 `_interp_traj` 调用** (17 次): 每次调用包含 `jp.mod`、`jp.floor`、`jp.clip` 和线性插值，虽然计算简单但 trace 节点多。
2. **6D 旋转编码链** (6 次 `_encode_joint_pos`): `_hinge_to_tan_norm` → `_axis_angle_to_quat` → `_quat_to_tan_norm` → `_quat_rotate` ×2，形成 4 层深的调用链，每层产生多个 JAX 原语。
3. **相同函数不同参数多次调用**: `_get_traj_target` 被调用 7 次，但参数 (time) 不同，JAX 会展开为不同的 trace 分支。

**Reorient** 的 JIT tracing 复杂度主要来自:
1. **`mjx.step` 在 scan 内被调用 5 次**: 这是最大的计算开销，但 `jax.lax.scan` 不会展开循环，仅 trace 一次循环体。
2. **重复调用**: `_cube_orientation_error` 和 `_get_termination` 各被调用 2 次，产生重复的 trace 图。
3. **大量 `get_sensor_data` 调用** (20 次): 每次调用虽然是简单的数组切片，但会产生 20 个独立的 slice 操作。
4. **`math.*` 四元数运算** (~12 次): 多种不同的四元数操作，每种有不同的内部实现。

### 4.2 编译时间预期

| 因素 | Massage | Reorient |
|------|---------|----------|
| 物理模拟 (`mjx.step`) | 1 次 (快) | 5 次 in scan (trace 1 次，但循环体更大) |
| 独立函数调用总数 | ~50+ | ~60+ |
| Trace 图深度 | 较深 (6D编码链: 4层) | 较浅但更宽 (多传感器读取) |
| 潜在 CSE 优化 | `_interp_traj` 可能部分合并 | `get_sensor_data` 调用独立，难合并 |
| 可优化的重复计算 | `_get_traj_target` 的 7 次调用中部分 time 参数相同，XLA 可能做 CSE | `_cube_orientation_error` 2 次、`_get_termination` 2 次完全重复，XLA CSE 可消除 |

### 4.3 优化建议

**Massage:**
- `_get_traj_target(sim_time)` 在 `_get_obs`(critic)、`_get_reward`(pose, vel)、metrics 中被重复调用，可缓存为局部变量。
- `_get_traj_body_pos(sim_time)` 在 `_get_termination` 和 metrics 中重复调用，可缓存。
- **预期效果**: 减少 `_interp_traj` 从 17 次到约 11 次 (减少 3 次 `_get_traj_target` + 1 次 `_get_traj_body_pos` 的重复)。

**Reorient:**
- `_cube_orientation_error(data)` 在 success 判定和 `_reward_cube_orientation` 中被重复调用，可将结果缓存传入 reward 函数。
- `_get_termination(data, info)` 在主流程和 `_reward_termination` 中被重复调用，已有 `done` 变量可直接传入。
- `get_cube_position` 被调用 5 次，`get_cube_orientation` 被调用 4 次，可在 step 开头一次性读取。
- **预期效果**: 减少 `get_sensor_data` 从 20 次到约 12 次；消除 `_cube_orientation_error` 和 `_get_termination` 的重复调用。

> **注**: 在实际 XLA 编译中，Common Subexpression Elimination (CSE) 可能自动消除部分重复计算，但手动消除可减少 trace 时间和 IR 图大小，从而加速首次 JIT 编译。
