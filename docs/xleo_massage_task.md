# Xleo Dual Hand Massage Task 技术文档

## 1. 任务概述

双 Xleo 灵巧手按摩任务：两只三指灵巧手面对面放置，中间夹一个简化的人体手臂（capsule），策略需要跟踪专家示教的周期性按摩轨迹。训练基于 MJX（MuJoCo on JAX），使用 PPO 算法在 GPU 上大规模并行仿真。

**关键文件：**
| 文件 | 用途 |
|------|------|
| `massage.py` | 环境主体（`Massage` 类） |
| `constants.py` | 关节/执行器/body 名称常量 |
| `models/xmls/ftl_xleo_dual_hand.xml` | 双手 MJCF 模型 |
| `models/xmls/ftl_xleo_dual_hand.scene.xml` | 场景（含地面、灯光、人体手臂） |
| `data/massage_traj.pkl` | 专家示教轨迹（已重采样至策略频率） |

---

## 2. 模型结构

### 2.1 自由度

| 部位 | 类型 | 数量 | 关节名示例 |
|------|------|------|-----------|
| 左手腕 XYZ | slide | 3 | `J_LINK_HAND_BASE_L_X/Y/Z` |
| 左手腕 RPY | hinge | 3 | `J_LINK_HAND_BASE_L_ROLL/PITCH/YAW` |
| 左手 3 指 × 3 关节 | hinge | 9 | `J_F0_L0`, `J_F0_L1`, `J_F0_L2`, ... |
| 右手（同左手） | — | 15 | 同上，`_L` → `_R` |
| **合计** | — | **NQ = NV = NU = 30** | — |

- 手腕 slide 关节范围：±0.05 m
- 手腕 hinge 关节范围：±0.5236 rad（±30°）
- 手指关节范围：各不相同，最大约 0~1.92 rad

### 2.2 碰撞 Mesh

每个 body 有两个 geom：
- **visual**：高精度渲染 mesh（`contype=0`，不参与碰撞）
- **collision**：简化凸包 mesh（`contype=1`，参与碰撞）

碰撞凸包已简化至 64~128 faces（原始 1000~2570 faces），存放于 `ftl_meshes/convex_new/`，以降低 MJX 显存开销。

右手复用左手的 finger mesh（仅 `HAND_BASE_L/R` 不同），共 11 个独立碰撞 mesh。

### 2.3 接触排除

XML `<contact>` 中排除了手腕与相邻指节 body 的自碰撞（每手 6 对，共 12 对）。

### 2.4 人体手臂

场景中包含一个静态 capsule（`simple_arm`），代表被按摩的人体手臂：
```xml
<geom name="simple_arm" type="capsule" size="0.045 0.12"
      pos="0.08 0 0.5" class="convex_decomposition" />
```

---

## 3. 控制方式

### 3.1 PD 力矩控制

策略输出的 action 是 **关节位置偏移量**（相对于默认姿态），经 PD 控制器转换为力矩：

```
position_targets = default_pose + action × action_scale (0.5)
torque = kp × (target - q) - kd × q̇
```

PD 增益按部位区分：

| 部位 | kp | kd |
|------|----|----|
| 手腕（6+6 关节） | 100.0 | 5.0 |
| 手指（9+9 关节） | 5.0 | 0.1 |

### 3.2 时间参数

| 参数 | 值 | 含义 |
|------|---|------|
| `sim_dt` | 0.002 s | 物理仿真步长 |
| `ctrl_dt` | 0.002 s | 控制步长 |
| `action_repeat` | 5 | 每个策略步重复 5 次物理步 |
| 实际控制频率 | 100 Hz | `1 / (ctrl_dt × action_repeat)` |
| `episode_length` | 1000 步 | 每 episode 1000 个策略步 |

---

## 4. 专家轨迹

从 `massage_traj.pkl` 加载（已预先重采样至策略频率），包含：
- `qpos`: `(T, 30)` — 30 个关节的位置序列
- `qvel`: `(T, 30)` — 30 个关节的速度序列
- `tracked_body_xpos`: `(T, 20, 3)` — 20 个 tracked body 的笛卡尔位置序列
- `key_body_xpos`: `(T, 8, 3)` — 8 个 key body 的笛卡尔位置序列
- `duration`: 轨迹周期（秒）

轨迹为**周期性循环**，通过离散索引取模 (`traj_idx % traj_len`) 直接查表获取目标状态。

### 4.1 初始化

Reset 时随机采样一个整数步偏移 `step_offset ∈ [0, traj_len)`，从轨迹的随机位置开始：
```
traj_idx = step_offset % traj_len
qpos₀ = expert_qpos[traj_idx] + N(0, 0.1)  (clipped to joint range)
qvel₀ = expert_qvel[traj_idx] + N(0, 0.1)
```

每个 step 中的轨迹索引计算：
```
traj_idx = (step + step_offset) % traj_len
```

---

## 5. 观测空间

采用 **Asymmetric Actor-Critic** 架构：policy 和 value 使用不同的观测。

### 5.1 Policy 观测（`state`，92 维）

| 分量 | 维度 | 说明 |
|------|------|------|
| `noisy_joint_pos_rel` | 30 | 当前关节位置（相对默认姿态，带噪声） |
| `joint_vel` | 30 | 当前关节速度 |
| `phase` | 2 | `[sin(2π × traj_idx / traj_len), cos(2π × traj_idx / traj_len)]` 相位编码 |
| `last_act` | 30 | 上一步动作 |
| **合计** | **92** | |

> **注意**：代码中计算了未来目标观测 `tar_obs_list`（`tar_obs_steps=[1,2,3]`，每步 30 维，共 90 维），但当前已从拼接中注释掉。若启用，policy 观测将为 182 维。

### 5.2 Value 观测（`privileged_state`，212 维）

| 分量 | 维度 | 说明 |
|------|------|------|
| `state_obs` | 92 | 完整 policy 观测 |
| `joint_pos_rel` | 30 | 真实关节位置（无噪声） |
| `joint_vel` | 30 | 真实关节速度（重复，供 critic 使用） |
| `qpos_error` | 30 | 位置跟踪误差（无噪声） |
| `qvel_error` | 30 | 速度跟踪误差 |
| **合计** | **212** | |

### 5.3 观测噪声

仅对 policy 观测中的 `joint_pos_rel` 添加均匀噪声：
```
noise = U(-1, 1) × level(1.0) × scale(0.05)
```
即 ±0.05 rad/m 的均匀扰动。Value 观测使用无噪声的真实值。

---

## 6. 奖励函数设计

总奖励 = Σ(各项奖励 × 权重) × dt

### 6.1 各项奖励

| 奖励项 | 权重 | 公式 | 含义 |
|--------|------|------|------|
| **pose** | +0.5 | `exp(-MSE(q, q*) / (2 × 0.3²))` | 关节位置跟踪，直接在关节位置空间计算 MSE（弧度/米），Gaussian kernel σ=0.3 |
| **vel** | +0.1 | `exp(-MSE(q̇, q̇*) / (2 × 2.0²))` | 关节速度跟踪（σ=2.0） |
| **key_pos** | +0.25 | `exp(-mean(‖x - x*‖²) / (2 × 0.1²))` | 关键 body 笛卡尔位置跟踪（σ=0.1） |
| **action_rate** | -0.001 | `‖a - a_prev‖² + ‖a - 2a_prev + a_prev_prev‖²` | 动作平滑度惩罚（一阶+二阶差分） |
| **energy** | -0.0001 | `Σ\|q̇ × actuator_force\|` | 能耗惩罚 |

### 6.2 关键 body

`key_pos` 奖励跟踪的 8 个 body（指尖 + 手腕）：
- 左手：`L_WRIST`, `LINK_F0_L2`, `LINK_F1_L2`, `LINK_F2_L2`
- 右手：`R_WRIST`, `LINK_F0_R2`, `LINK_F1_R2`, `LINK_F2_R2`

### 6.3 设计理念

- **pose + vel** 提供关节空间的稠密跟踪信号（直接在关节位置空间计算 MSE，适用于 hinge 和 slide 关节）
- **key_pos** 补充笛卡尔空间指尖位置约束，防止关节角正确但末端位置偏移
- **action_rate** 鼓励平滑动作，二阶差分项抑制高频振荡
- **energy** 鼓励高效运动，避免不必要的大力矩

---

## 7. 终止条件设计

三种终止条件（第一步不终止，允许初始稳定）：

### 7.1 NaN 检测
```python
done |= any(isnan(qpos)) or any(isnan(qvel))
```
仿真发散立即终止。

### 7.2 Body 位置偏差
```python
max_body_dist = max(‖xpos_current - xpos_ref‖)  # 对 20 个 tracked body
done |= max_body_dist > 0.1 m
```
任何一个 body 的笛卡尔位置偏离参考轨迹超过 **10 cm** 即终止。

跟踪的 20 个 body：双手各 10 个（1 手腕 + 9 指节）。

### 7.3 关节限位违反
```python
done |= any(q < q_min - margin) or any(q > q_max + margin)
```
关节位置超出限位加裕量即终止：
- Hinge 关节裕量：0.05 rad
- Slide 关节裕量：0.01 m

---

## 8. 域随机化设计

`domain_randomize()` 函数在每次 reset 时对模型参数进行随机化，提升策略的 sim-to-real 泛化能力。

### 8.1 每 Episode 随机化（vmap over envs）

| 参数 | 随机化方式 | 范围 | 说明 |
|------|-----------|------|------|
| 接触摩擦系数 | 赋值 U(a,b) | [0.3, 1.0] | 所有碰撞 geom 统一随机 |
| 连杆质量 | 乘以 U(a,b) | [0.9, 1.1] | 20 个 hand body 各自独立 |
| 连杆质心偏移 | 加上 U(a,b) | [-5mm, 5mm] | 20 个 hand body，每轴独立 |
| 关节摩擦损耗 | 乘以 U(a,b) | [0.5, 2.0] | 30 个关节各自独立 |
| 关节等效转动惯量 | 乘以 U(a,b) | [1.0, 1.05] | 30 个关节各自独立 |
| 关节阻尼 | 乘以 U(a,b) | [0.8, 1.2] | 30 个关节各自独立 |

### 8.2 每 Episode 随机化（在 reset 中）

| 参数 | 随机化方式 | 范围 |
|------|-----------|------|
| PD kp | 乘以 U(a,b) | [0.8, 1.2] |
| PD kd | 乘以 U(a,b) | [0.8, 1.2] |
| 初始关节位置 | 专家 + N(0, σ) | σ = 0.1 |
| 初始关节速度 | 专家 + N(0, σ) | σ = 0.1 |
| 初始轨迹相位 | U(0, traj_len) | 整数步，完整周期 |

---

## 9. 外力扰动系统（可选）

通过 `pert_config.enable` 开关控制（默认关闭），对 6 个指尖 body 施加周期性随机力/力矩。

### 9.1 参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `force_pert` | [0.0, 0.5] N | 力幅值范围 |
| `torque_pert` | [0.0, 0.1] Nm | 力矩幅值范围 |
| `pert_duration_steps` | [1, 50] | 单次扰动持续步数 |
| `pert_wait_steps` | [50, 150] | 两次扰动间隔步数 |

### 9.2 扰动包络

采用正弦包络使扰动平滑起止：
```
u(t) = 0.5 × sin(π × t / T_duration)
force = u(t) × magnitude × direction
```

每次扰动开始时为每个指尖随机采样一个 6D 方向（归一化），扰动期间方向固定。

---

## 10. 网络架构

采用 Asymmetric Actor-Critic（非对称 Actor-Critic）：

| 网络 | 输入 | 隐藏层 | 输出 |
|------|------|--------|------|
| Policy | `state` (92-dim) | [512, 256, 128] | 30-dim action |
| Value | `privileged_state` (212-dim) | [512, 256, 128] | 1-dim value |

---

## 11. PPO 训练参数

| 参数 | 值 |
|------|---|
| `num_envs` | 8192 |
| `num_timesteps` | 200,000,000 |
| `batch_size` | 1024 |
| `unroll_length` | 40 |
| `num_minibatches` | 8 |
| `num_updates_per_batch` | 4 |
| `learning_rate` | 3e-4 |
| `discounting` (γ) | 0.99 |
| `entropy_cost` | 0.01 |
| `reward_scaling` | 1.0 |
| `normalize_observations` | True |

---

## 12. MJX 仿真参数

| 参数 | 值 | 含义 |
|------|---|------|
| `naconmax` | 30 × 1024 = 30,720 | 最大活跃接触数（影响显存） |
| `njmax` | 160 | 最大约束数 |
| `impl` | `"jax"` | 使用 JAX 后端 |

### 碰撞 Mesh 优化

原始碰撞凸包 1000~2570 faces，MJX 按面片数分配显存，导致 24GB GPU OOM。
已将 11 个凸包简化至 64~128 faces，存放于 `ftl_meshes/convex_new/`。

---

## 13. 监控指标

训练过程中记录的 metrics：

| 指标 | 含义 |
|------|------|
| `reward/pose` | 关节角度跟踪奖励 |
| `reward/vel` | 关节速度跟踪奖励 |
| `reward/key_pos` | 关键 body 位置跟踪奖励 |
| `reward/action_rate` | 动作平滑度惩罚（原始值） |
| `reward/energy` | 能耗惩罚（原始值） |
| `tracking_pos_error` | 关节位置 MSE |
| `tracking_vel_error` | 关节速度 MSE |
| `max_body_pos_error` | 最大 body 笛卡尔位置误差 (m) |
