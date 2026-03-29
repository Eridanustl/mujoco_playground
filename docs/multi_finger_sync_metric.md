# 多指协同同步率指标分析

## 1. 指标含义

在双手按摩模仿学习任务中，"多指协同同步率"衡量的是：**左右两只手（以及同一只手内的多根手指）在执行按摩动作时，是否在时间上协调一致、步调同步。**

该指标包含两个层次：

### 1.1 双手间协同（Inter-hand Synchronization）

按摩是一个双手配合的动作，左手和右手需要在正确的时间做正确的事（比如同时合拢施力、交替揉捏等）。如果左手已经到位了但右手还在运动，或者两只手的相位不一致，按摩效果就会很差。

### 1.2 同一手内的多指协同（Intra-hand Finger Coordination）

同一只手的三根手指需要协调配合。比如三指同时抱紧物体时，不能一根到位了另外两根还没动。

---

## 2. 量化方案

### 2.1 方案 A：基于轨迹跟踪误差的时间一致性（推荐）

**核心思想：** 如果多指协同做得好，那么所有手指相对于参考轨迹的"误差水平"应该是一致的——不应该出现"有些手指提前到位、有些手指还在落后"的现象。

**计算步骤：**

对每个时间步 $t$：

**Step 1 — 计算每根手指的归一化跟踪误差：**

$$e_k(t) = \frac{\|q_k(t) - q_k^*(t)\|}{\|q_k^{\text{range}}\|}, \quad k = 1, \ldots, 6 \text{ (左3右3)}$$

其中 $q_k(t)$ 为第 $k$ 根手指当前关节角向量，$q_k^*(t)$ 为参考轨迹目标值，$q_k^{\text{range}}$ 为关节范围（用于归一化）。

**Step 2 — 同手内同步率：**

$$\text{sync\_intra}(t) = 1 - \frac{\text{std}(e_{\text{same\_hand}})}{\text{mean}(e_{\text{same\_hand}}) + \epsilon}$$

分别对左手三指和右手三指计算，然后取均值。直觉：如果同一只手的三根手指误差水平一致（方差为 0），则同步率为 1。

**Step 3 — 双手间同步率：**

$$\text{sync\_inter}(t) = 1 - \frac{|\text{mean}(e_{\text{left}}) - \text{mean}(e_{\text{right}})|}{\text{mean}(e_{\text{all}}) + \epsilon}$$

衡量左手整体和右手整体的跟踪误差是否一致。

**Step 4 — 总体多指协同同步率：**

$$\text{sync}(t) = \alpha \cdot \text{sync\_intra}(t) + (1 - \alpha) \cdot \text{sync\_inter}(t)$$

其中 $\alpha$ 为权重系数（建议 $\alpha = 0.5$），值域 $[0, 1]$，1 表示完全同步。

**优点：**
- 已有 `tracking_pos_error_per_step` 指标，在此基础上按手指分组计算方差即可
- 物理含义清晰，容易解释
- 与现有 DeepMimic reward 体系兼容，可直接作为额外的 reward 项

---

### 2.2 方案 B：基于关节速度方向的相关性

**核心思想：** 协同的手指应该在该动的时候一起动、该停的时候一起停。

对于应该协同的手指组（如左手三指），计算速度的皮尔逊相关系数：

$$\text{sync\_vel} = \text{mean}\big(\text{corr}(\dot{q}_i, \dot{q}_j)\big), \quad \forall \text{ finger pairs } (i, j)$$

其中 $\text{corr}$ 为皮尔逊相关系数。相关系数越接近 1（或 -1，取决于动作对称性），协同越好。

**优点：**
- 直接反映运动方向的一致性
- 不依赖参考轨迹，可用于评估无参考场景

**缺点：**
- 需要在时间窗口上计算，单步无法获得
- 对于交替运动模式（一只手动另一只手停），相关系数可能为负，需额外处理

---

### 2.3 方案 C：基于相位差的同步率

**核心思想：** 因为按摩动作是周期性轨迹，可以用经典的相位同步指标。

对每根手指用 Hilbert 变换提取瞬时相位 $\phi_k(t)$，然后计算相位同步率：

$$\text{sync\_phase}(t) = \left| \frac{1}{N} \sum_{k} e^{j(\phi_i(t) - \phi_k(t))} \right|$$

值域 $[0, 1]$，1 = 完全相位同步。

**优点：**
- 经典的同步性度量方法，理论基础扎实
- 天然适合周期性任务

**缺点：**
- Hilbert 变换在 JAX JIT 中实现较复杂
- 对非平稳信号（如开始/结束阶段）可能不稳定

---

## 3. 方案对比

| 维度 | 方案 A（误差一致性） | 方案 B（速度相关性） | 方案 C（相位同步） |
|------|---------------------|---------------------|-------------------|
| 实现难度 | 低 | 中 | 高 |
| 单步可计算 | ✅ | ❌（需时间窗口） | ❌（需时间窗口） |
| 可作为 reward | ✅ | ❌ | ❌ |
| 依赖参考轨迹 | 是 | 否 | 否 |
| 物理直觉 | 强 | 中 | 强 |
| 适合周期性任务 | ✅ | ✅ | ✅✅ |

---

## 4. 推荐方案

**推荐方案 A**，理由：

1. **实现简单**：在现有 `tracking_pos_error_per_step` 基础上按手指分组计算方差即可
2. **单步可计算**：可直接加入 MJX 环境的 metrics，无需时间窗口
3. **可作为 reward**：如需要，可直接作为额外的奖励项引导策略学习协同行为
4. **物理含义清晰**：值域 [0, 1]，1 表示完全同步，便于汇报和对比
5. **与现有体系兼容**：与 DeepMimic 风格的 reward 设计无冲突

### 4.1 实现位置

在 `massage.py` 的 `step()` 方法中，与现有的 `tracking_pos_error_per_step` 等 metrics 并列，新增 `finger_sync_rate` 指标。

### 4.2 伪代码

```python
# 在 step() 方法中，计算完 joint_pos 和 target_qpos 后：

# 每根手指的跟踪误差 (6 个标量: 左手3指 + 右手3指)
# 左手手指: F0_L (3 joints), F1_L (3 joints), F2_L (3 joints)
# 右手手指: F0_R (3 joints), F1_R (3 joints), F2_R (3 joints)
finger_errors = []
for finger_joint_indices in [F0_L, F1_L, F2_L, F0_R, F1_R, F2_R]:
    err = jp.sqrt(jp.mean(jp.square(
        joint_pos[finger_joint_indices] - target_qpos[finger_joint_indices]
    )))
    finger_errors.append(err)

finger_errors = jp.array(finger_errors)  # (6,)
left_errors = finger_errors[:3]   # 左手三指
right_errors = finger_errors[3:]  # 右手三指

eps = 1e-6

# 同手内同步率
left_sync = 1.0 - jp.std(left_errors) / (jp.mean(left_errors) + eps)
right_sync = 1.0 - jp.std(right_errors) / (jp.mean(right_errors) + eps)
sync_intra = (left_sync + right_sync) / 2.0

# 双手间同步率
sync_inter = 1.0 - jp.abs(jp.mean(left_errors) - jp.mean(right_errors)) / (jp.mean(finger_errors) + eps)

# 总体同步率
alpha = 0.5
sync_rate = alpha * sync_intra + (1.0 - alpha) * sync_inter

# 记录指标
state.metrics["finger_sync_rate"] = jp.clip(sync_rate, 0.0, 1.0)
```
