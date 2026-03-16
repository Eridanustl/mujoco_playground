# `ctrl_dt`、`sim_dt`、`action_repeat` 与 `policy_dt` 的关系

以 `leap_hand/reorient.py` 为例，整个时间层级从底到顶有 **3 层**：

## 1. 底层：`sim_dt` — 物理仿真时间步

这是 MuJoCo 引擎每次 `mjx.step()` 推进的最小时间单位。

```python
# mujoco_playground/_src/manipulation/leap_hand/reorient.py:33-36
def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.01,
```

这里 `sim_dt = 0.01` 秒，即物理仿真以 **100 Hz** 运行。

## 2. 中层：`ctrl_dt` — 控制时间步（环境 step 的时间步）

每次调用 `env.step()` 时，控制信号保持不变，连续执行 `n_substeps` 次物理仿真：

```python
# mujoco_playground/_src/mjx_env.py:262-274
@property
def dt(self) -> float:
    """Control timestep for the environment."""
    return self._ctrl_dt

@property
def n_substeps(self) -> int:
    """Number of sim steps per control step."""
    return int(round(self.dt / self.sim_dt))
```

**关键公式**：`n_substeps = ctrl_dt / sim_dt = 0.05 / 0.01 = 5`

在 `reorient.py` 的 `step()` 中可以看到这个调用：

```python
# mujoco_playground/_src/manipulation/leap_hand/reorient.py:213-215
data = mjx_env.step(
    self.mjx_model, state.data, motor_targets, self.n_substeps
)
```

而 `mjx_env.step()` 的实现是用 `jax.lax.scan` 循环执行 `n_substeps` 次：

```python
# mujoco_playground/_src/mjx_env.py:163-174
def step(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
  def single_step(data, _):
    data = data.replace(ctrl=action)
    data = mjx.step(model, data)
    return data, None

  return jax.lax.scan(single_step, data, (), n_substeps)[0]
```

所以**每次 `env.step()` 调用 = 5 次 `mjx.step()` = 推进 0.05 秒仿真时间**。

## 3. 顶层：`action_repeat` — 策略推理频率（`policy_dt`）

`action_repeat` 在 Brax 的 `EpisodeWrapper` 中使用，它将**同一个 action 重复喂给 `env.step()` 多次**：

```python
# brax/envs/wrappers/training.py:98-104
def step(self, state: State, action: jax.Array) -> State:
    def f(state, _):
      nstate = self.env.step(state, action)
      return nstate, nstate.reward

    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
    state = state.replace(reward=jp.sum(rewards, axis=0))
```

**关键公式**：`policy_dt = ctrl_dt × action_repeat`

对于 `reorient.py`：`action_repeat = 1`，所以 `policy_dt = 0.05 × 1 = 0.05` 秒，策略推理频率 = **20 Hz**。

---

## 总结关系图

```
policy_dt = ctrl_dt × action_repeat = 0.05 × 1 = 0.05s  (策略推理频率 20 Hz)
    │
    └── 每次策略推理重复 action_repeat=1 次 env.step()
         │
         ctrl_dt = 0.05s  (控制频率 20 Hz)
             │
             └── 每次 env.step() 执行 n_substeps=5 次 mjx.step()
                  │
                  sim_dt = 0.01s  (物理仿真频率 100 Hz)
```

| 参数 | 值 | 含义 |
|---|---|---|
| `sim_dt` | 0.01s | 物理仿真最小步长 (100 Hz) |
| `ctrl_dt` | 0.05s | 控制信号更新间隔 (20 Hz) |
| `n_substeps` | `ctrl_dt/sim_dt = 5` | 每个控制步中的仿真步数 |
| `action_repeat` | 1 | 同一 action 重复施加的控制步数 |
| `policy_dt` (隐含) | `ctrl_dt × action_repeat = 0.05s` | 策略网络推理间隔 (20 Hz) |

**每次策略推理推进的总物理仿真步数** = `action_repeat × n_substeps = 1 × 5 = 5` 步。

---

## 完整调用链（逐层代码走读）

下面从最外层（策略推理）到最内层（物理仿真），逐层看代码。以 `ctrl_dt=0.01, sim_dt=0.002, action_repeat=4` 为例。

### 第 1 层：策略推理 → `action_repeat=4` 次 `env.step()`

策略网络输出一个 action 后，`EpisodeWrapper` 把**同一个 action** 重复喂给 `env.step()` 4 次：

```python
# brax/envs/wrappers/training.py:98-104
def step(self, state: State, action: jax.Array) -> State:
    def f(state, _):
      nstate = self.env.step(state, action)  # 调用环境的 step
      return nstate, nstate.reward

    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)  # 循环 4 次
    state = state.replace(reward=jp.sum(rewards, axis=0))  # 4 次 reward 求和
```

`self.action_repeat = 4`，所以 `jax.lax.scan` 循环 4 次，每次调用 `self.env.step(state, action)`。

这个 wrapper 在训练入口处被包裹上去：

```python
# brax/envs/wrappers/training.py:54
env = EpisodeWrapper(env, episode_length, action_repeat)
```

### 第 2 层：`env.step()` → 计算 `n_substeps`

每次 `env.step()` 被调用时，先算出 `n_substeps`：

```python
# mujoco_playground/_src/mjx_env.py:230-231
self._ctrl_dt = config.ctrl_dt   # 0.01  (100 Hz)
self._sim_dt = config.sim_dt     # 0.002 (500 Hz)
```

```python
# mujoco_playground/_src/mjx_env.py:262-264
@property
def dt(self) -> float:
    return self._ctrl_dt  # 返回 0.01
```

```python
# mujoco_playground/_src/mjx_env.py:271-274
@property
def n_substeps(self) -> int:
    """Number of sim steps per control step."""
    return int(round(self.dt / self.sim_dt))  # round(0.01 / 0.002) = 5
```

然后在具体环境的 `step()` 中，把 `n_substeps` 传进去：

```python
# mujoco_playground/_src/manipulation/leap_hand/reorient.py:213-215
data = mjx_env.step(
    self.mjx_model, state.data, motor_targets, self.n_substeps  # n_substeps=5
)
```

### 第 3 层：`mjx_env.step()` → 循环 `n_substeps=5` 次 `mjx.step()`

这是最内层，同一个控制信号 `action` 保持不变，连续做 5 次物理仿真：

```python
# mujoco_playground/_src/mjx_env.py:163-174
def step(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int = 1,        # 传入 5
) -> mjx.Data:
  def single_step(data, _):
    data = data.replace(ctrl=action)   # 设置控制信号（不变）
    data = mjx.step(model, data)       # MuJoCo 物理引擎推进 sim_dt=0.002s
    return data, None

  return jax.lax.scan(single_step, data, (), n_substeps)[0]  # 循环 5 次
```

每次 `mjx.step()` 推进 `sim_dt = 0.002s`，循环 5 次共推进 `0.01s = ctrl_dt`。

### 完整调用链图

```
策略网络输出 action
  │
  ▼  brax EpisodeWrapper.step()  ── jax.lax.scan 循环 action_repeat=4 次
  │
  ├── 第1次: env.step(state, action)
  │     │
  │     ▼  reorient.py step() → mjx_env.step(model, data, ctrl, n_substeps=5)
  │     │
  │     ├── mjx.step()  推进 0.002s
  │     ├── mjx.step()  推进 0.002s
  │     ├── mjx.step()  推进 0.002s
  │     ├── mjx.step()  推进 0.002s
  │     └── mjx.step()  推进 0.002s
  │     共推进 0.01s
  │
  ├── 第2次: env.step(state, action)   ... 同上 5 次 mjx.step()
  ├── 第3次: env.step(state, action)   ... 同上 5 次 mjx.step()
  └── 第4次: env.step(state, action)   ... 同上 5 次 mjx.step()

总计: 4 × 5 = 20 次 mjx.step()
总仿真时间: 20 × 0.002s = 0.04s
策略推理频率: 1/0.04 = 25 Hz
```
