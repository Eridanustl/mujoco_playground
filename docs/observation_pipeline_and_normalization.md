# 观测值调用链与归一化机制

## 文件调用链总览

```
learning/train_jax_ppo.py                          ← 训练入口
  └→ brax/training/agents/ppo/train.py              ← PPO 训练循环
       ├→ brax/training/agents/ppo/networks.py      ← 构建网络 + 推理函数
       │    └→ brax/training/networks.py            ← MLP 定义 + apply 闭包
       └→ brax/training/acme/running_statistics.py  ← 归一化（Welford 算法）
```

---

## 第 1 站：train_jax_ppo.py

构建 `network_factory` 并传给 `ppo.train`：

```python
# 第 365-369 行：选择网络构建函数
network_fn = ppo_networks.make_ppo_networks

# 第 370-373 行：用 partial 冻结配置参数（hidden_layer_sizes 等）
network_factory = functools.partial(
    network_fn, **ppo_params.network_factory
)

# 第 387-402 行：把 network_factory 传给 ppo.train
train_fn = functools.partial(ppo.train, ..., network_factory=network_factory, ...)

# 第 468 行：开始训练
make_inference_fn, params, _ = train_fn(environment=env, ...)
```

---

## 第 2 站：ppo/train.py 中的 train() 函数

### 2a. 构建网络（第 418-426 行）

```python
# 从环境 reset 后的 obs 获取观测维度
obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

# 决定是否归一化
normalize = lambda x, y: x                    # 默认：恒等函数
if normalize_observations:
    normalize = running_statistics.normalize   # 开启归一化

# 调用 network_factory 构建网络
ppo_network = network_factory(
    obs_shape, env.action_size, preprocess_observations_fn=normalize
)
```

### 2b. 训练循环中收集数据（第 536-569 行，training_step 函数）

```python
# 用当前参数构建策略
policy = make_policy((
    training_state.normalizer_params,   # params[0]
    training_state.params.policy,       # params[1]
    training_state.params.value,        # params[2]
))

# 用策略收集 rollout，内部反复调用 policy(state.obs, key)
# state.obs 就是环境中拼接的原始观测
next_state, data = acting.generate_unroll(
    env, current_state, policy, current_key, unroll_length, ...
)
```

### 2c. 更新 normalizer 统计量（第 589-596 行）

```python
# 收集完数据后，用这批原始观测更新 mean/std
normalizer_params = running_statistics.update(
    normalizer_params,
    _remove_pixels(data.observation),
    pmap_axis_name=_PMAP_AXIS_NAME,
)
```

### 2d. SGD 训练（第 598-604 行）

```python
# 用更新后的 normalizer_params 做梯度更新
(optimizer_state, params, _), metrics = jax.lax.scan(
    functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
    ...,
    length=num_updates_per_batch,
)
```

---

## 第 3 站：ppo/networks.py 的 make_inference_fn（第 35-76 行）

策略被调用时：

```python
def policy(observations, key_sample):
    param_subset = (params[0], params[1])   # (normalizer_params, policy_params)
    logits = policy_network.apply(*param_subset, observations)
    #                              ↑↑↑         ↑↑↑          ↑↑↑
    #                    normalizer_params  policy_params  原始 obs
    ...
```

---

## 第 4 站：networks.py 的 make_policy_network 中的 apply 闭包（第 558-565 行）

**观测真正进入网络的地方：**

```python
def apply(processor_params, policy_params, obs):
    if isinstance(obs, Mapping):       # dict 观测的分支
        obs = preprocess_observations_fn(
            obs[obs_key], normalizer_select(processor_params, obs_key)
        )
    else:                               # ← 普通数组走这里
        obs = preprocess_observations_fn(obs, processor_params)
        # 即: obs = running_statistics.normalize(obs, processor_params)
        # 即: obs = (obs - mean) / std

    return policy_module.apply(policy_params, obs)
    #      ↑ 归一化后的 obs 进入 MLP
```

---

## 第 5 站：networks.py 的 MLP.__call__（第 148-165 行）

归一化后的观测逐层前传：

```python
class MLP(linen.Module):
    @linen.compact
    def __call__(self, data: jnp.ndarray):     # data = 归一化后的 obs
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            is_last = i == len(self.layer_sizes) - 1
            hidden = linen.Dense(hidden_size)(hidden)    # 线性层: W @ hidden + b
            if not is_last or self.activate_final:
                hidden = self.activation(hidden)          # swish 激活
        return hidden    # → logits: 动作分布参数
```

---

## 完整调用链图

```
环境 _get_obs() 拼接原始观测 obs_raw
            │
            ▼
train_jax_ppo.py :: train_fn(environment=env)
            │
            ▼
ppo/train.py :: training_step()
│  policy = make_policy(params)
│  data = generate_unroll(env, state, policy, ...)
│                │
│                │ 内部反复调用:
│                │   action = policy(state.obs, key)
│                │                    ↓
│                ▼
│       ppo/networks.py :: policy()                  ← 第 49 行
│       │
│       │  logits = policy_network.apply(
│       │      normalizer_params,
│       │      policy_params,
│       │      observations            ← 原始 obs
│       │  )
│       │         ↓
│       ▼
│ networks.py :: apply() 闭包                        ← 第 558 行
│ │
│ │  obs = normalize(obs, processor_params)          ← 归一化
│ │  return policy_module.apply(policy_params, obs)
│ │                              ↓
│ ▼
│ networks.py :: MLP.__call__()                      ← 第 148 行
│ │
│ │  Dense(32) → swish → Dense(32) → swish → ... → Dense(param_size)
│ │                                                       ↓
│ │                                              logits [mean, log_std]
│ ▼
│ 回到 ppo/networks.py :: policy()
│ │
│ │  raw_action = sample(logits, key)
│ │  action = tanh(raw_action)                       ← 压缩到 [-1, 1]
│ │  return action
│ ▼
│ 回到 ppo/train.py
│
│  normalizer_params = update(normalizer_params, data.observation)  ← 更新统计量
│  sgd_step(data, normalizer_params)                                ← 梯度更新
│
└── 重复直到训练结束
```

---

## 归一化机制

### Normalizer 参数（RunningStatisticsState）

训练过程中在线积累的观测统计量：

| 字段 | 类型 | 含义 |
|---|---|---|
| `mean` | `jnp.ndarray` | 观测每个维度的运行均值 |
| `std` | `jnp.ndarray` | 观测每个维度的运行标准差 |
| `count` | `UInt64` | 总共见过多少个样本 |
| `summed_variance` | `jnp.ndarray` | 方差的累计和（用于增量计算 std） |

### 归一化公式

```python
# running_statistics.py :: normalize() 第 297-313 行
obs_normalized = (obs - mean) / std
```

标准的 z-score 归一化，将观测变换到均值约 0、标准差约 1 的分布。

### 为什么需要归一化

1. **不同维度量纲差异巨大**：关节角度在 [-3.14, 3.14]，速度在 [-100, 100]，力在 [0, 500]。不归一化会导致大值维度主导梯度。
2. **激活函数对输入范围敏感**：swish/tanh 在输入接近 0 时梯度最好，输入过大会饱和。
3. **推理时必须用训练时的统计量**：否则网络输入分布与训练时不一致，策略输出错误。

### Welford 在线算法

归一化统计量的更新使用 Welford 算法（`running_statistics.py :: update()`），用 O(1) 内存精确计算运行均值和方差：

```python
# 每来一个新 batch：
count += batch_size
delta = batch_mean - old_mean
mean  = old_mean + delta / count              # 增量更新均值
M2    = M2 + (batch - old_mean) * (batch - new_mean)  # 增量更新方差
std   = sqrt(M2 / count)
```

特性：训练前 5%-10% 的时间统计量快速变化，之后趋于收敛，后期相当于固定缩放。

### 训练时的时序

```
1. 环境产生原始观测 obs_raw
       ↓
2. 用当前的 mean/std 归一化: obs_norm = (obs_raw - mean) / std
       ↓
3. 归一化后的 obs_norm 喂给 policy 和 value 网络
       ↓
4. 网络输出动作 → 环境执行 → 得到 reward 和下一个 obs
       ↓
5. 用这批新的 obs_raw 更新 mean/std（Welford 算法）
```

先用旧的统计量归一化、喂给网络，然后再用新数据更新统计量。

---

## 与 IsaacLab 的对比

IsaacLab（RSL-RL / PyTorch）提供两种观测预处理机制：

### 机制 1：手动缩放（环境侧）

在环境代码中手动乘固定常数：

```python
# isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py
obs = torch.cat((
    unscale(self.hand_dof_pos, lower, upper),        # 关节角度 → [-1, 1]
    self.cfg.vel_obs_scale * self.hand_dof_vel,       # 速度 × 0.2
    self.cfg.force_torque_obs_scale * self.force,      # 力 × 10.0
), dim=-1)
```

其中 `unscale(x, lower, upper) = (2x - upper - lower) / (upper - lower)`。

### 机制 2：运行时归一化（网络侧）

RSL-RL 的 `obs_normalization` 配置项，默认关闭：

```python
# isaaclab_rl/rsl_rl/rl_cfg.py
class RslRlMLPModelCfg:
    obs_normalization: bool = False   # 默认关闭
```

开启后效果与 Brax 的 `running_statistics.normalize` 类似，但归一化模块被封装在 `nn.Module` 内部，保存/加载时随 `state_dict` 自动处理。

### 对比总结

| | Brax（在线归一化） | IsaacLab（手动缩放） |
|---|---|---|
| **方式** | 运行时 (obs - mean) / std | 固定常数乘法 |
| **需要训练数据** | 是 | 否 |
| **参数是否变化** | 随训练不断更新 | 不变，硬编码 |
| **需保存到 checkpoint** | 需要 | 不需要 |
| **精度** | 精确，自动适应数据分布 | 粗略，靠经验设值 |
| **推理时额外处理** | 必须加载 normalizer 参数 | 无（缩放在环境代码里） |

### 架构差异（JAX vs PyTorch）

| | Brax (JAX) | RSL-RL (PyTorch) |
|---|---|---|
| 归一化存储方式 | normalizer 参数是独立的 pytree | normalizer 是 `nn.Module` 的子模块 |
| 保存 checkpoint | 手动打包 `(norm, policy, value)` | `model.state_dict()` 自动包含一切 |
| 加载 checkpoint | 手动拆分 `params[0]`, `params[1]`... | `model.load_state_dict()` 一步到位 |
| 推理代码 | 需显式传 normalizer params | 直接 `model(obs)`，归一化在内部自动发生 |

---

## TensorBoard 可视化

`train_jax_ppo.py` 中通过 `policy_params_fn` 回调记录 normalizer 的 mean/std 到 TensorBoard：

```bash
# 训练时启用 TensorBoard 日志
python learning/train_jax_ppo.py --env_name=XXX --use_tb

# 查看
tensorboard --logdir logs/
```

在 TensorBoard 的 `normalizer/` 分组下，可以看到每个观测维度的 mean 和 std 曲线随训练步数的变化。预期行为：训练初期变化较大，后期趋于收敛。

使用 `policy_params_fn` 回调的原因：这是 Brax `ppo/train.py` 暴露给外部的**唯一能拿到 normalizer_params 的回调接口**。另一个回调 `progress_fn` 只接收 metrics 字典，不包含网络参数。
