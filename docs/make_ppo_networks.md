# `make_ppo_networks` 函数详解

> 源文件：`brax/training/agents/ppo/networks.py`

## 函数签名

```python
def make_ppo_networks(
    observation_size: types.ObservationSize,                          # 必须
    action_size: int,                                                  # 必须
    preprocess_observations_fn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    policy_network_kernel_init_fn = jax.nn.initializers.lecun_uniform,
    policy_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    value_network_kernel_init_fn = jax.nn.initializers.lecun_uniform,
    value_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    mean_clip_scale: float | None = None,
    mean_kernel_init_fn: networks.Initializer | None = None,
    mean_kernel_init_kwargs: Mapping[str, Any] | None = None,
) -> PPONetworks:
```

---

## 必须参数

| 参数 | 类型 | 含义 |
|---|---|---|
| `observation_size` | `int` 或 `dict` | 观测空间的维度 |
| `action_size` | `int` | 动作空间的维度 |

## 可选参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `preprocess_observations_fn` | `identity` | 观测预处理函数。训练时通常用 `running_statistics.normalize` 做在线归一化 |
| `policy_hidden_layer_sizes` | `(32, 32, 32, 32)` | 策略网络的隐藏层结构 |
| `value_hidden_layer_sizes` | `(256, 256, 256, 256, 256)` | 价值网络的隐藏层结构 |
| `activation` | `linen.swish` | 激活函数 |
| `policy_obs_key` | `'state'` | 当观测是 dict 时，策略网络使用的 key |
| `value_obs_key` | `'state'` | 当观测是 dict 时，价值网络使用的 key |
| `distribution_type` | `'tanh_normal'` | 动作分布类型：`'normal'`（无界高斯）或 `'tanh_normal'`（tanh 压缩到 [-1,1]） |
| `noise_std_type` | `'scalar'` | 噪声标准差的参数化方式 |
| `init_noise_std` | `1.0` | 初始噪声标准差 |
| `state_dependent_std` | `False` | 是否让标准差依赖于状态（而非固定可学习标量） |
| `policy_network_kernel_init_fn` | `lecun_uniform` | 策略网络权重初始化方法 |
| `value_network_kernel_init_fn` | `lecun_uniform` | 价值网络权重初始化方法 |
| `mean_clip_scale` | `None` | 均值输出的裁剪比例 |
| `mean_kernel_init_fn` | `None` | 均值输出层的权重初始化方法 |

---

## 内部逻辑（3 步）

### 第 1 步：创建动作分布

```python
if distribution_type == 'normal':
    parametric_action_distribution = NormalDistribution(event_size=action_size)
elif distribution_type == 'tanh_normal':
    parametric_action_distribution = NormalTanhDistribution(event_size=action_size)
```

- **`'normal'`**：普通高斯分布，动作范围无界。
- **`'tanh_normal'`**（默认）：高斯采样后过 `tanh`，将动作压缩到 **[-1, 1]**。机器人控制中几乎都用这个，因为关节力矩/速度有物理上限。

该分布对象的作用：
- 确定策略网络输出层的大小（`param_size` = 均值参数数 + 标准差参数数）
- 推理时用于采样动作、计算 log_prob

### 第 2 步：构建 Policy Network（Actor）

```python
policy_network = networks.make_policy_network(
    parametric_action_distribution.param_size,   # 输出维度
    observation_size,                             # 输入维度
    preprocess_observations_fn=...,
    hidden_layer_sizes=policy_hidden_layer_sizes, # 默认 (32, 32, 32, 32)
    activation=activation,                        # 默认 swish
    ...
)
```

- **功能**：输入观测 → 输出动作分布的参数（均值和标准差）。
- **默认结构**：4 层 MLP，每层 32 个神经元。
- 对于 `tanh_normal` 分布，输出维度 = `2 * action_size`（均值 + log_std）。

### 第 3 步：构建 Value Network（Critic）

```python
value_network = networks.make_value_network(
    observation_size,
    preprocess_observations_fn=...,
    hidden_layer_sizes=value_hidden_layer_sizes,  # 默认 (256, 256, 256, 256, 256)
    activation=activation,
    ...
)
```

- **功能**：输入观测 → 输出标量 V(s)（状态价值估计）。
- **默认结构**：5 层 MLP，每层 256 个神经元——比策略网络大得多，因为准确的价值估计对训练稳定性至关重要。

---

## 返回值

```python
return PPONetworks(
    policy_network=policy_network,
    value_network=value_network,
    parametric_action_distribution=parametric_action_distribution,
)
```

返回 `PPONetworks` dataclass，包含三个组件：

```
PPONetworks
├── policy_network                  # Actor:  obs → 动作分布参数
├── value_network                   # Critic: obs → V(s)
└── parametric_action_distribution  # 从分布参数 → 采样动作
```

---

## 典型使用方式

```python
import functools
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

# 1. 用 functools.partial 冻结可选参数，构建网络工厂
network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    **ppo_params.network_factory,                          # 配置文件中的超参数
    preprocess_observations_fn=running_statistics.normalize, # 观测归一化
)

# 2. 传入必须参数，实例化网络（此时权重是随机初始化的）
ppo_network = network_factory(obs_size, act_size)

# 3. 加载 checkpoint，获取训练好的权重
params = brax_load(ckpt_path)
# params[0] = normalizer_params（均值/方差）
# params[1] = network_params（网络权重）
```

> **关键**：推理时的网络结构和预处理方式必须与训练时**完全一致**，否则加载的权重无法正确使用。
