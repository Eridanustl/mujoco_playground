# Brax Distribution 模块详解
> 源文件：`.venv/lib/python3.12/site-packages/brax/training/distribution.py`
>
> 本文档逐层剖析 Brax 强化学习框架中动作分布的实现，涵盖数学推导和代码解析。
---
## 整体架构
```
                    ParametricDistribution（抽象基类）
                    ┌─────────────────────────────┐
                    │ sample()                     │
                    │ sample_no_postprocessing()   │
                    │ log_prob()                   │
                    │ entropy()                    │
                    │ mode()                       │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                 ▼
   NormalTanhDistribution              NormalDistribution
   ┌──────────────────────┐           ┌─────────────────┐
   │ param_size = 2*action│           │ param_size=action│
   │ postprocessor = Tanh │           │ postprocessor=Id │
   │                      │           │                  │
   │ create_dist():       │           │ create_dist():   │
   │   split → (μ, raw_σ) │           │   unpack (μ, σ)  │
   │   σ = softplus+min   │           │                  │
   └──────────┬───────────┘           └────────┬────────┘
              │                                │
              ▼                                ▼
         _NormalDistribution（共享）
         ┌────────────────────────┐
         │ sample(): μ + σε       │
         │ mode(): μ              │
         │ log_prob(): 高斯pdf    │
         │ entropy(): 高斯熵      │
         │ kl_divergence(): KL散度│
         └────────────────────────┘
   TanhBijector                    IdentityPostprocessor
   ┌──────────────────┐            ┌────────────────────┐
   │ forward: tanh     │            │ forward: x          │
   │ inverse: arctanh  │            │ inverse: x          │
   │ log_det_jacobian: │            │ log_det_jacobian: 0 │
   │   log(1-tanh²(x)) │            │                    │
   └──────────────────┘            └────────────────────┘
```
---
## 第一层：`_NormalDistribution` — 底层正态分布
手写的正态分布实现（不依赖 TensorFlow Probability 等外部库），提供四个核心操作。
### `sample` — 重参数化采样
```python
def sample(self, seed):
    return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc
```
重参数化采样（Reparameterization Trick）：
$$u = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$
写成 `μ + σε` 而不是直接从 N(μ,σ) 采样，是为了让**梯度能够穿过采样操作反传到 μ 和 σ**。如果直接采样，采样过程不可微，梯度会断掉。
### `mode` — 众数
```python
def mode(self):
    return self.loc
```
正态分布的众数（概率密度最大值处）就是均值 μ。用于**确定性推理**（eval 时不加噪声）。
### `log_prob` — 对数概率密度
```python
def log_prob(self, x):
    log_unnormalized = -0.5 * jnp.square(x / self.scale - self.loc / self.scale)
    log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
    return log_unnormalized - log_normalization
```
正态分布 pdf：
$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
取对数：
$$\log p(x) = \underbrace{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}_{\texttt{log\_unnormalized}} - \underbrace{\left(\frac{1}{2}\log(2\pi) + \log\sigma\right)}_{\texttt{log\_normalization}}$$
代码把 `(x-μ)/σ` 拆成了 `x/scale - loc/scale`，数学上等价，但避免了先算差再除时中间结果可能溢出的问题。
### `entropy` — 信息熵
```python
def entropy(self):
    log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
    entropy = 0.5 + log_normalization
    return entropy * jnp.ones_like(self.loc)
```
正态分布的熵有封闭解：
$$H = \frac{1}{2}\ln(2\pi\sigma^2) + \frac{1}{2} = \frac{1}{2}\ln(2\pi) + \ln\sigma + \frac{1}{2}$$
`jnp.ones_like(self.loc)` 确保输出形状和 μ 一致（广播保形状）。
### `kl_divergence` — KL 散度
```python
def kl_divergence(self, old_dist):
    """Computes KL(old_dist || self)."""
    return jnp.sum(
        jnp.log(self.scale / old_dist.scale + 1e-5)
        + (jnp.square(old_dist.scale) + jnp.square(old_dist.loc - self.loc))
        / (2.0 * jnp.square(self.scale))
        - 0.5,
        axis=-1,
    )
```
两个正态分布之间的 KL 散度封闭解：
$$\text{KL}(q \| p) = \log\frac{\sigma_p}{\sigma_q} + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{2\sigma_p^2} - \frac{1}{2}$$
其中 `self` 是 p（新策略），`old_dist` 是 q（旧策略），计算 KL(old || new)。
`+ 1e-5` 在 log 里面防止 `self.scale / old_dist.scale = 0` 时取 log 爆炸。
**用途**：PPO 训练时监控策略更新幅度，以及自适应 KL 学习率调度。
---
## 第二层：后处理器（Postprocessor / Bijector）
### `TanhBijector` — tanh 变换
```python
class TanhBijector:
    def forward(self, x):         # u → a
        return jnp.tanh(x)
    def inverse(self, y):         # a → u
        return jnp.arctanh(y)
    def forward_log_det_jacobian(self, x):
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))
```
`forward_log_det_jacobian` 是最关键的部分。推导过程：
$$\frac{d\tanh(x)}{dx} = 1 - \tanh^2(x)$$
$$\log|J| = \log(1 - \tanh^2(x))$$
但直接算 `log(1 - tanh²(x))` 有数值问题：当 x 很大时 tanh(x) ≈ 1，1-1=0，log(0) = -∞。
用恒等式变换到数值稳定形式：
$$1 - \tanh^2(x) = \text{sech}^2(x) = \frac{4}{(e^x + e^{-x})^2} = \frac{4e^{-2x}}{(1 + e^{-2x})^2}$$
取 log：
$$\log(1-\tanh^2(x)) = \log 4 - 2x - 2\log(1+e^{-2x}) = 2\log 2 - 2x - 2\,\text{softplus}(-2x)$$
代码写的是 `2.0 * (log(2.0) - x - softplus(-2.0 * x))`，展开为 `2log2 - 2x - 2softplus(-2x)`，完全一致。
`softplus(z) = log(1 + e^z)` 在 JAX 中有数值稳定的实现，所以整个表达式在 x 很大时也不会炸。
### `IdentityPostprocessor` — 恒等变换
```python
class IdentityPostprocessor:
    def forward(self, x):
        return x                     # 不做变换
    def inverse(self, x):
        return x
    def forward_log_det_jacobian(self, x):
        return jnp.zeros_like(x)    # 恒等变换的雅可比 = 1，log(1) = 0
```
什么都不做。给 `NormalDistribution`（不需要 tanh 压缩）使用。
---
## 第三层：`ParametricDistribution` — 统一抽象基类
策略模式（Strategy Pattern）的体现：把"底层分布"和"后处理变换"组合在一起，对外暴露统一接口。
### 构造函数
```python
def __init__(self, param_size, postprocessor, event_ndims, reparametrizable):
    self._param_size = param_size       # 网络需要输出多少维参数
    self._postprocessor = postprocessor # TanhBijector 或 IdentityPostprocessor
    self._event_ndims = event_ndims     # 动作的 rank（0=标量，1=向量）
    self._reparametrizable = reparametrizable  # 是否支持重参数化
```
### `sample_no_postprocessing` — 不做后处理的采样
```python
def sample_no_postprocessing(self, parameters, seed):
    return self.create_dist(parameters).sample(seed=seed)
    # 返回 u = μ + σε，即 raw_action
```
### `sample` — 完整采样
```python
def sample(self, parameters, seed):
    return self.postprocess(self.sample_no_postprocessing(parameters, seed))
    # 返回 tanh(u) 或 u（取决于 postprocessor）
```
完整链路：`parameters → create_dist → sample → postprocess`
### `mode` — 确定性输出
```python
def mode(self, parameters):
    return self.postprocess(self.create_dist(parameters).mode())
    # 对 tanh_normal：返回 tanh(μ)
    # 对 normal：返回 μ
```
用于确定性推理（eval 时 `deterministic=True`）。
### `log_prob` — 对数概率（核心方法）
```python
def log_prob(self, parameters, actions):
    dist = self.create_dist(parameters)
    log_probs = dist.log_prob(actions)                                # (1)
    log_probs -= self._postprocessor.forward_log_det_jacobian(actions) # (2)
    if self._event_ndims == 1:
        log_probs = jnp.sum(log_probs, axis=-1)                      # (3)
    return log_probs
```
三步拆解：
1. **`dist.log_prob(actions)`**：`actions` 是 `raw_action`（tanh 之前的 u），计算每个动作维度的 log N(u_i | μ_i, σ_i)，形状 `[batch, action_dim]`
2. **`-= forward_log_det_jacobian(actions)`**：变量替换修正
   - 对 `tanh_normal`：减去 log(1 - tanh²(u_i))
   - 对 `normal`：减去 0，无影响
   对应的数学公式：
   $$\log\pi(a_i|s) = \log\mathcal{N}(u_i|\mu_i,\sigma_i) - \log(1-\tanh^2(u_i))$$
3. **`jnp.sum(..., axis=-1)`**：各动作维度独立，联合概率的 log = 各维 log 之和：
   $$\log\pi(\mathbf{a}|s) = \sum_i \log\pi(a_i|s)$$
   输出形状从 `[batch, action_dim]` 变成 `[batch]`。
**重要**：`actions` 参数必须传 `raw_action`（tanh 之前的值）。如果传入 tanh 之后的值，需要先 `arctanh` 反变换，在边界附近会数值爆炸。
### `entropy` — 信息熵
```python
def entropy(self, parameters, seed):
    dist = self.create_dist(parameters)
    entropy = dist.entropy()                                     # 正态分布熵（封闭解）
    entropy += self._postprocessor.forward_log_det_jacobian(     # 加雅可比修正
        dist.sample(seed=seed)                                   # 需要采样来估计
    )
    if self._event_ndims == 1:
        entropy = jnp.sum(entropy, axis=-1)
    return entropy
```
**注意和 `log_prob` 的符号相反**：`log_prob` 是减，`entropy` 是加。
原因：熵的定义是 H = -E[log p]，变量替换后：
$$H(\text{tanh\_normal}) = H(\text{normal}) + \mathbb{E}[\log(1-\tanh^2(u))]$$
tanh 把概率"压缩"了，熵会变小（加的是负数，因为 log(1-tanh²) ≤ 0）。
这里用了**采样估计**（`dist.sample`），因为 tanh 变换后的熵没有封闭解。
---
## 第四层：具体分布实现
### `NormalTanhDistribution` — tanh 压缩正态分布
```python
class NormalTanhDistribution(ParametricDistribution):
    def __init__(self, event_size, min_std=0.001, var_scale=1):
        super().__init__(
            param_size=2 * event_size,      # μ 和 σ 各 event_size 维
            postprocessor=TanhBijector(),    # 后处理 = tanh
            event_ndims=1,                   # 向量动作
            reparametrizable=True,
        )
        self._min_std = min_std
        self._var_scale = var_scale
    def create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
        return _NormalDistribution(loc=loc, scale=scale)
```
`create_dist` 的数据流：
```
网络输出 [μ₁, μ₂, ..., μₙ, s₁, s₂, ..., sₙ]    （2n 维）
                           ↓ split
           loc = [μ₁, ..., μₙ]
           raw_scale = [s₁, ..., sₙ]
                           ↓ softplus + min_std
           scale = [σ₁, ..., σₙ]                   （保证 σ > min_std > 0）
```
`softplus(x) = log(1+eˣ)` 是平滑版的 ReLU，保证输出为正。再加 `min_std=0.001` 防止标准差趋近于 0（否则策略变得完全确定性，探索能力消失）。
**关于 tanh 饱和问题**（源码注释第153-159行）：
> We can't use TransformedDistribution here because of **tanh saturation** which would make log_prob computations impossible.
如果先 `tanh(u)` 得到 a ∈ (-1,1)，再想算 `log_prob(a)`，需要 `arctanh(a)` 反变换回 u。但当 a 非常接近 ±1 时，`arctanh` 会爆炸到 ±∞。所以代码的策略是**始终在 tanh 之前的空间操作**，只在最后送给环境时才做 tanh。
### `NormalDistribution` — 普通正态分布
```python
class NormalDistribution(ParametricDistribution):
    def __init__(self, event_size):
        super().__init__(
            param_size=event_size,                 # 只需要 event_size
            postprocessor=IdentityPostprocessor(), # 不做后处理
            event_ndims=1,
            reparametrizable=True,
        )
    def create_dist(self, parameters):
        return _NormalDistribution(*parameters)    # parameters = (mean, std) 元组
```
与 `NormalTanhDistribution` 的关键区别：
| | `NormalTanhDistribution` | `NormalDistribution` |
|---|---|---|
| `param_size` | `2 * event_size`（μ和σ拼在一起） | `event_size`（μ和σ由网络分开输出） |
| `create_dist` 输入 | 一个 2n 维向量，内部 split | 一个 (mean, std) 元组，直接解包 |
| 后处理 | `TanhBijector` | `IdentityPostprocessor` |
| 动作范围 | (-1, 1) | (-∞, +∞) |
---
## 各方法的调用场景
| 方法 | 调用者 | 场景 |
|---|---|---|
| `sample()` | SAC 推理 | 采样动作发给环境 |
| `sample_no_postprocessing()` | PPO 推理 / SAC 训练 | 先拿 raw_action，分步处理 |
| `postprocess()` | PPO 推理 / SAC 训练 | 对 raw_action 做 tanh |
| `log_prob()` | PPO 训练 / SAC actor & critic loss | 计算重要性权重 / 熵正则化 |
| `entropy()` | PPO 训练 | 熵奖励项 |
| `mode()` | 推理（deterministic=True） | eval 时取确定性动作 |
| `kl_divergence()` | PPO 训练（自适应 KL） | 监控策略变化幅度 |
### PPO 推理中的调用链
```python
# 步骤1：tanh 之前采样 → raw_action
raw_actions = distribution.sample_no_postprocessing(logits, key)
# 步骤2：用 raw_action 计算 log_prob（含雅可比修正）
log_prob = distribution.log_prob(logits, raw_actions)
# 步骤3：tanh 后处理 → 发给环境的 action
action = distribution.postprocess(raw_actions)
```
### SAC 推理中的调用链
```python
# 一步完成：内部 sample + tanh
action = distribution.sample(logits, key)
```
### PPO 训练中的调用
```python
# 用新策略重算旧 raw_action 的 log_prob
new_log_prob = distribution.log_prob(new_logits, old_raw_action)
old_log_prob = data['log_prob']  # 采集时存的
# 重要性采样比率
ratio = exp(new_log_prob - old_log_prob)
```
### SAC 训练中的调用
```python
# actor loss: 采样 → 算 log_prob → 算 Q → 损失
action = distribution.sample_no_postprocessing(dist_params, key)
log_prob = distribution.log_prob(dist_params, action)
action = distribution.postprocess(action)
q_value = q_network(obs, action)
loss = alpha * log_prob - min(q_value)
```
---
## raw_action vs action
| | `raw_action` | `action` |
|---|---|---|
| 定义 | u = μ + σε | a = tanh(u) |
| 值域 | (-∞, +∞) | (-1, 1) |
| 用途 | **计算 log_prob** | **发给环境执行** |
| 保存原因 | 训练时需要重新算概率 | 环境需要有界动作 |
为什么 `log_prob` 必须用 `raw_action`：因为 `log_prob` 内部先调用正态分布的 `dist.log_prob(actions)`，这里期望的输入是正态分布空间中的值（tanh 之前的 u）。如果传入 tanh 之后的 action，相当于拿被压缩过的值去算正态分布概率，结果完全错误。
