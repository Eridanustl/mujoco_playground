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
"""将 XleoMassage 任务训练得到的 Brax PPO checkpoint 转换为 ONNX 格式.

该脚本参考 brax_network_to_onnx.ipynb，专门适配 XleoMassage 双手按摩任务。
转换流程：Brax PPO checkpoint → JAX params → Flax MLP → ONNX

用法:
    python export_xleo_massage_onnx.py \
        --ckpt_path /path/to/XleoMassage-checkpoint \
        --output sim2sim/onnx/xleo_massage_policy.onnx

输出的 ONNX 模型可被 play_xleo_massage.py 加载用于 sim2sim 部署。
"""

import argparse
import functools
import os

# 抑制 JAX 的 GPU 日志和预分配，避免不必要的显存占用。
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as rt
from jax2onnx import to_onnx

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.checkpoint import load as brax_load

from mujoco_playground import manipulation
from mujoco_playground.config import manipulation_params


# ---------------------------------------------------------------------------
# Flax MLP：复现 Brax 策略网络结构
# ---------------------------------------------------------------------------


class PolicyMLP(nn.Module):
  """Flax 策略网络，与 Brax PPO 的 MLP policy 结构一一对应.

  特点:
    - 内嵌 running-statistics 归一化 (mean / std)
    - 输出 2 * action_size (均值 + log_std)，取 tanh(mean) 作为确定性动作
    - 隐藏层激活函数用 swish (与 Brax 训练一致)
  """

  layer_sizes: tuple[int, ...]
  obs_mean: jnp.ndarray | None = None
  obs_std: jnp.ndarray | None = None

  @nn.compact
  def __call__(self, x):
    # 归一化：(obs - mean) / std
    if self.obs_mean is not None and self.obs_std is not None:
      x = (x - self.obs_mean) / self.obs_std

    # MLP 前向传播：除最后一层外，每层都使用 swish 激活
    for i, size in enumerate(self.layer_sizes):
      x = nn.Dense(size, name=f"hidden_{i}")(x)
      if i < len(self.layer_sizes) - 1:
        x = nn.swish(x)

    # logits = [mean, log_std]，只取 mean 并做 tanh 限幅
    loc, _ = jnp.split(x, 2, axis=-1)
    return jnp.tanh(loc)


# ---------------------------------------------------------------------------
# 权重迁移：Brax JAX params → Flax 模型参数
# ---------------------------------------------------------------------------


def build_flax_params(
    jax_policy_params: dict,
    layer_sizes: list[int],
    state_dim: int,
    obs_mean: jnp.ndarray | None = None,
    obs_std: jnp.ndarray | None = None,
) -> dict:
  """从 Brax checkpoint 的 JAX 参数构建 Flax 模型参数字典.

  Brax 的 policy 参数结构:
    {
      'hidden_0': {'kernel': ndarray, 'bias': ndarray},
      'hidden_1': {'kernel': ndarray, 'bias': ndarray},
      ...
    }

  Flax nn.Dense 期望的参数格式完全相同 (kernel, bias)，因此可以直接复用。
  """
  # 先通过 init 获取正确的参数结构模板
  model = PolicyMLP(
      layer_sizes=tuple(layer_sizes),
      obs_mean=obs_mean,
      obs_std=obs_std,
  )
  dummy = jnp.zeros((1, state_dim))
  init_params = model.init(jax.random.PRNGKey(42), dummy)["params"]

  # 用 Brax 的权重替换模板中的值
  new_params = {}
  for layer_name in init_params:
    if layer_name in jax_policy_params:
      new_params[layer_name] = {
          "kernel": jnp.array(jax_policy_params[layer_name]["kernel"]),
          "bias": jnp.array(jax_policy_params[layer_name]["bias"]),
      }
      k_shape = jax_policy_params[layer_name]["kernel"].shape
      b_shape = jax_policy_params[layer_name]["bias"].shape
      print(f"  迁移 {layer_name}: kernel {k_shape}, bias {b_shape}")
    else:
      print(f"  [WARN] Brax 参数中未找到层 '{layer_name}'，使用初始化值")
      new_params[layer_name] = init_params[layer_name]

  print("  权重迁移完成 ✓")
  return new_params


def main():
  parser = argparse.ArgumentParser(
      description="将 XleoMassage Brax PPO checkpoint 导出为 ONNX 格式"
  )
  parser.add_argument(
      "--ckpt_path",
      type=str,
      required=True,
      help=(
          "Brax PPO checkpoint 目录路径 (如"
          " checkpoints/XleoMassage-20260319-123456)"
      ),
  )
  parser.add_argument(
      "--output",
      type=str,
      default=None,
      help=(
          "输出 ONNX 文件路径 (默认: "
          "mujoco_playground/experimental/sim2sim/onnx/xleo_massage_policy.onnx)"
      ),
  )
  args = parser.parse_args()

  # 默认输出到 sim2sim/onnx/ 目录
  if args.output is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.output = os.path.join(
        script_dir, "sim2sim", "onnx", "xleo_massage_policy.onnx"
    )
  os.makedirs(os.path.dirname(args.output), exist_ok=True)

  # ------------------------------------------------------------------
  # 1. 加载 XleoMassage 环境，获取 obs / action 维度和网络配置
  # ------------------------------------------------------------------
  env_name = "XleoMassage"
  print(f"[1/6] 加载环境 {env_name} ...")

  ppo_params = manipulation_params.brax_ppo_config(env_name)
  env_cfg = manipulation.get_default_config(env_name)
  env = manipulation.load(env_name, config=env_cfg)

  obs_size = (
      env.observation_size
  )  # dict: {"state": (92,), "privileged_state": (302,)}
  act_size = env.action_size  # 30
  state_dim = obs_size["state"][0]  # 92 — 策略网络的输入维度
  print(f"  obs_size  = {obs_size}")
  print(f"  act_size  = {act_size}")
  print(f"  state_dim = {state_dim} (策略网络输入)")

  # ------------------------------------------------------------------
  # 2. 构建 Brax PPO 网络并加载 checkpoint
  # ------------------------------------------------------------------
  print(f"[2/6] 加载 checkpoint: {args.ckpt_path} ...")

  # 构建与训练时完全相同的网络工厂
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory,
      # 训练时 normalize_observations=True，这里需要显式指定归一化函数
      preprocess_observations_fn=running_statistics.normalize,
  )
  ppo_network = network_factory(obs_size, act_size)

  # 加载 checkpoint：返回 (normalizer_params, network_params)
  params = brax_load(args.ckpt_path)
  params = (params[0], params[1])
  print("  checkpoint 加载完成 ✓")

  # ------------------------------------------------------------------
  # 3. 构建 JAX 推理函数（用于后续验证）
  # ------------------------------------------------------------------
  print("[3/6] 构建 JAX 推理函数 ...")

  make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
  inference_fn = make_inference_fn(params, deterministic=True)
  print("  JAX inference_fn 就绪 ✓")

  # ------------------------------------------------------------------
  # 4. 构建等价的 Flax 模型并迁移权重
  # ------------------------------------------------------------------
  print("[4/6] 构建 Flax 策略网络并迁移权重 ...")

  # 从 checkpoint 中提取 running-statistics 的 mean / std
  # normalizer_params 按 obs key 索引，策略网络只使用 "state"
  obs_mean = jnp.array(params[0].mean["state"])
  obs_std = jnp.array(params[0].std["state"])

  # 隐藏层大小从 PPO config 中读取 (XleoMassage: (512, 256, 128))
  hidden_sizes = list(ppo_params.network_factory.policy_hidden_layer_sizes)
  print(f"  hidden_layer_sizes = {hidden_sizes}")

  # 输出层大小 = action_size * 2 (mean + log_std)
  layer_sizes = hidden_sizes + [act_size * 2]

  flax_model = PolicyMLP(
      layer_sizes=tuple(layer_sizes),
      obs_mean=obs_mean,
      obs_std=obs_std,
  )

  # 从 Brax checkpoint 构建 Flax 参数
  # params[1] 结构: {'params': {'hidden_0': {...}, 'hidden_1': {...}, ...}}
  flax_params = build_flax_params(
      jax_policy_params=params[1]["params"],
      layer_sizes=layer_sizes,
      state_dim=state_dim,
      obs_mean=obs_mean,
      obs_std=obs_std,
  )

  # 验证 Flax 模型能正常前向传播
  dummy_input = jnp.zeros((1, state_dim))
  flax_output = flax_model.apply({"params": flax_params}, dummy_input)
  print(f"  Flax 模型输出 shape: {flax_output.shape}")

  # ------------------------------------------------------------------
  # 5. 导出 ONNX
  # ------------------------------------------------------------------
  print(f"[5/6] 导出 ONNX → {args.output} ...")

  # 定义纯函数用于 ONNX 导出：将模型参数闭包化
  def policy_fn(obs):
    return flax_model.apply({"params": flax_params}, obs)

  # 使用 jax2onnx 将 JAX 函数转换为 ONNX
  onnx_model = to_onnx(
      policy_fn,
      [jnp.zeros((1, state_dim))],  # 示例输入，用于推断 shape/dtype
  )

  # 重命名输入输出节点，使其与下游部署脚本兼容
  onnx_model.graph.input[0].name = "obs"
  onnx_model.graph.output[0].name = "continuous_actions"
  # 同时更新引用了旧名字的节点
  for node in onnx_model.graph.node:
    for i, inp in enumerate(node.input):
      if inp == onnx_model.graph.input[0].name:
        pass  # 已经是 "obs"
    for i, out in enumerate(node.output):
      if out == onnx_model.graph.output[0].name:
        pass  # 已经是 "continuous_actions"

  onnx.save(onnx_model, args.output)
  print(f"  ONNX 导出完成 ✓  →  {args.output}")

  # ------------------------------------------------------------------
  # 6. 验证：对比 JAX vs ONNX 推理结果
  # ------------------------------------------------------------------
  print("[6/6] 验证 JAX vs ONNX 一致性 ...")

  # 用全 1 向量做测试输入
  test_np = np.ones((1, state_dim), dtype=np.float32)

  # JAX 推理：inference_fn 接收 obs dict
  jax_obs = {
      "state": jnp.ones(obs_size["state"]),
      "privileged_state": jnp.zeros(obs_size["privileged_state"]),
  }
  jax_pred, _ = inference_fn(jax_obs, jax.random.PRNGKey(0))
  jax_pred = np.array(jax_pred)

  # ONNX 推理
  onnx_session = rt.InferenceSession(
      args.output, providers=["CPUExecutionProvider"]
  )
  onnx_pred = onnx_session.run(["continuous_actions"], {"obs": test_np})[0][0]

  max_diff = np.max(np.abs(jax_pred - onnx_pred))
  print(f"  JAX  output (前5): {jax_pred[:5]}")
  print(f"  ONNX output (前5): {onnx_pred[:5]}")
  print(f"  max |JAX - ONNX|  = {max_diff:.2e}")

  if max_diff < 1e-5:
    print("  一致性检查通过 ✓ (diff < 1e-5)")
  elif max_diff < 1e-3:
    print("  一致性检查警告 ⚠ (1e-5 < diff < 1e-3，精度可接受)")
  else:
    print("  一致性检查失败 ✗ (diff >= 1e-3，请检查权重迁移)")

  print(f"\n完成！ONNX 模型已保存到: {args.output}")
  print("可复制到 sim2sim/onnx/ 并运行:")
  print("  python play_xleo_massage.py")


if __name__ == "__main__":
  main()
