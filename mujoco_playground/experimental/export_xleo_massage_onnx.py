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
转换流程：Brax PPO checkpoint → JAX params → TensorFlow MLP → ONNX

用法:
    python export_xleo_massage_onnx.py \
        --ckpt_path /path/to/XleoMassage-checkpoint \
        --output sim2sim/onnx/xleo_massage_policy.onnx

输出的 ONNX 模型可被 play_xleo_massage.py 加载用于 sim2sim 部署。
"""

import argparse
import functools
import os

# 抑制 TF/JAX 的 GPU 日志和预分配，避免不必要的显存占用。
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jp
import numpy as np
import onnxruntime as rt
import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.checkpoint import load as brax_load

from mujoco_playground import manipulation
from mujoco_playground.config import manipulation_params


# ---------------------------------------------------------------------------
# TensorFlow MLP：复现 Brax 策略网络结构
# ---------------------------------------------------------------------------

class PolicyMLP(tf.keras.Model):
  """TensorFlow 策略网络，与 Brax PPO 的 MLP policy 结构一一对应.

  特点:
    - 内嵌 running-statistics 归一化 (mean / std)
    - 输出 2 * action_size (均值 + log_std)，取 tanh(mean) 作为确定性动作
    - 隐藏层激活函数用 swish (与 Brax 训练一致)
  """

  def __init__(
      self,
      layer_sizes: list[int],
      activation=tf.nn.swish,
      mean_std: tuple[tf.Tensor, tf.Tensor] | None = None,
  ):
    super().__init__()
    self._mean = None
    self._std = None
    if mean_std is not None:
      # 保存归一化参数：推理时对输入做 (obs - mean) / std
      self._mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
      self._std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)

    # 构建 MLP 层序列，与 Brax flax MLP 的层名 hidden_0, hidden_1, ... 对应
    self._mlp = tf.keras.Sequential(name="MLP_0")
    for i, size in enumerate(layer_sizes):
      self._mlp.add(layers.Dense(
          size,
          activation=activation,
          kernel_initializer="lecun_uniform",
          name=f"hidden_{i}",
          use_bias=True,
      ))
    # 最后一层不要激活函数（输出原始 logits）
    if self._mlp.layers:
      last = self._mlp.layers[-1]
      if hasattr(last, "activation") and last.activation is not None:
        last.activation = None

    self.submodules = [self._mlp]

  def call(self, inputs):
    if isinstance(inputs, list):
      inputs = inputs[0]
    # 归一化：(obs - mean) / std
    if self._mean is not None and self._std is not None:
      inputs = (inputs - self._mean) / self._std
    logits = self._mlp(inputs)
    # logits = [mean, log_std]，只取 mean 并做 tanh 限幅
    loc, _ = tf.split(logits, 2, axis=-1)
    return tf.tanh(loc)


# ---------------------------------------------------------------------------
# 权重迁移：JAX params → TensorFlow 模型
# ---------------------------------------------------------------------------

def transfer_weights(
    jax_params: dict,
    tf_model: PolicyMLP,
) -> None:
  """将 JAX (Flax) 的网络参数复制到对应的 TensorFlow Dense 层.

  JAX 参数结构示例:
    {
      'hidden_0': {'kernel': ndarray, 'bias': ndarray},
      'hidden_1': {'kernel': ndarray, 'bias': ndarray},
      'hidden_2': {'kernel': ndarray, 'bias': ndarray},
    }

  TF 模型中对应的层名为 MLP_0/hidden_0, MLP_0/hidden_1, ...
  """
  for layer_name, layer_params in jax_params.items():
    try:
      tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
    except ValueError:
      print(f"  [WARN] TF 模型中未找到层 '{layer_name}'，跳过")
      continue

    if isinstance(tf_layer, tf.keras.layers.Dense):
      kernel = np.array(layer_params["kernel"])
      bias = np.array(layer_params["bias"])
      tf_layer.set_weights([kernel, bias])
      print(f"  迁移 {layer_name}: kernel {kernel.shape}, bias {bias.shape}")
    else:
      print(f"  [WARN] 未处理的层类型 {layer_name}: {type(tf_layer)}")

  print("  权重迁移完成 ✓")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(
      description="将 XleoMassage Brax PPO checkpoint 导出为 ONNX 格式"
  )
  parser.add_argument(
      "--ckpt_path",
      type=str,
      required=True,
      help="Brax PPO checkpoint 目录路径 (如 checkpoints/XleoMassage-20260319-123456)",
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

  obs_size = env.observation_size   # dict: {"state": (92,), "privileged_state": (302,)}
  act_size = env.action_size        # 30
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
  # 4. 构建等价的 TensorFlow 模型并迁移权重
  # ------------------------------------------------------------------
  print("[4/6] 构建 TF 策略网络并迁移权重 ...")

  # 从 checkpoint 中提取 running-statistics 的 mean / std
  # normalizer_params 按 obs key 索引，策略网络只使用 "state"
  mean = params[0].mean["state"]
  std = params[0].std["state"]
  mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))

  # 隐藏层大小从 PPO config 中读取 (XleoMassage: (512, 256, 128))
  hidden_sizes = list(ppo_params.network_factory.policy_hidden_layer_sizes)
  print(f"  hidden_layer_sizes = {hidden_sizes}")

  # 输出层大小 = action_size * 2 (mean + log_std)
  tf_policy = PolicyMLP(
      layer_sizes=hidden_sizes + [act_size * 2],
      activation=tf.nn.swish,
      mean_std=mean_std,
  )

  # 用零输入触发模型构建（Keras lazy build）
  dummy_input = tf.zeros((1, state_dim))
  _ = tf_policy(dummy_input)

  # 迁移 JAX 权重到 TF 模型
  # params[1] 结构: {'params': {'hidden_0': {...}, 'hidden_1': {...}, ...}}
  transfer_weights(params[1]["params"], tf_policy)

  # ------------------------------------------------------------------
  # 5. 导出 ONNX
  # ------------------------------------------------------------------
  print(f"[5/6] 导出 ONNX → {args.output} ...")

  # 定义输入签名：(batch=1, state_dim) float32, 命名为 "obs"
  spec = [tf.TensorSpec(shape=(1, state_dim), dtype=tf.float32, name="obs")]
  tf_policy.output_names = ["continuous_actions"]

  model_proto, _ = tf2onnx.convert.from_keras(
      tf_policy,
      input_signature=spec,
      opset=11,  # opset 11 与 Isaac Lab 兼容
      output_path=args.output,
  )
  print(f"  ONNX 导出完成 ✓  →  {args.output}")

  # ------------------------------------------------------------------
  # 6. 验证：对比 JAX vs ONNX 推理结果
  # ------------------------------------------------------------------
  print("[6/6] 验证 JAX vs ONNX 一致性 ...")

  # 用全 1 向量做测试输入
  test_np = np.ones((1, state_dim), dtype=np.float32)

  # JAX 推理：inference_fn 接收 obs dict
  jax_obs = {
      "state": jp.ones(obs_size["state"]),
      "privileged_state": jp.zeros(obs_size["privileged_state"]),
  }
  jax_pred, _ = inference_fn(jax_obs, jax.random.PRNGKey(0))
  jax_pred = np.array(jax_pred)

  # ONNX 推理
  onnx_session = rt.InferenceSession(
      args.output, providers=["CPUExecutionProvider"]
  )
  onnx_pred = onnx_session.run(
      ["continuous_actions"], {"obs": test_np}
  )[0][0]

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
