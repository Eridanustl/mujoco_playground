"""Single-step debugger for the XleoMassage environment.

Runs one reset + a few steps, printing detailed diagnostics to identify
why episodes terminate immediately.

Usage:
    python debug_massage_termination.py
"""

import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground import registry
from mujoco_playground import wrapper


def main():
  env_name = "XleoMassage"
  env_cfg = registry.get_default_config(env_name)
  env = registry.load(env_name, config=env_cfg)
  env = wrapper.wrap_for_brax_training(env, full_reset=True)

  mj = env._mj_model
  joint_ids = env._joint_ids
  joint_qids = env._joint_qids
  joint_dqids = env._joint_dqids
  jnt_range_low = mj.jnt_range[joint_ids, 0]
  jnt_range_high = mj.jnt_range[joint_ids, 1]

  print("=" * 70)
  print("Joint ranges (low, high):")
  for i, name in enumerate(
      env._mj_model.joint(joint_ids[i]).name for i in range(len(joint_ids))
  ):
    print(
        f"  [{i:2d}] {name:30s}  [{jnt_range_low[i]:+.4f},"
        f" {jnt_range_high[i]:+.4f}]"
    )
  print()

  # Reset
  # wrap_for_brax_training 使用 VmapWrapper，需要 batch 维度的 rng
  rng = jax.random.PRNGKey(0)
  rng = jax.random.split(rng, 1)  # 添加 batch 维度：shape (1, 2)
  state = jax.jit(env.reset)(rng)

  # # Check qpos right after reset
  # qpos_after_reset = np.array(state.data.qpos[joint_qids])
  # qvel_after_reset = np.array(state.data.qvel[joint_dqids])
  # print("=" * 70)
  # print("After reset:")
  # print(f"  info['step'] = {state.info['step']}")
  # print(f"  done = {state.done}")

  # violations_low = qpos_after_reset < jnt_range_low
  # violations_high = qpos_after_reset > jnt_range_high
  # print(f"  qpos below limit: {np.sum(violations_low)} joints")
  # print(f"  qpos above limit: {np.sum(violations_high)} joints")
  # for i in range(len(joint_ids)):
  #   flag = ""
  #   if violations_low[i]:
  #     flag = f" *** BELOW LIMIT by {jnt_range_low[i] - qpos_after_reset[i]:.6f}"
  #   if violations_high[i]:
  #     flag = (
  #         f" *** ABOVE LIMIT by {qpos_after_reset[i] - jnt_range_high[i]:.6f}"
  #     )
  #   if flag:
  #     print(
  #         f"    [{i:2d}] qpos={qpos_after_reset[i]:+.6f}  "
  #         f"range=[{jnt_range_low[i]:+.6f}, {jnt_range_high[i]:+.6f}]{flag}"
  #     )

  # has_nan = np.any(np.isnan(qpos_after_reset)) or np.any(
  #     np.isnan(qvel_after_reset)
  # )
  # print(f"  NaN in qpos: {np.any(np.isnan(qpos_after_reset))}")
  # print(f"  NaN in qvel: {np.any(np.isnan(qvel_after_reset))}")

  # # Show default_pose vs reset qpos difference
  # default_pose = np.array(env._default_pose)
  # action_scale = np.array(env._action_scale)
  # print(f"\n  Per-joint action_scale (action=1 offset):")
  # for i in range(len(joint_ids)):
  #   diff = qpos_after_reset[i] - default_pose[i]
  #   print(
  #       f"    [{i:2d}] default={default_pose[i]:+.4f} "
  #       f" reset_qpos={qpos_after_reset[i]:+.4f}  diff={diff:+.6f} "
  #       f" action_scale={action_scale[i]:.4f}  range=[{jnt_range_low[i]:+.4f},"
  #       f" {jnt_range_high[i]:+.4f}]"
  #   )
  # print()

  # Step with zero action (should be safest)
  step_fn = jax.jit(env.step)

  # Test with random actions to simulate 1 step of training
  print("Test with random actions to simulate 1 step of training")
  rng = jax.random.PRNGKey(42)
  rng, act_rng = jax.random.split(rng)
  action = jax.random.uniform(
      act_rng, (1, env.action_size), minval=-1.0, maxval=1.0
  )  # 添加 batch 维度：shape (1, action_size)
  state = step_fn(state, action)
  state = step_fn(state, action)

  # # Test with random actions to simulate early training
  # rng = jax.random.PRNGKey(42)
  # for step_i in range(20):
  #   rng, act_rng = jax.random.split(rng)
  #   action = jax.random.uniform(
  #       act_rng, (env.action_size,), minval=-1.0, maxval=1.0
  #   )
  #   state = step_fn(state, action)

  #   qpos_now = np.array(state.data.qpos[joint_qids])
  #   qvel_now = np.array(state.data.qvel[joint_dqids])

  #   violations_low = qpos_now < jnt_range_low
  #   violations_high = qpos_now > jnt_range_high

  #   print(f"--- Step {step_i + 1} ---")
  #   print(f"  info['step'] = {state.info['step']}")
  #   print(f"  done = {float(state.done):.4f}")
  #   print(f"  term/nan = {float(state.metrics['term/nan']):.4f}")
  #   print(f"  term/pose = {float(state.metrics['term/pose']):.4f}")
  #   print(
  #       f"  term/joint_limit = {float(state.metrics['term/joint_limit']):.4f}"
  #   )
  #   print(
  #       f"  NaN in qpos: {np.any(np.isnan(qpos_now))}, NaN in qvel:"
  #       f" {np.any(np.isnan(qvel_now))}"
  #   )
  #   print(
  #       f"  qpos below limit: {np.sum(violations_low)}, above limit:"
  #       f" {np.sum(violations_high)}"
  #   )

  #   for i in range(len(joint_ids)):
  #     flag = ""
  #     if violations_low[i]:
  #       flag = f" *** BELOW by {jnt_range_low[i] - qpos_now[i]:.6f}"
  #     if violations_high[i]:
  #       flag = f" *** ABOVE by {qpos_now[i] - jnt_range_high[i]:.6f}"
  #     if flag:
  #       print(
  #           f"    [{i:2d}] qpos={qpos_now[i]:+.6f}  "
  #           f"range=[{jnt_range_low[i]:+.6f}, {jnt_range_high[i]:+.6f}]{flag}"
  #       )

  # # Run multiple seeds to check robustness
  # print("\n" + "=" * 70)
  # print("Multi-seed test (50 steps each, random actions):")
  # for seed in range(10):
  #   rng = jax.random.PRNGKey(seed)
  #   state = jax.jit(env.reset)(rng)
  #   rng = jax.random.PRNGKey(seed + 1000)
  #   survived = 0
  #   for step_i in range(50):
  #     rng, act_rng = jax.random.split(rng)
  #     action = jax.random.uniform(act_rng, (env.action_size,), minval=-1.0, maxval=1.0)
  #     state = step_fn(state, action)
  #     if float(state.done) > 0.5:
  #       print(f"  Seed {seed}: terminated at step {step_i + 1}  "
  #             f"(nan={float(state.metrics['term/nan']):.0f}, "
  #             f"pose={float(state.metrics['term/pose']):.0f}, "
  #             f"jlimit={float(state.metrics['term/joint_limit']):.0f})")
  #       break
  #     survived += 1
  #   else:
  #     print(f"  Seed {seed}: survived all 50 steps")


if __name__ == "__main__":
  main()
