# Massage Demo Deployment Guide

本文档介绍如何部署和运行按摩演示系统，包括启动 MuJoCo MInK 仿真环境、部署策略模型以及控制按摩的开始与停止。
## 前置条件
- 已编译好的 ROS2 工作空间 (`~/your_ros2_ws`)
- Python 虚拟环境 (`~/xleo_rl/.venv`)
- 训练好的模型 checkpoint
- 按摩轨迹 JSON 文件
## 步骤一：启动仿真环境（Terminal 1）
```bash
source ~/your_ros2_ws/install/setup.bash
ros2 launch model_interface massage_demo_mink.launch.py
```

该命令会启动 MuJoCo MInK 按摩演示的 launch 文件，加载机器人模型和仿真场景。
## 步骤二：部署策略模型（Terminal 2）
```bash
source ~/your_ros2_ws/install/setup.bash
source ~/xleo_rl/.venv/bin/activate
python3 model_interface/scripts/policy_deploy_xleohand_massage_wrist.py \
  --checkpoint_path /home/yuwei/xleo_rl/logs/XleohandMassageWrist-20260307-142812/checkpoints \
  --trajectory_json /home/yuwei/xleo_rl/mujoco_playground/collected_data/massage_data/extracted_data_json/shoufa123_jingyiyang_1127/rightarm_shoufa1/nierou_shangbizhongdianwaice_03.json \
  --hand left \
  --arm none \
  --loop_trajectory \
  --frequency 100
```
### 参数说明
| 参数 | 说明 |
|------|------|
| `--checkpoint_path` | 训练好的模型 checkpoint 路径 |
| `--trajectory_json` | 按摩轨迹 JSON 文件路径 |
| `--hand` | 使用的手，可选 `left` / `right` |
| `--arm` | 使用的臂，此处设为 `none` |
| `--loop_trajectory` | 循环执行轨迹 |
| `--frequency` | 控制频率（Hz） |
## 步骤三：控制按摩流程（Terminal 3）
### 开始按摩
```bash
ros2 topic pub /massage_start std_msgs/msg/Bool "{data: true}" --once --qos-durability transient_local
```
### 停止按摩
```bash
ros2 topic pub /massage_end std_msgs/msg/Bool "{data: true}" --once --qos-durability transient_local
```
> **注意：** 使用 `--qos-durability transient_local` 确保消息即使在订阅者稍后连接时也能被接收到。
## 操作流程总结
```
Terminal 1: 启动仿真环境
     ↓
Terminal 2: 部署策略模型（等待仿真就绪后启动）
     ↓
Terminal 3: 发送开始/停止指令控制按摩
```
