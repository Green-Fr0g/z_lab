# z_lab

## 项目概览

`z_lab` 是基于 Isaac Lab 3.0.0 的人形机器人强化学习项目，支持 Unitree G1（29 和 23 自由度）以及 DR02。任务覆盖速度跟踪、对抗式动作模仿、全身动作跟踪和基于世界模型的策略训练。

| 任务 | 用途 | 代表性任务 ID |
| --- | --- | --- |
| Velocity | 跟踪机身速度指令 | `ZLab-Velocity-Unitree-G1` |
| AMP | 用参考动作约束行走风格 | `ZLab-AMP-Unitree-G1` |
| Mimic | 跟踪全身参考动作 | `ZLab-Mimic-Unitree-G1` |
| SuperMimic | 联合训练可微世界模型和多编码器策略 | `ZLab-SuperMimic-GenCoord-G1` |

运行 `./z_lab.sh -l --show_presets` 查看完整任务 ID 和物理后端预设。

## 版本与安装

项目使用 Python **3.12** 和 Isaac Lab **3.0.0**。安装 uv 和 Git LFS 后，在仓库根目录运行以下命令。 [uv 安装指南](https://docs.astral.sh/uv/getting-started/installation/) · [Git LFS](https://git-lfs.com/)

```bash
./z_lab.sh -i
```

脚本拉取 Git LFS 资产，创建 `.venv`，并按 `uv.lock` 安装 Isaac Lab、本地可编辑的 RSL-RL（含 AMP-PPO）和 SuperMimic，以及浏览器可视化所需的 Viser。Isaac Lab 从官方 Python 包安装。直接使用 `uv run` 或 `z_lab.sh` 即可进入项目环境。

已有项目环境时，可用 `./z_lab.sh --install-rsl-rl` 单独重装本地 RSL-RL。

### 资产转换依赖（Kitless）

项目的 `importers` extra 对应 `isaaclab[importers]==3.0.0`，提供 URDF 和 MuJoCo MJCF 的独立导入器。运行转换示例时，`uv run --extra importers` 会按需安装该依赖。转换可以在未安装 Isaac Sim 或未启动 Kit 的环境中运行。[转换教程](#在-kitless-环境中将-urdf-或-mjcf-转换为-usd)

### 选择物理后端

默认后端是 Newton/MJWarp，可直接运行无界面仿真。也可以根据任务的预设选择 Isaac Sim PhysX 或 OvPhysX，并通过相应的 extra 安装运行时。

```bash
# Newton/MJWarp
uv run isaaclab zero_agent --task ZLab-Velocity-Unitree-G1 --num_envs 4 --max_steps 100 --viz none

# Isaac Sim PhysX，使用 Kit 窗口
uv run --extra isaacsim isaaclab zero_agent --task ZLab-Velocity-Unitree-G1 --num_envs 4 --max_steps 100 --viz kit physics=isaacsim_physx

# OvPhysX，无界面
uv run --extra ovphysx isaaclab zero_agent --task ZLab-Velocity-Unitree-G1 --num_envs 4 --max_steps 100 --viz none physics=ovphysx
```

## 训练与播放

### Velocity、AMP 和 Mimic

这三类任务使用 RSL-RL。`./z_lab.sh -t` 和 `./z_lab.sh -p` 分别调用 Isaac Lab 的训练和播放入口。播放默认使用 Newton GL，可通过 `--viz` 选择其他可视化方式。

```bash
# 训练速度跟踪策略并播放最近的检查点
./z_lab.sh -t --task ZLab-Velocity-Unitree-G1 --viz none
./z_lab.sh -p --task ZLab-Velocity-Unitree-G1 --checkpoint latest

# 使用任务自带的参考动作训练 AMP 或 Mimic
./z_lab.sh -t --task ZLab-AMP-Unitree-G1 --viz none
./z_lab.sh -t --task ZLab-Mimic-Unitree-G1 --viz none
```

AMP 的参考动作目录由 agent 配置中的 `amp_motion_files` 指定。Mimic 可以通过 `env.commands.motion.motion_file=/path/to/motion.npz` 指定参考动作。播放时也可将 `--checkpoint latest` 换成具体的检查点路径。

### SuperMimic

SuperMimic 使用独立的训练和播放脚本。每种机器人都有 GenCoord 和 MaxCoord 任务。GenCoord 用根连杆和关节状态描述机器人，MaxCoord 用跟踪连杆的位姿和速度描述机器人。

```bash
# 训练世界模型和策略
uv run --extra supermimic python scripts/supermimic/train.py --task ZLab-SuperMimic-GenCoord-G1 --viz none

# 在浏览器中播放参考动作跟踪策略
uv run --extra supermimic --extra viser python scripts/supermimic/play.py --task ZLab-SuperMimic-GenCoord-G1 --checkpoint latest --viz viser
```

训练会保存 `model_<iteration>.pt` 和 `world_model.pt`。用 `--checkpoint` 加载完整检查点并继续训练，或用 `--world-model-checkpoint` 初始化世界模型并从第 0 轮训练新策略。两种参数均接受文件路径或 `latest`，每次训练选择其中一种。

播放默认使用 Motion encoder，播放当前参考片段，并显示蓝色参考动作影子和橙色单步世界模型预测。使用 `--task-encoder velocity` 可切换到速度控制，使用 `--velocity-command 0.3 0 0` 可指定固定的前向速度、侧向速度和偏航角速度。两种模式均可用 `--num-steps` 限制播放步数。

### 多 GPU、基准测试和日志

```bash
# 使用 RSL-RL 在两张 GPU 上训练
uv run isaaclab train_multigpu --rl_library rsl_rl --task ZLab-Velocity-Unitree-G1 --num_gpus 2

# 测量仿真吞吐和训练耗时
uv run isaaclab benchmark runtime --task ZLab-Velocity-Unitree-G1 --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library rsl_rl --task ZLab-Velocity-Unitree-G1 --max_iterations 10

# 查看训练曲线
uv run tensorboard --logdir logs
```

## 项目结构

任务代码位于 `src/z_lab/tasks`。locomotion 任务的公共环境配置在 `locomotion_env_cfg.py` 中，公共 MDP 项位于 `mdp` 目录。三个机器人分别拥有独立的环境注册、机器人场景和 RSL-RL 配置：

```text
src/z_lab/
├── assets/
│   ├── data/
│   │   ├── models/{g1,g1_23,dr02}/*.usd
│   │   └── motions/config/*.yaml
│   └── robots/{g1.py,g1_23.py,dr02.py}
└── tasks/locomotion/
    ├── locomotion_env_cfg.py
    ├── mdp/
    └── config/
        ├── g1/
        ├── g1_23dof/
        └── dr02/
```

AMP 任务和机器人专用配置位于 `src/z_lab/tasks/amp/config/{g1,g1_23,dr02}`，对应的参考动作 NPZ 保存在各自的 `motions` 目录中。AMP 任务默认使用 `newton_mjwarp`；训练时可通过 `physics=<预设名称>` 选择其他已注册物理后端。

新增机器人时，在 `assets/robots` 增加对应的资产配置，并在 `tasks/locomotion/config` 下增加独立的机器人配置包。机器人自己的关节名称、动作顺序和奖励权重应在该机器人的配置范围内设置；公共 locomotion 配置只保存通用环境逻辑。

## 项目资产

项目自有的 USD、动作数据和其他资源统一放在 `src/z_lab/assets/data` 下。资产模块提供了一个在 editable checkout 和安装后的 wheel 中都有效的路径：

```python
from z_lab.assets import Z_LAB_ASSETS_DIR

robot_usd_path = Z_LAB_ASSETS_DIR / "models" / "g1" / "g1_29dof_rev_1_0.usd"
```

需要传入字符串的配置字段应使用 `str(robot_usd_path)`。项目的 `.gitattributes` 已为 USD 等大文件配置 Git LFS；添加大型二进制资产前请先安装 Git LFS。

### 在 Kitless 环境中将 URDF 或 MJCF 转换为 USD

在项目根目录运行脚本。`--out_dir` 指定输出目录，`--urdf_dir` 和 `--xml_dir` 可以单独使用，也可以一起传入。输入路径可以是单个文件或目录，目录会递归查找模型。扫描 XML 目录时，脚本会跳过 ROS 的 `package.ros1.xml`、`package.ros2.xml` 等根元素并非 `<mujoco>` 的文件。

```bash
zoo=/home/zhang/code/deploy/rl_sar/src/rl_sar_zoo

# 批量转换 G1 的 MuJoCo MJCF 文件
uv run --extra importers python scripts/convert_to_usd.py \
  --xml_dir "$zoo/g1_description/mjcf" --out_dir outputs/usd

# 转换引用 package:// 网格的 Go2 URDF
uv run --extra importers python scripts/convert_to_usd.py \
  --urdf_dir "$zoo/go2_description/urdf/go2_description.urdf" --out_dir outputs/usd
```

保持模型目录结构。Isaac Lab 导入器会沿 URDF 的上级目录查找 `package://` 引用的网格，因此 `rl_sar_zoo` 的 Go2 可以直接转换。MJCF 的网格、纹理和 `<include>` 文件按原相对路径读取，物理场景设置由 Isaac Lab 任务提供。

脚本会打印生成的 `.usda` 入口路径，重复转换时输出目录可能带编号。接入任务时，将生成的资产子目录连同 `payloads/` 等引用文件一起移入 `src/z_lab/assets/data/models/`，再把入口路径填入对应机器人配置的 `UsdFileCfg.usd_path`。[Isaac Lab 资产导入文档](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html)

## 致谢

感谢以下开源项目的作者、维护者和贡献者。`z_lab` 的仿真环境、训练算法和可视化得益于他们的工作。

- Isaac Lab 提供了任务配置、仿真接口和强化学习工作流。 [Isaac Lab 仓库](https://github.com/isaac-sim/IsaacLab)
- SuperMimic 提供了可微世界模型和多编码器策略的训练框架。本项目将其从 mjlab 迁移到了 Isaac Lab。 [SuperMimic 仓库](https://github.com/HybridRobotics/supermimic) · [mjlab 仓库](https://github.com/mujocolab/mjlab)
- RSL-RL 提供了策略训练框架。 [RSL-RL 仓库](https://github.com/leggedrobotics/rsl_rl)
- 感谢 unitree_rl_lab、unitree_rl_mjlab 和 robot_lab 开源了机器人运动控制任务和训练实现。 [unitree_rl_lab 仓库](https://github.com/unitreerobotics/unitree_rl_lab) · [unitree_rl_mjlab 仓库](https://github.com/unitreerobotics/unitree_rl_mjlab) · [robot_lab 仓库](https://github.com/fan-ziqi/robot_lab)
- AMP_mjlab 为本项目的 AMP 观测、奖励和训练配置提供了参考。 [AMP_mjlab 仓库](https://github.com/ccrpRepo/AMP_mjlab)
- Newton 提供了物理仿真后端，Viser 提供了浏览器中的策略回放可视化。 [Newton 仓库](https://github.com/newton-physics/newton) · [Viser 仓库](https://github.com/viser-project/viser)
