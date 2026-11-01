# z_lab

这是一个基于 Isaac Lab 3.0 的下游任务包，使用标准的 uv `src` 项目布局。

当前注册了三个基于管理器的速度跟踪 locomotion 任务：

- `ZLab-Velocity-Unitree-G1`：Unitree G1，29 个策略自由度。
- `ZLab-Velocity-Unitree-G1-23DOF`：Unitree G1，23 个策略自由度。
- `ZLab-Velocity-DR02`：DR02，29 个策略自由度。

另外注册了 `ZLab-AMP-Unitree-G1-23DOF`，使用本地 RSL-RL 5.5.1 的 AMP-PPO 算法，支持动作初始化和倒地恢复。

机器人资产使用 Isaac Lab 的隐式关节执行器。每个机器人的关节顺序、动作缩放、连杆名称、执行器参数和 USD 路径保存在 `src/z_lab/assets/robots` 中。

## 安装

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 和 Git LFS，然后在项目根目录执行一键安装脚本：

```bash
./z_lab.sh -i
```

该命令依次完成以下工作：

1. 初始化 Git LFS 并执行 `git lfs pull`，拉取 USD、模型和其他大文件。
2. 使用 Python 3.12 创建项目的 `.venv` 虚拟环境。
3. 执行 `uv sync --extra rsl-rl`，安装项目和 Isaac Lab 依赖。
4. 将 `src/rsl_rl` 中的 RSL-RL 5.5.1 + AMP 以 editable 方式安装，并检查 AMP 算法入口。

已有项目环境时，可以单独安装本地 RSL-RL：

```bash
./z_lab.sh --install-rsl-rl
```

`pyproject.toml` 使用官方发布的 `isaaclab==3.0.0` 安装包，包含所需的 Isaac Lab 子模块。安装只需本项目，无需额外克隆 IsaacLab 仓库，也不依赖项目外的本地源码目录；首次安装需要联网下载 Python 依赖和 Git LFS 资产。依赖版本由 `uv.lock` 锁定。安装完成后不需要手动激活虚拟环境，脚本会通过 `uv run` 使用项目环境。

项目会转发 Isaac Lab 提供的可选依赖，包括 Isaac Sim、物理后端、渲染、可视化、强化学习和开发工具等。根据使用场景选择相应的 extra，例如：

```bash
# 使用 OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <任务名称> physics=ovphysx

# 使用 Isaac Sim PhysX 和 RTX 渲染
uv run --extra isaacsim isaaclab random_agent --task <任务名称> physics=isaacsim_physx

# 使用 Isaac Sim 和 Rerun 可视化
uv run --extra isaacsim --extra rerun isaaclab random_agent --task <任务名称> --viz rerun
```

## 运行任务

当前三个任务的 agent 配置均使用 RSL-RL。可以先查看已经注册的环境：

```bash
uv run isaaclab list_envs --show_presets
```

将 `<任务名称>` 替换为上面列出的任务 ID：

也可以使用项目入口脚本。训练默认使用 `rsl_rl`，脚本会自动传递 `--rl_library rsl_rl`；如需使用其他强化学习库，可以在命令中显式传入 `--rl_library`。训练和播放参数遵循 Isaac Lab 3.0 接口，不再使用旧版的 `--headless` 参数：

```bash
# 列出已注册任务
./z_lab.sh -l

# 使用 rsl_rl 训练
./z_lab.sh -t --task ZLab-Velocity-Unitree-G1

# 播放策略
./z_lab.sh -p --task ZLab-Velocity-Unitree-G1 --checkpoint <检查点路径>
```

`-l`、`-t` 和 `-p` 后面的参数会传递给 Isaac Lab 3.0 对应入口。使用 `./z_lab.sh --help` 查看脚本帮助。

```bash
# 使用零动作运行
uv run isaaclab zero_agent --task <任务名称> --num_envs 16

# 使用随机动作运行
uv run isaaclab random_agent --task <任务名称> --num_envs 16

# 训练
uv run isaaclab train --rl_library rsl_rl --task <任务名称>

# 加载最新检查点进行测试
uv run isaaclab play --rl_library rsl_rl --task <任务名称> --checkpoint latest

# 多 GPU 训练
uv run isaaclab train_multigpu --rl_library rsl_rl --task <任务名称> --num_gpus 2

# 运行基准测试
uv run isaaclab benchmark runtime --task <任务名称> --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library rsl_rl --task <任务名称> --max_iterations 10
```

可以使用 `physics=<预设名称>` 选择 `isaaclab list_envs --show_presets` 输出的物理后端预设。

### G1_23 AMP 训练

训练前，将 CSV 转换为 50 Hz、xyzw 四元数的 NPZ 数据，输出到任务的 `motions` 目录：

```bash
uv run python scripts/mimic/config_csv_to_npz.py \
  --config src/z_lab/assets/data/motions/config/g1_23_amp.yaml \
  --output_dir src/z_lab/tasks/amp/config/g1_23/motions \
  --output_fps 50 --physics newton_mjwarp --visual none
```

任务使用的恢复片段名称为 `fallAndGetUp1_subject1_frames409_1948_g1_23.npz`。批量转换脚本默认沿用 CSV 文件名；若生成的恢复片段不带 `_g1_23` 后缀，需要在训练前将其重命名为上述名称。恢复片段只用于恢复环境初始化，不参与判别器的专家采样。

```bash
./z_lab.sh -t --task ZLab-AMP-Unitree-G1-23DOF physics=newton_mjwarp
```

速度命令的采样上下界在 120,000 个全局控制步内，由 `ranges` 线性扩展到 `limit_ranges`，随后保持不变；每次重置更新范围，命令在当前范围内均匀重采样。当前每轮 rollout 为 24 步，约 5,000 轮达到上限。恢复环境的 pelvis 辅助力从 200 N 向世界坐标 +Z 施加，在同样的控制步数内线性衰减到 0。

检查点保存策略、价值网络、判别器、归一化统计和 optimizer；当前不恢复环境的全局控制步计数，因此恢复训练会重新开始速度课程和辅助力衰减。

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

新增机器人时，在 `assets/robots` 增加对应的资产配置，并在 `tasks/locomotion/config` 下增加独立的机器人配置包。机器人自己的关节名称、动作顺序和奖励权重应在该机器人的配置范围内设置；公共 locomotion 配置只保存通用环境逻辑。

## 项目资产

项目自有的 USD、动作数据和其他资源统一放在 `src/z_lab/assets/data` 下。资产模块提供了一个在 editable checkout 和安装后的 wheel 中都有效的路径：

```python
from z_lab.assets import Z_LAB_ASSETS_DIR

robot_usd_path = Z_LAB_ASSETS_DIR / "models" / "g1" / "g1_29dof_rev_1_0.usd"
```

需要传入字符串的配置字段应使用 `str(robot_usd_path)`。项目的 `.gitattributes` 已为 USD 等大文件配置 Git LFS；添加大型二进制资产前请先安装 Git LFS。

### USD 和 Git LFS 说明

G1 29DOF 和 G1 23DOF 的顶层 USD 是 USD 组合层，文件本身很小，通过组合关系引用 `configuration` 目录中的 base、physics 和 sensor 层。只要 Git LFS 资源已完整拉取，两个模型可以正常打开，顶层文件较小不代表模型不完整。

DR02 的主体、关节、物理和碰撞层可以正常加载。原始 `DR02-pro_base.usd` 中 6 个传感器连杆的视觉引用指向不存在的几何模型，现已清除这些失效视觉引用；`d435_front_link`、`d435_head_link`、`d435_rear_link`、`imu_base_link`、`imu_head_link` 和 `lidar_link` 连杆及其传感器/碰撞结构仍然保留。若远程仓库中看到 USD 文件只有 `version`、`oid` 和 `size` 三行，这是 Git LFS 指针文件，应在项目根目录执行：

```bash
git lfs install
git lfs pull
```

之后再运行 `./z_lab.sh -i` 同步 Python 环境。

## 开发与检查

运行注册测试和代码质量检查：

```bash
uv run pytest
uv run pre-commit run --all-files
```

VS Code/Pylance 或 Cursor/basedpyright 应使用项目的 `.venv` 解释器。需要根据已安装的依赖生成编辑器配置时运行：

```bash
uv run isaaclab --editor
```

如果使用 Isaac Sim extra，配置编辑器时也需要带上对应 extra：

```bash
uv run --extra isaacsim isaaclab --editor
```

如果 Isaac Sim 是独立安装而不在项目环境中，可以显式指定路径：

```bash
uv run isaaclab --editor --isaac_path <isaac-sim路径>
```

如果编辑器无法解析模块，请确认编辑器选择的 Python 解释器与执行 setup 命令时使用的解释器一致，然后重新加载窗口。
