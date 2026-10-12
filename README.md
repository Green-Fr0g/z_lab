# z_lab

这是一个基于 Isaac Lab 3.0 的下游任务包，使用标准的 uv `src` 项目布局。

当前注册了三个基于管理器的速度跟踪 locomotion 任务：

- `ZLab-Velocity-Unitree-G1`：Unitree G1，29 个策略自由度。
- `ZLab-Velocity-Unitree-G1-23DOF`：Unitree G1，23 个策略自由度。
- `ZLab-Velocity-DR02`：DR02，29 个策略自由度。

机器人资产使用 Isaac Lab 的隐式关节执行器。每个机器人的关节顺序、动作缩放、连杆名称、执行器参数和 USD 路径保存在 `src/z_lab/assets/robots` 中。

## 安装

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后在项目根目录创建环境：

```bash
uv sync
```

`pyproject.toml` 中的 `[tool.uv.sources]` 使用了相对于本项目的 Isaac Lab 源码路径。如果 Isaac Lab 目录位置发生变化，需要同步修改这些路径。

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

## 开发与检查

运行注册测试和代码质量检查：

```bash
uv run pytest
uv run pre-commit run --all-files
```

项目中的 `pyrightconfig.json` 已包含项目 `src` 目录和 Isaac Lab 源码目录，VS Code/Pylance 或 Cursor/basedpyright 可以直接跳转到上游实现。需要配置编辑器环境时运行：

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
