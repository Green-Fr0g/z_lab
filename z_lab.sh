#!/usr/bin/env bash

# z_lab 统一入口脚本。
# -i/--install: 初始化 Git LFS、拉取大文件、创建 uv 虚拟环境并安装依赖和本地 rsl_rl
# --install-rsl-rl: 在项目虚拟环境中安装本地可编辑版 rsl_rl
# -l/--list:    列出已注册任务
# -t/--train:   使用 rsl_rl 运行训练
# -p/--play:    运行策略回放

set -euo pipefail

Z_LAB_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${Z_LAB_PATH}"

die() {
    echo "[错误] $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "找不到命令 '$1'，请先安装后重试。"
}

install_rsl_rl() {
    require_command uv
    [[ -x "${Z_LAB_PATH}/.venv/bin/python" ]] || die "项目虚拟环境不存在，请先执行 ./z_lab.sh -i。"

    echo "[信息] 安装本地 rsl_rl（editable，5.5.1 + AMP）。"
    uv pip install --python "${Z_LAB_PATH}/.venv/bin/python" --no-deps --editable "${Z_LAB_PATH}/src/rsl_rl"
    "${Z_LAB_PATH}/.venv/bin/python" - <<'PY'
from importlib.metadata import version

import rsl_rl
from rsl_rl.algorithms import AMPPPO

print(f"[信息] rsl_rl {version('rsl-rl-lib')}: {rsl_rl.__file__}")
print(f"[信息] AMP 算法: {AMPPPO.__module__}.{AMPPPO.__name__}")
PY
}

install_project() {
    require_command git
    require_command uv
    require_command git-lfs

    echo "[信息] 初始化 Git LFS。"
    git lfs install
    echo "[信息] 拉取 Git LFS 资产。"
    git lfs pull

    if [[ ! -d .venv ]]; then
        echo "[信息] 创建 Python 虚拟环境。"
        uv venv .venv --python 3.12
    fi

    echo "[信息] 同步项目依赖。"
    uv sync --extra rsl-rl
    install_rsl_rl
    echo "[完成] z_lab 安装完成。"
}

run_train() {
    require_command uv
    local args=("$@")
    local has_rl_library=0
    for arg in "${args[@]}"; do
        if [[ "${arg}" == "--rl_library" || "${arg}" == --rl_library=* ]]; then
            has_rl_library=1
            break
        fi
    done
    if [[ ${has_rl_library} -eq 0 ]]; then
        args=(--rl_library rsl_rl "${args[@]}")
    fi
    uv run isaaclab train "${args[@]}"
}

run_play() {
    require_command uv
    local args=("$@")
    local has_rl_library=0
    local has_visualizer=0
    for arg in "${args[@]}"; do
        if [[ "${arg}" == "--rl_library" || "${arg}" == --rl_library=* ]]; then
            has_rl_library=1
        fi
        if [[ "${arg}" == "--visualizer" || "${arg}" == --visualizer=* || "${arg}" == "--viz" || "${arg}" == --viz=* ]]; then
            has_visualizer=1
        fi
    done
    if [[ ${has_rl_library} -eq 0 ]]; then
        args=(--rl_library rsl_rl "${args[@]}")
    fi
    if [[ ${has_visualizer} -eq 0 ]]; then
        args+=(--visualizer newton_gl)
    fi
    uv run isaaclab play "${args[@]}"
}

list_tasks() {
    require_command uv
    uv run isaaclab list_envs "$@"
}

case "${1:-}" in
    -i|--install)
        shift
        install_project "$@"
        ;;
    --install-rsl-rl)
        shift
        install_rsl_rl "$@"
        ;;
    -t|--train)
        shift
        run_train "$@"
        ;;
    -p|--play)
        shift
        run_play "$@"
        ;;
    -l|--list)
        shift
        list_tasks "$@"
        ;;
    -h|--help|"")
        cat <<'EOF'
用法：
  ./z_lab.sh -i                         初始化环境、拉取资产并安装本地 rsl_rl
  ./z_lab.sh --install-rsl-rl            单独安装本地可编辑版 rsl_rl
  ./z_lab.sh -l                         列出已注册训练任务
  ./z_lab.sh -t --task <任务名称>        使用 rsl_rl 训练
  ./z_lab.sh -p --task <任务名称>        播放已训练策略

示例：
  ./z_lab.sh -i
  ./z_lab.sh --install-rsl-rl
  ./z_lab.sh -t --task ZLab-Velocity-Unitree-G1
  ./z_lab.sh -p --task ZLab-Velocity-Unitree-G1 --checkpoint <检查点路径>

所有未识别参数都会继续传递给对应的 Isaac Lab 入口。
EOF
        ;;
    *)
        die "未知选项 '$1'。使用 './z_lab.sh --help' 查看用法。"
        ;;
esac
