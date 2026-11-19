#!/usr/bin/env bash
# run_all.sh：实验执行与环境配置入口。
# 对应 DESIGN.md 第 6 节。
# 参数和运行配置交由共用脚本处理。
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python3}" -B scripts/run_experiments.py "$@"
