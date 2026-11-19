#!/usr/bin/env bash
# bench_e4.sh：实验执行与环境配置入口。
# 对应 DESIGN.md 第 6 节。
# 参数和运行配置交由共用脚本处理。
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_all.sh --only E4 "$@"
