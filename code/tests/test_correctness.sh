#!/usr/bin/env bash
# test_correctness.sh：回归校验与输出比较入口。
# 对应 DESIGN.md 第 9 节。
# 参数和运行配置交由共用脚本处理。
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
exec "${PYTHON:-python3}" tests/correctness.py "$@"
