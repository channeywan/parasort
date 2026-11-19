#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python3}" -B scripts/run_experiments.py "$@"
