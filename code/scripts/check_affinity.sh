#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
exec "${PYTHON:-python3}" scripts/affinity_check.py "$@"
