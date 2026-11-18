#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
if [[ ${1:-} != "" && ${1:-} != --quick ]]; then echo "Usage: $0 [--quick]" >&2; exit 2; fi
make -j"${BUILD_JOBS:-8}" all
for experiment in {1..13}; do
    "${PYTHON:-python3}" scripts/bench.py "E${experiment}" "$@"
done
