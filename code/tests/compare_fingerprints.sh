#!/usr/bin/env bash
# DESIGN.md §4: compare actual output checksums across formatting revisions.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
exec "${PYTHON:-python3}" -B tests/fingerprints.py "$@"
