#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
{
    date -u --iso-8601=seconds
    uname -a
    lscpu
    free -h
    for tool in numactl nvidia-smi g++ mpicxx nvcc; do
        if command -v "$tool" >/dev/null 2>&1; then
            case "$tool" in
                numactl) numactl -H ;;
                nvidia-smi) nvidia-smi; nvidia-smi topo -m ;;
                *) "$tool" --version ;;
            esac
        else
            echo "$tool: unavailable"
        fi
    done
    python3 -c 'import sys,pandas,matplotlib; print(sys.version); print("pandas",pandas.__version__,"matplotlib",matplotlib.__version__)'
} >> results/env.txt 2>&1
