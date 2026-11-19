#!/usr/bin/env bash
# env_info.sh：实验执行与环境配置入口。
# 对应 DESIGN.md 第 6 节。
# 参数和运行配置交由共用脚本处理。
# Structured, append-only snapshots; this script is never invoked by --dry-run.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
{
    echo "=== ENVIRONMENT ${1:-snapshot} ==="
    echo "timestamp_utc=$(date -u --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "ignore_load=${PARASORT_IGNORE_LOAD:-false}"
    echo "loadavg=$(cat /proc/loadavg)"
    if [[ ${1:-snapshot} != finish ]]; then
        echo '[os]'
        cat /etc/os-release
        uname -a
        echo '[lscpu]'
        lscpu
        echo '[memory]'
        free -h
        echo '[numactl -H]'
        if command -v numactl >/dev/null 2>&1; then numactl -H; else echo unavailable; fi
        echo '[nvidia-smi]'
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi
            echo '[nvidia-smi topo -m]'
            nvidia-smi topo -m
            echo '[gpu pci numa]'
            for device in /sys/bus/pci/devices/*; do
                if [[ -r "$device/vendor" && $(cat "$device/vendor") == 0x10de && -r "$device/numa_node" ]]; then
                    echo "$(basename "$device") numa_node=$(cat "$device/numa_node")"
                fi
            done
        else echo unavailable; fi
        for tool in g++ mpicxx mpiexec nvcc; do
            echo "[$tool --version]"
            if command -v "$tool" >/dev/null 2>&1; then "$tool" --version; else echo unavailable; fi
        done
        echo '[python]'
        "${PYTHON:-python3}" -B - <<'PY'
import importlib.metadata
import sys
print(sys.version)
for package in ('pandas', 'matplotlib'):
    try:
        print(package, importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        print(package, 'unavailable')
PY
    fi
    echo "=== END ENVIRONMENT ${1:-snapshot} ==="
} 2>&1 | sed 's/[[:blank:]]*$//' >> results/env.txt
