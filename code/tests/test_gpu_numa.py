#!/usr/bin/env python3
# test_gpu_numa.py：验证算法输出和实验方法的边界条件。
# 对应 DESIGN.md 第 9 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""在实际 GPU 上核对驱动线程 CPU 与 sysfs NUMA 集合，随后检查关闭策略。"""

import csv
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def cpus(value):
    result = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        result.update(range(int(lo), int(hi or lo) + 1))
    return result


def main():
    with tempfile.TemporaryDirectory() as temporary:
        for exe, count in [("gpu_sort", 1), ("gpu_multi", 2)]:
            for mode in ["numa", "none"]:
                path = Path(temporary) / f"{exe}_{mode}.csv"
                result = subprocess.run(
                    [
                        str(ROOT / "bin" / exe),
                        "--n",
                        "1000003",
                        "--dist",
                        "hotkeys",
                        "--variant",
                        "sample",
                        "--gpu-affinity",
                        mode,
                        "--repeat",
                        "1",
                        "--csv",
                        str(path),
                    ],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                row = list(csv.DictReader(path.open()))[0]
                assert row["verified"] == "1"
                extra = dict(
                    part.split("=", 1)
                    for part in row["extra"].split(";")
                    if "=" in part
                )
                assert extra["gpu_affinity"] == mode, extra
                if mode == "numa":
                    for gpu in range(count):
                        node = int(extra[f"gpu{gpu}_numa"])
                        expected = cpus(
                            Path(f"/sys/devices/system/node/node{node}/cpulist")
                            .read_text()
                            .strip()
                        )
                        actual = cpus(extra[f"gpu{gpu}_cpus"])
                        assert actual and actual <= expected, (gpu, expected, actual)
                        assert extra[f"gpu{gpu}_affinity"].startswith("numa"), extra
                        print(
                            f"PASS {exe}: GPU{gpu}, NUMA{node}, CPUs={sorted(actual)}"
                        )
                else:
                    print(f"PASS {exe}: --gpu-affinity none; verified=1")
                print(result.stderr.strip())


if __name__ == "__main__":
    main()
