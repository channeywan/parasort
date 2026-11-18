# E11 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpu | sample | uniform | 16384 | 1 | 1 | 1 | 1 | 1 | static | 0.5789 | 0.57403 | 0.58378 | 1.5691 | 1.5691 | 1.1304 |
| gpu | sample | uniform | 16384 | 1 | 1 | 1 | 1 | 1 | static | 1.5341 | 1.3244 | 1.7439 | 0.5921 | 0.5921 | 1.1304 |
| gpu | sample | uniform | 65536 | 1 | 1 | 1 | 1 | 1 | static | 0.62918 | 0.61119 | 0.64717 | 6.5759 | 6.5759 | 1.2788 |
| gpu | sample | uniform | 65536 | 1 | 1 | 1 | 1 | 1 | static | 1.124 | 0.91588 | 1.3322 | 3.6809 | 3.6809 | 1.2788 |
| gpu | thrust | uniform | 16384 | 1 | 1 | 1 | 1 | 1 | static | 0.32177 | 0.32161 | 0.32193 | 2.823 | 2.823 | 1 |
| gpu | thrust | uniform | 65536 | 1 | 1 | 1 | 1 | 1 | static | 0.34804 | 0.34493 | 0.35114 | 11.888 | 11.888 | 1 |
| gpu_multi | sample | uniform | 16384 | 1 | 2 | 2 | 1 | 1 | static | 9.8988 | 0.67969 | 19.118 | 0.091764 | 0.045882 | 1.0106 |
| gpu_multi | sample | uniform | 65536 | 1 | 2 | 2 | 1 | 1 | static | 5.9021 | 1.1743 | 10.63 | 0.70101 | 0.3505 | 1.0038 |
| seq | std | uniform | 16384 | 1 | 1 | 0 | 1 | 1 | static | 0.90836 | 0.90817 | 0.90854 | 1 | 1 | 1 |
| seq | std | uniform | 65536 | 1 | 1 | 0 | 1 | 1 | static | 4.1374 | 4.1137 | 4.1612 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
