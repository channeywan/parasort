# E11 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpu | sample | uniform | 16384 | 1 | 1 | 1 | 1 | 1 | static | 0.54896 | 0.51855 | 0.58378 | 1.8334 | 1.8334 | 1.1304 |
| gpu | sample | uniform | 16384 | 1 | 1 | 1 | 1 | 1 | static | 0.92359 | 0.52246 | 1.7439 | 1.0897 | 1.0897 | 1.1304 |
| gpu | sample | uniform | 65536 | 1 | 1 | 1 | 1 | 1 | static | 0.58179 | 0.54465 | 0.64717 | 7.1115 | 7.1115 | 1.2788 |
| gpu | sample | uniform | 65536 | 1 | 1 | 1 | 1 | 1 | static | 0.74725 | 0.57226 | 1.3322 | 5.5368 | 5.5368 | 1.2788 |
| gpu | thrust | uniform | 16384 | 1 | 1 | 1 | 1 | 1 | static | 0.30361 | 0.27908 | 0.32193 | 3.315 | 3.315 | 1 |
| gpu | thrust | uniform | 65536 | 1 | 1 | 1 | 1 | 1 | static | 0.33103 | 0.31497 | 0.35114 | 12.499 | 12.499 | 1 |
| gpu_multi | sample | uniform | 16384 | 1 | 2 | 2 | 1 | 1 | static | 0.77848 | 0.67969 | 19.118 | 1.2929 | 0.64643 | 1.0106 |
| gpu_multi | sample | uniform | 65536 | 1 | 2 | 2 | 1 | 1 | static | 0.97497 | 0.76953 | 10.63 | 4.2436 | 2.1218 | 1.0038 |
| seq | std | uniform | 16384 | 1 | 1 | 0 | 1 | 1 | static | 1.0065 | 0.90817 | 1.124 | 1 | 1 | 1 |
| seq | std | uniform | 65536 | 1 | 1 | 0 | 1 | 1 | static | 4.1374 | 3.6852 | 5.3593 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
