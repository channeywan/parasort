# E9 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid | psrs | uniform | 262144 | 2 | 4 | 0 | 1 | 1 | static | 21.399 | 20.172 | 22.626 | 0.75486 | 0.094357 | 1.0014 |
| hybrid | psrs | uniform | 262144 | 4 | 2 | 0 | 1 | 1 | static | 78.506 | 68.82 | 88.192 | 0.20576 | 0.02572 | 1.0104 |
| mpi | allgather | uniform | 262144 | 2 | 1 | 0 | 1 | 1 | static | 13.388 | 12.992 | 13.784 | 1.2066 | 0.60328 | 1.0014 |
| mpi | allgather | uniform | 262144 | 4 | 1 | 0 | 1 | 1 | static | 37.175 | 24.303 | 50.047 | 0.43452 | 0.10863 | 1.0104 |
| mpi | allgather | uniform | 262144 | 8 | 1 | 0 | 1 | 1 | static | 50.371 | 49.282 | 51.46 | 0.32068 | 0.040085 | 1.0435 |
| mpi | psrs | uniform | 262144 | 2 | 1 | 0 | 1 | 1 | static | 13.332 | 13.223 | 13.441 | 1.2116 | 0.6058 | 1.0014 |
| mpi | psrs | uniform | 262144 | 2 | 1 | 0 | 1 | 1 | static | 42.73 | 25.918 | 59.542 | 0.37803 | 0.18901 | 1.0014 |
| mpi | psrs | uniform | 262144 | 4 | 1 | 0 | 1 | 1 | static | 46.538 | 43.126 | 49.95 | 0.3471 | 0.086774 | 1.0104 |
| mpi | psrs | uniform | 262144 | 8 | 1 | 0 | 1 | 1 | static | 96.82 | 96.435 | 97.204 | 0.16684 | 0.020855 | 1.0435 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 16.153 | 16.138 | 16.169 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
