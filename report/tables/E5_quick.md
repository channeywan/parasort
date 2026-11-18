# E5 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mpi | psrs | uniform | 262144 | 2 | 1 | 0 | 1 | 1 | static | 13.466 | 13.224 | 13.708 | 1.1987 | 0.59937 | 1.0014 |
| mpi | psrs | uniform | 262144 | 4 | 1 | 0 | 1 | 1 | static | 43.771 | 41.387 | 46.156 | 0.36879 | 0.092198 | 1.0104 |
| mpi | psrs | uniform | 262144 | 8 | 1 | 0 | 1 | 1 | static | 71.575 | 67.52 | 75.63 | 0.22553 | 0.028192 | 1.0435 |
| omp | psrs | uniform | 262144 | 1 | 2 | 0 | 1 | 1 | static | 14.086 | 12.786 | 15.386 | 1.146 | 0.573 | 1.0014 |
| omp | psrs | uniform | 262144 | 1 | 4 | 0 | 1 | 1 | static | 9.7135 | 9.158 | 10.269 | 1.6619 | 0.41547 | 1.0104 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.5442 | 4.5288 | 4.5596 | 3.5524 | 0.44405 | 1.0435 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 16.143 | 16.117 | 16.168 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
