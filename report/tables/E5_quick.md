# E5 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mpi | psrs | uniform | 262144 | 2 | 1 | 0 | 1 | 1 | static | 12.385 | 12.29 | 12.481 | 1.2962 | 0.64808 | 1.0014 |
| mpi | psrs | uniform | 262144 | 4 | 1 | 0 | 1 | 1 | static | 9.0048 | 8.9133 | 9.0963 | 1.7827 | 0.44568 | 1.0104 |
| mpi | psrs | uniform | 262144 | 8 | 1 | 0 | 1 | 1 | static | 4.8388 | 4.8367 | 4.841 | 3.3176 | 0.4147 | 1.0435 |
| omp | psrs | uniform | 262144 | 1 | 2 | 0 | 1 | 1 | static | 14.26 | 13.008 | 15.512 | 1.1258 | 0.56289 | 1.0014 |
| omp | psrs | uniform | 262144 | 1 | 4 | 0 | 1 | 1 | static | 9.7519 | 8.9913 | 10.513 | 1.6462 | 0.41154 | 1.0104 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.6299 | 4.6089 | 4.651 | 3.4673 | 0.43341 | 1.0435 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 16.053 | 15.951 | 16.155 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
