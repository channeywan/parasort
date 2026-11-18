# E12 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpu | sample | uniform | 262144 | 1 | 1 | 1 | 1 | 1 | static | 0.88505 | 0.88347 | 0.88663 | 20.514 | 20.514 | 1.2681 |
| gpu | thrust | uniform | 262144 | 1 | 1 | 1 | 1 | 1 | static | 0.72052 | 0.7189 | 0.72214 | 25.199 | 25.199 | 1 |
| gpu_multi | sample | uniform | 262144 | 1 | 2 | 2 | 1 | 1 | static | 3.985 | 3.6345 | 4.3355 | 4.5561 | 2.2781 | 1.0014 |
| hybrid | psrs | uniform | 262144 | 2 | 4 | 0 | 1 | 1 | static | 14.978 | 8.8814 | 21.074 | 1.2122 | 0.15152 | 1.0014 |
| mpi | psrs | uniform | 262144 | 8 | 1 | 0 | 1 | 1 | static | 85.347 | 85.344 | 85.351 | 0.21273 | 0.026591 | 1.0435 |
| omp | local | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 5.0322 | 4.9539 | 5.1105 | 3.608 | 0.451 | 1.0069 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 8.0722 | 6.7352 | 9.4093 | 2.2492 | 0.28115 | 1.0435 |
| ref | gnu_parallel | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 5.7527 | 5.6651 | 5.8402 | 3.1561 | 0.39451 | 1 |
| ref | omp_task_qsort | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 10.127 | 8.2276 | 12.027 | 1.7928 | 0.2241 | 1 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 18.156 | 18.146 | 18.166 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
