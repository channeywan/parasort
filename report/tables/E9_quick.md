# E9 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid | psrs | uniform | 262144 | 2 | 4 | 0 | 1 | 1 | static | 8.1401 | 6.7492 | 9.5309 | 1.9803 | 0.24754 | 1.0014 |
| hybrid | psrs | uniform | 262144 | 4 | 2 | 0 | 1 | 1 | static | 6.3409 | 6.2753 | 6.4065 | 2.5422 | 0.31777 | 1.0104 |
| mpi | allgather | uniform | 262144 | 2 | 1 | 0 | 1 | 1 | static | 12.13 | 11.71 | 12.549 | 1.329 | 0.66448 | 1.0014 |
| mpi | allgather | uniform | 262144 | 4 | 1 | 0 | 1 | 1 | static | 6.5968 | 6.1693 | 7.0243 | 2.4436 | 0.61089 | 1.0104 |
| mpi | allgather | uniform | 262144 | 8 | 1 | 0 | 1 | 1 | static | 4.7213 | 4.7127 | 4.7299 | 3.4142 | 0.42678 | 1.0435 |
| mpi | psrs | uniform | 262144 | 2 | 1 | 0 | 1 | 1 | static | 12.367 | 12.292 | 12.442 | 1.3034 | 0.65171 | 1.0014 |
| mpi | psrs | uniform | 262144 | 2 | 1 | 0 | 1 | 1 | static | 12.239 | 12.056 | 12.422 | 1.3171 | 0.65854 | 1.0014 |
| mpi | psrs | uniform | 262144 | 4 | 1 | 0 | 1 | 1 | static | 9.3739 | 9.2221 | 9.5257 | 1.7196 | 0.42991 | 1.0104 |
| mpi | psrs | uniform | 262144 | 8 | 1 | 0 | 1 | 1 | static | 4.6614 | 4.6497 | 4.673 | 3.4582 | 0.43227 | 1.0435 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 16.12 | 16.098 | 16.141 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
