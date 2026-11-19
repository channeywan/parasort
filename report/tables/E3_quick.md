# E3 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mpi | psrs | uniform | 16384 | 1 | 1 | 0 | 1 | 1 | static | 1.193 | 1.1921 | 1.1939 | 0.95715 | 0.95715 | 1 |
| mpi | psrs | uniform | 32768 | 2 | 1 | 0 | 1 | 1 | static | 1.5288 | 1.3659 | 1.6917 | 1.5579 | 0.77896 | 1.0042 |
| mpi | psrs | uniform | 65536 | 4 | 1 | 0 | 1 | 1 | static | 2.0667 | 2.0312 | 2.1022 | 2.1435 | 0.53588 | 1.0165 |
| mpi | psrs | uniform | 131072 | 8 | 1 | 0 | 1 | 1 | static | 2.1923 | 2.1891 | 2.1955 | 4.1624 | 0.5203 | 1.0261 |
| omp | psrs | uniform | 16384 | 1 | 1 | 0 | 1 | 1 | static | 1.6488 | 1.593 | 1.7047 | 0.69255 | 0.69255 | 1 |
| omp | psrs | uniform | 32768 | 1 | 2 | 0 | 1 | 1 | static | 2.2905 | 1.953 | 2.6279 | 1.0399 | 0.51993 | 1.0042 |
| omp | psrs | uniform | 65536 | 1 | 4 | 0 | 1 | 1 | static | 2.3939 | 2.3213 | 2.4665 | 1.8505 | 0.46264 | 1.0165 |
| omp | psrs | uniform | 131072 | 1 | 8 | 0 | 1 | 1 | static | 2.2438 | 2.2018 | 2.2858 | 4.0669 | 0.50836 | 1.0261 |
| seq | std | uniform | 16384 | 1 | 1 | 0 | 1 | 1 | static | 1.1419 | 1.1396 | 1.1442 | 1 | 1 | 1 |
| seq | std | uniform | 32768 | 1 | 1 | 0 | 1 | 1 | static | 2.3817 | 2.3598 | 2.4036 | 1 | 1 | 1 |
| seq | std | uniform | 65536 | 1 | 1 | 0 | 1 | 1 | static | 4.4301 | 3.6578 | 5.2023 | 1 | 1 | 1 |
| seq | std | uniform | 131072 | 1 | 1 | 0 | 1 | 1 | static | 9.1253 | 8.0927 | 10.158 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
