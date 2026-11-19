# E5 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mpi | psrs | uniform | 262144 | 2 | 1 | 0 | 1 | 1 | static | 12.062 | 12.009 | 12.116 | 1.3233 | 0.66163 | 1.0014 |
| mpi | psrs | uniform | 262144 | 4 | 1 | 0 | 1 | 1 | static | 9.4387 | 8.9549 | 9.9225 | 1.6911 | 0.42277 | 1.0104 |
| mpi | psrs | uniform | 262144 | 8 | 1 | 0 | 1 | 1 | static | 4.6675 | 4.6538 | 4.6812 | 3.4197 | 0.42747 | 1.0435 |
| omp | psrs | uniform | 262144 | 1 | 2 | 0 | 1 | 1 | static | 13.867 | 12.406 | 15.329 | 1.151 | 0.57551 | 1.0014 |
| omp | psrs | uniform | 262144 | 1 | 4 | 0 | 1 | 1 | static | 9.4721 | 8.9314 | 10.013 | 1.6851 | 0.42128 | 1.0104 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.658 | 4.3975 | 4.9185 | 3.4267 | 0.42834 | 1.0435 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 15.962 | 15.929 | 15.994 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
