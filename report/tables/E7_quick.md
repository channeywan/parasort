# E7 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.4805 | 4.3659 | 7.4753 | 3.593 | 0.44912 | 1.0435 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.609 | 4.5997 | 4.6183 | 3.4928 | 0.4366 | 1.0435 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.4977 | 4.384 | 4.6661 | 3.5793 | 0.44741 | 1.0435 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.5096 | 4.4145 | 4.5299 | 3.5697 | 0.44622 | 1.0435 |
| omp | serial_init | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.5449 | 4.493 | 4.5727 | 3.5421 | 0.44276 | 1.0435 |
| omp | serial_init | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.5187 | 4.4662 | 4.6982 | 3.5626 | 0.44532 | 1.0435 |
| omp | serial_init | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.4304 | 4.3686 | 4.4822 | 3.6336 | 0.4542 | 1.0435 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 16.098 | 16.066 | 16.239 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
