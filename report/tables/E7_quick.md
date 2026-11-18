# E7 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.7978 | 4.3919 | 7.4753 | 3.3732 | 0.42166 | 1.0435 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.5899 | 4.5136 | 4.6661 | 3.5261 | 0.44076 | 1.0435 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.5168 | 4.5038 | 4.5299 | 3.5831 | 0.44789 | 1.0435 |
| omp | serial_init | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.5505 | 4.5283 | 4.5727 | 3.5566 | 0.44457 | 1.0435 |
| omp | serial_init | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.6346 | 4.5709 | 4.6982 | 3.4921 | 0.43651 | 1.0435 |
| omp | serial_init | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.4316 | 4.381 | 4.4822 | 3.652 | 0.4565 | 1.0435 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 16.184 | 16.129 | 16.239 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
