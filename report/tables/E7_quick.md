# E7 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 7.3924 | 7.2504 | 7.5344 | 2.1595 | 0.26994 | 1.0435 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.4815 | 4.4458 | 4.5173 | 3.5622 | 0.44527 | 1.0435 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.7001 | 4.6029 | 4.7974 | 3.3965 | 0.42456 | 1.0435 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.8799 | 4.7485 | 5.0112 | 3.2714 | 0.40892 | 1.0435 |
| omp | serial_init | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.5149 | 4.4925 | 4.5374 | 3.5358 | 0.44197 | 1.0435 |
| omp | serial_init | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.8091 | 4.7005 | 4.9177 | 3.3195 | 0.41494 | 1.0435 |
| omp | serial_init | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.7749 | 4.6745 | 4.8754 | 3.3433 | 0.41791 | 1.0435 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 15.964 | 15.938 | 15.99 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
