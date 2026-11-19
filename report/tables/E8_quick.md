# E8 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pthreads | local | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 26.015 | 25.694 | 26.168 | 0.62276 | 0.62276 | 1 |
| pthreads | local | uniform | 262144 | 1 | 2 | 0 | 1 | 1 | static | 11.462 | 10.986 | 15.222 | 1.4135 | 0.70676 | 1.0015 |
| pthreads | local | uniform | 262144 | 1 | 4 | 0 | 1 | 1 | static | 11.452 | 10.763 | 12.144 | 1.4148 | 0.35369 | 1.0063 |
| pthreads | local | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 11.378 | 10.678 | 12.423 | 1.424 | 0.178 | 1.0069 |
| pthreads | locked | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 27.551 | 27.043 | 28.41 | 0.58805 | 0.58805 | 1 |
| pthreads | locked | uniform | 262144 | 1 | 2 | 0 | 1 | 1 | static | 16.715 | 16.346 | 17.039 | 0.96928 | 0.48464 | 1.0015 |
| pthreads | locked | uniform | 262144 | 1 | 4 | 0 | 1 | 1 | static | 16.401 | 14.201 | 18.138 | 0.98783 | 0.24696 | 1.0063 |
| pthreads | locked | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 13.911 | 13.723 | 14.84 | 1.1646 | 0.14558 | 1.0069 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 16.201 | 16.079 | 16.278 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
