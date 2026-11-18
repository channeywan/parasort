# E8 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pthreads | local | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 26.048 | 25.928 | 26.168 | 0.62477 | 0.62477 | 1 |
| pthreads | local | uniform | 262144 | 1 | 2 | 0 | 1 | 1 | static | 13.104 | 10.986 | 15.222 | 1.2419 | 0.62095 | 1.0015 |
| pthreads | local | uniform | 262144 | 1 | 4 | 0 | 1 | 1 | static | 11.814 | 11.484 | 12.144 | 1.3775 | 0.34438 | 1.0063 |
| pthreads | local | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 11.623 | 10.823 | 12.423 | 1.4002 | 0.17502 | 1.0069 |
| pthreads | locked | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 27.813 | 27.217 | 28.41 | 0.58512 | 0.58512 | 1 |
| pthreads | locked | uniform | 262144 | 1 | 2 | 0 | 1 | 1 | static | 16.805 | 16.572 | 17.039 | 0.9684 | 0.4842 | 1.0015 |
| pthreads | locked | uniform | 262144 | 1 | 4 | 0 | 1 | 1 | static | 16.111 | 14.201 | 18.021 | 1.0101 | 0.25253 | 1.0063 |
| pthreads | locked | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 14.391 | 13.943 | 14.84 | 1.1308 | 0.14135 | 1.0069 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 16.274 | 16.27 | 16.278 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
