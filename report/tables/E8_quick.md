# E8 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pthreads | local | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 25.592 | 25.561 | 25.624 | 0.62738 | 0.62738 | 1 |
| pthreads | local | uniform | 262144 | 1 | 2 | 0 | 1 | 1 | static | 13.647 | 11.265 | 16.029 | 1.1765 | 0.58827 | 1.0015 |
| pthreads | local | uniform | 262144 | 1 | 4 | 0 | 1 | 1 | static | 12.824 | 11.483 | 14.164 | 1.2521 | 0.31302 | 1.0063 |
| pthreads | local | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 11.62 | 10.811 | 12.429 | 1.3818 | 0.17272 | 1.0069 |
| pthreads | locked | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 27.608 | 26.987 | 28.229 | 0.58158 | 0.58158 | 1 |
| pthreads | locked | uniform | 262144 | 1 | 2 | 0 | 1 | 1 | static | 24.86 | 24.71 | 25.01 | 0.64586 | 0.32293 | 1.0015 |
| pthreads | locked | uniform | 262144 | 1 | 4 | 0 | 1 | 1 | static | 16.308 | 15.44 | 17.177 | 0.98456 | 0.24614 | 1.0063 |
| pthreads | locked | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 15.242 | 15.242 | 15.243 | 1.0534 | 0.13167 | 1.0069 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 16.056 | 15.99 | 16.123 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
