# E10 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mpi | psrs | uniform | 65536 | 2 | 1 | 0 | 1 | 1 | static | 3.8274 | 3.5721 | 4.0827 | 1.1459 | 0.57296 | 1.0038 |
| mpi | psrs | uniform | 65536 | 4 | 1 | 0 | 1 | 1 | static | 2.2518 | 2.0332 | 2.4704 | 1.9477 | 0.48694 | 1.0165 |
| pipeline | block | uniform | 65536 | 2 | 1 | 0 | 1 | 1 | static | 4.6613 | 4.591 | 4.7315 | 0.94092 | 0.47046 | 1 |
| pipeline | block | uniform | 65536 | 2 | 1 | 0 | 1 | 1 | static | 4.4743 | 4.2158 | 4.7328 | 0.98024 | 0.49012 | 1 |
| pipeline | block | uniform | 65536 | 2 | 1 | 0 | 1 | 1 | static | 4.0027 | 3.9662 | 4.0392 | 1.0957 | 0.54787 | 1 |
| pipeline | block | uniform | 65536 | 4 | 1 | 0 | 1 | 1 | static | 4.4857 | 3.9606 | 5.0107 | 0.97776 | 0.24444 | 1 |
| pipeline | block | uniform | 65536 | 4 | 1 | 0 | 1 | 1 | static | 4.4477 | 4.4384 | 4.4571 | 0.98609 | 0.24652 | 1 |
| pipeline | block | uniform | 65536 | 4 | 1 | 0 | 1 | 1 | static | 4.2286 | 4.1575 | 4.2997 | 1.0372 | 0.2593 | 1 |
| pipeline | elem | uniform | 2 | 2 | 1 | 0 | 1 | 1 | static | 0.0088765 | 0.007835 | 0.009918 | — | — | 1 |
| pipeline | elem | uniform | 4 | 4 | 1 | 0 | 1 | 1 | static | 0.022647 | 0.018845 | 0.02645 | — | — | 1 |
| pipeline | elem | uniform | 8 | 8 | 1 | 0 | 1 | 1 | static | 0.052292 | 0.049873 | 0.054712 | — | — | 1 |
| seq | std | uniform | 65536 | 1 | 1 | 0 | 1 | 1 | static | 4.3859 | 3.6391 | 5.1327 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
