# E10 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mpi | psrs | uniform | 65536 | 2 | 1 | 0 | 1 | 1 | static | 3.5342 | 2.8131 | 4.2553 | 1.1603 | 0.58013 | 1.0038 |
| mpi | psrs | uniform | 65536 | 4 | 1 | 0 | 1 | 1 | static | 2.1355 | 2.1321 | 2.1388 | 1.9202 | 0.48006 | 1.0165 |
| pipeline | block | uniform | 65536 | 2 | 1 | 0 | 1 | 1 | static | 4.23 | 4.1375 | 4.3224 | 0.96942 | 0.48471 | 1 |
| pipeline | block | uniform | 65536 | 2 | 1 | 0 | 1 | 1 | static | 4.4862 | 4.2644 | 4.7081 | 0.91404 | 0.45702 | 1 |
| pipeline | block | uniform | 65536 | 2 | 1 | 0 | 1 | 1 | static | 3.838 | 3.7665 | 3.9096 | 1.0684 | 0.53421 | 1 |
| pipeline | block | uniform | 65536 | 4 | 1 | 0 | 1 | 1 | static | 3.6829 | 3.6171 | 3.7488 | 1.1134 | 0.27835 | 1 |
| pipeline | block | uniform | 65536 | 4 | 1 | 0 | 1 | 1 | static | 4.2414 | 4.2392 | 4.2437 | 0.9668 | 0.2417 | 1 |
| pipeline | block | uniform | 65536 | 4 | 1 | 0 | 1 | 1 | static | 5.0341 | 4.1449 | 5.9232 | 0.81457 | 0.20364 | 1 |
| pipeline | elem | uniform | 2 | 2 | 1 | 0 | 1 | 1 | static | 0.010575 | 0.007674 | 0.013475 | — | — | 1 |
| pipeline | elem | uniform | 4 | 4 | 1 | 0 | 1 | 1 | static | 0.026655 | 0.025037 | 0.028273 | — | — | 1 |
| pipeline | elem | uniform | 8 | 8 | 1 | 0 | 1 | 1 | static | 0.057363 | 0.045295 | 0.06943 | — | — | 1 |
| seq | std | uniform | 65536 | 1 | 1 | 0 | 1 | 1 | static | 4.1006 | 3.7624 | 4.4388 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
