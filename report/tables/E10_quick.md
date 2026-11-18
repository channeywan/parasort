# E10 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mpi | psrs | uniform | 65536 | 2 | 1 | 0 | 1 | 1 | static | 26.272 | 3.0911 | 49.453 | 0.15814 | 0.079069 | 1.0038 |
| mpi | psrs | uniform | 65536 | 4 | 1 | 0 | 1 | 1 | static | 65.589 | 56.35 | 74.829 | 0.063342 | 0.015836 | 1.0165 |
| pipeline | block | uniform | 65536 | 2 | 1 | 0 | 1 | 1 | static | 8.8083 | 5.7262 | 11.89 | 0.47167 | 0.23583 | 1 |
| pipeline | block | uniform | 65536 | 2 | 1 | 0 | 1 | 1 | static | 5.3747 | 5.0347 | 5.7146 | 0.77299 | 0.3865 | 1 |
| pipeline | block | uniform | 65536 | 2 | 1 | 0 | 1 | 1 | static | 4.751 | 4.6455 | 4.8566 | 0.87446 | 0.43723 | 1 |
| pipeline | block | uniform | 65536 | 4 | 1 | 0 | 1 | 1 | static | 15.799 | 4.9865 | 26.612 | 0.26296 | 0.06574 | 1 |
| pipeline | block | uniform | 65536 | 4 | 1 | 0 | 1 | 1 | static | 24.062 | 8.113 | 40.01 | 0.17266 | 0.043166 | 1 |
| pipeline | block | uniform | 65536 | 4 | 1 | 0 | 1 | 1 | static | 18.407 | 8.8218 | 27.992 | 0.22571 | 0.056427 | 1 |
| pipeline | elem | uniform | 2 | 2 | 1 | 0 | 1 | 1 | static | 0.012202 | 0.009768 | 0.014637 | — | — | 1 |
| pipeline | elem | uniform | 4 | 4 | 1 | 0 | 1 | 1 | static | 14.998 | 9.9797 | 20.016 | — | — | 1 |
| pipeline | elem | uniform | 8 | 8 | 1 | 0 | 1 | 1 | static | 30.695 | 13.375 | 48.015 | — | — | 1 |
| seq | std | uniform | 65536 | 1 | 1 | 0 | 1 | 1 | static | 4.1546 | 4.1122 | 4.1969 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
