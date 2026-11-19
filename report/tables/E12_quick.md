# E12 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpu | sample | uniform | 262144 | 1 | 1 | 1 | 1 | 1 | static | 1.0459 | 0.86244 | 1.2293 | 15.442 | 15.442 | 1.2681 |
| gpu | thrust | uniform | 262144 | 1 | 1 | 1 | 1 | 1 | static | 1.0296 | 1.0031 | 1.0561 | 15.686 | 15.686 | 1 |
| gpu_multi | sample | uniform | 262144 | 1 | 2 | 2 | 1 | 1 | static | 2.3815 | 2.3087 | 2.4543 | 6.7816 | 3.3908 | 1.0014 |
| hybrid | psrs | uniform | 262144 | 2 | 4 | 0 | 1 | 1 | static | 8.0985 | 6.8224 | 9.3746 | 1.9943 | 0.24928 | 1.0014 |
| mpi | psrs | uniform | 262144 | 8 | 1 | 0 | 1 | 1 | static | 4.8869 | 4.5908 | 5.1829 | 3.3049 | 0.41311 | 1.0435 |
| omp | local | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 3.3887 | 3.3669 | 3.4105 | 4.766 | 0.59575 | 1.0069 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.4422 | 4.3967 | 4.4877 | 3.6357 | 0.45446 | 1.0435 |
| ref | gnu_parallel | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 3.52 | 3.5026 | 3.5375 | 4.5882 | 0.57352 | 1 |
| ref | omp_task_qsort | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 8.167 | 8.1307 | 8.2033 | 1.9775 | 0.24719 | 1 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 16.151 | 16.107 | 16.194 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
