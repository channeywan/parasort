# E12 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpu | sample | uniform | 262144 | 1 | 1 | 1 | 1 | 1 | static | 0.81814 | 0.80625 | 0.83002 | 19.655 | 19.655 | 1.2681 |
| gpu | thrust | uniform | 262144 | 1 | 1 | 1 | 1 | 1 | static | 0.54418 | 0.53854 | 0.54982 | 29.55 | 29.55 | 1 |
| gpu_multi | sample | uniform | 262144 | 1 | 2 | 2 | 1 | 1 | static | 5.8149 | 5.1752 | 6.4547 | 2.7654 | 1.3827 | 1.0014 |
| hybrid | psrs | uniform | 262144 | 2 | 4 | 0 | 1 | 1 | static | 10.045 | 9.3271 | 10.764 | 1.6008 | 0.2001 | 1.0014 |
| mpi | psrs | uniform | 262144 | 8 | 1 | 0 | 1 | 1 | static | 4.7641 | 4.6144 | 4.9139 | 3.3753 | 0.42192 | 1.0435 |
| omp | local | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 3.1197 | 2.9593 | 3.2801 | 5.1545 | 0.64431 | 1.0069 |
| omp | psrs | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 4.3931 | 4.3501 | 4.436 | 3.6604 | 0.45755 | 1.0435 |
| ref | gnu_parallel | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 3.5087 | 3.4812 | 3.5361 | 4.5831 | 0.57289 | 1 |
| ref | omp_task_qsort | uniform | 262144 | 1 | 8 | 0 | 1 | 1 | static | 7.6798 | 7.4242 | 7.9355 | 2.0939 | 0.26173 | 1 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 16.081 | 16.048 | 16.114 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
