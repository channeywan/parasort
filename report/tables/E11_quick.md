# E11 快速验证

| impl | variant | dist | n | procs | threads | gpus | tiebreak | buckets_per_thread | schedule | median_ms | min_ms | max_ms | speedup | efficiency | imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpu | sample | uniform | 16384 | 1 | 1 | 1 | 1 | 1 | static | 0.84267 | 0.83137 | 0.85397 | 1.4193 | 1.4193 | 1.1304 |
| gpu | sample | uniform | 16384 | 1 | 1 | 1 | 1 | 1 | static | 0.69606 | 0.56202 | 0.8301 | 1.7183 | 1.7183 | 1.1304 |
| gpu | sample | uniform | 65536 | 1 | 1 | 1 | 1 | 1 | static | 0.79274 | 0.58859 | 0.9969 | 5.8453 | 5.8453 | 1.2788 |
| gpu | sample | uniform | 65536 | 1 | 1 | 1 | 1 | 1 | static | 0.87745 | 0.84103 | 0.91386 | 5.2811 | 5.2811 | 1.2788 |
| gpu | thrust | uniform | 16384 | 1 | 1 | 1 | 1 | 1 | static | 0.53632 | 0.53616 | 0.53649 | 2.23 | 2.23 | 1 |
| gpu | thrust | uniform | 65536 | 1 | 1 | 1 | 1 | 1 | static | 0.45439 | 0.3448 | 0.56398 | 10.198 | 10.198 | 1 |
| gpu_multi | sample | uniform | 16384 | 1 | 2 | 2 | 1 | 1 | static | 2.2483 | 2.2399 | 2.2567 | 0.53197 | 0.26599 | 1.0106 |
| gpu_multi | sample | uniform | 65536 | 1 | 2 | 2 | 1 | 1 | static | 2.036 | 1.7857 | 2.2862 | 2.276 | 1.138 | 1.0038 |
| gpu_multi | sample | uniform | 262144 | 1 | 2 | 2 | 1 | 1 | static | 0.89363 | 0.84583 | 0.94143 | 17.978 | 8.9889 | 1.0014 |
| gpu_multi | sample | uniform | 262144 | 1 | 2 | 2 | 1 | 1 | static | 2.2544 | 2.0527 | 2.456 | 7.1263 | 3.5632 | 1.0014 |
| seq | std | uniform | 16384 | 1 | 1 | 0 | 1 | 1 | static | 1.196 | 1.1626 | 1.2295 | 1 | 1 | 1 |
| seq | std | uniform | 65536 | 1 | 1 | 0 | 1 | 1 | static | 4.6338 | 3.7846 | 5.4831 | 1 | 1 | 1 |
| seq | std | uniform | 262144 | 1 | 1 | 0 | 1 | 1 | static | 16.066 | 16.044 | 16.087 | 1 | 1 | 1 |

加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。
