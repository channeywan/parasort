# 实验方法事实

- 仅采用所选模式 CSV 中 verified=1 的行；快速模式和完整模式独立导出。
- 排序总时间和各阶段分别取同一完整配置的中位数，并保留最小值、最大值及重复数；阶段中位数之和不替代总时间。
- STREAM 使用 CSV 记录的最佳值和中位数；不由排序时间推算 STREAM 带宽。
- 归并有效带宽为 8n 字节 / merge_ms；本地排序和划分不是单遍读写，不估算有效带宽。
- CPU 总时间以稳态墙钟覆盖排序调用，不包括输入生成及验证；每组预热不写入 CSV。
- MPI 排序 total_ms 在输入生成/可选 Scatter 之后开始，scatter_ms 单独记录；阶段为各 rank 最大值，阶段比例可合计超过 100%。
- 流水线总时间包含流水线及结果汇集；GPU 总时间采用主机墙钟，设备阶段采用 CUDA 事件，双卡控制阶段包含主机时钟。
- 加速比只用同主机、seed、n、分布、模式的唯一串行 std::sort 基线；缺少匹配时留空。
- E9 的 sort_plus_scatter_ms 是两个实测中位数之和，仍不包括 root 输入生成。
- T5 运行环境的 NUMA 页放置回退已在 ../code/results/validation/T5_residency.log 中诊断：默认 first-touch 请求可能受运行环境限制而回退。该验证日志是方法约束证据，不作为完整 E14 带宽数据来源；正式数值仅来自所选模式的 E14 CSV。

## E1

- 没有所选模式的 verified=1 数据。

## E2

- 没有所选模式的 verified=1 数据。

## E3

- 没有所选模式的 verified=1 数据。

## E4

- 没有所选模式的 verified=1 数据。

## E5

- 没有所选模式的 verified=1 数据。

## E6

- 没有所选模式的 verified=1 数据。

## E7

- 没有所选模式的 verified=1 数据。

## E8

- 没有所选模式的 verified=1 数据。

## E9

- 没有所选模式的 verified=1 数据。

## E10

- 没有所选模式的 verified=1 数据。

## E11

- 没有所选模式的 verified=1 数据。

## E12

- 没有所选模式的 verified=1 数据。

## E13

- 没有所选模式的 verified=1 数据。

## E14

- 没有所选模式的 verified=1 数据。

## E4b

- 没有所选模式的 verified=1 数据。

## 跳过配置与读取问题

```json
[]
```

