# 实现事实（供算法章节引用）

- 空块和短块只贡献有效样本，MPI 使用 Gatherv/Allgatherv；未用最大键填充伪样本。
- PSRS 默认 oversample=1；平衡界依赖等长块、样本和取整条件，参数扫描记录实际 oversample。
- GPU 使用每块 1024 槽共享哈希表聚合 512 个输入，再更新全局直方图；散射按活跃桶批量预留位置。
- bitonic 桶排序容量为 8192 个 uint32 元素（32 KiB），更大桶回退 Thrust。
- MPI 交换计数后分配接收缓冲并 first-touch，接收分配包含在 exchange 时间，extra 记录 receive_allocation_timed=1。
- quick 缩小规模并减少重复；任何快速时间均不推算十亿元素的完整时间。
- 双卡 GPU 的设备阶段使用 CUDA 事件，跨卡控制使用主机墙钟；总时间为主机墙钟。
- 主机、种子、模式、规模、分布、调优参数分别统计；无匹配基线时加速比为空。
- 纯 MPI/流水线 rank 按 core 放置，超过物理核数时按 hwthread；OMP_NUM_THREADS=1、OMP_PROC_BIND=false。
- 混合 2 rank 使用 socket；8 rank 使用 NUMA；4 rank 每 rank 使用两个 NUMA 节点。OpenMP close，超出 rank 物理核容量时使用 threads places。
- GPU 默认请求 NUMA 绑定：读取 CUDA PCI 地址对应的 sysfs NUMA/CPU 列表；绑定、pinned 内存分配与 first-touch 在驱动该 GPU 的同一线程上完成，失败状态写入 extra。
- 新增 E4b 的 oversample 扫描及 tie-break/p 扫描；E14 测 Copy/Triad；E2/E3/E9 补混合 NUMA 拓扑和 Scatter 对照。
- T5 的 NUMA 页放置诊断记录在 ../code/results/validation/T5_residency.log；默认 first-touch 可能受运行环境限制而回退，完整数据包不从此验证日志提取带宽数值。
- 公共默认值：seed=42，threads=1，repeat=5，oversample=1，buckets_per_thread=1，schedule=static，tiebreak=true，gpu_affinity=numa。
- GPU 桶数为不小于 ceil(n/4096) 的最小二次幂；GPU block 大小与核函数具体实现一致，单卡路径为 256 线程/块。
- 流水线 block 默认批大小为 max(1, 最小阶段容量/8)，elem 版批大小为 1；显式 --batch 记录在 CSV。
- MPI 计数接口使用 int，超出可表示范围会报错；双卡无可用 peer 访问时经 pinned host 中转，实际路径由 extra 记录。

来源：IMPLEMENTATION.md「对设计细节的必要调整」、公共 Config、GPU 与 MPI 源码、T1–T6 实现；本文件记录实现，不提供性能判断。
