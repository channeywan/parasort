# ParaSort 实现与验收记录

项目目录：`/home/wcl/parasort`。依赖环境：`/home/wcl/parasort/.conda`，由 conda 创建，未使用 apt 或 pip 安装。`environment.yml` 提供可移植依赖，`conda-linux-64.lock.txt` 记录本机精确版本。

## 已实现范围

| 里程碑 | 结果与验收证据 |
| --- | --- |
| M0 | 公共 CLI、确定性生成、指纹、有序性、小数组逐元素验证、计时、CSV、std/bucket 基线；M0.csv verified=1；test_core 校验按块生成一致性和校验失败检测 |
| M1 | Pthreads locked/local 与 OpenMP local；CPU 扩展矩阵通过，正态输入等宽桶失衡度明显超过 1.5 |
| M2 | OpenMP PSRS，oversample、复合键、过分解、三种调度、串行初始化对照；64 线程/2²⁴ 元素/oversample=8 的 uniform 失衡度 1.04013；同规模 hotkeys 无/有 tie-break 为 4.10273/1.38218，见 M2_acceptance.csv |
| M3 | GNU parallel sort、OpenMP task 快排，通过边界与全部分布测试；TBB 可选项未引入 |
| M4 | MPI PSRS，有效样本 Gather(v)+Bcast、Allgather(v)、本地生成及 Scatterv、派生数据类型、Alltoallv、跨空桶边界与指纹验证；p=1/2/3/7/8 扩展测试通过 |
| M5 | MPI_THREAD_FUNNELED、rank 内 PSRS、本地值域再划分并行归并，通过扩展测试 |
| M6 | elem n=p 与 block；非阻塞双收发槽，支持非整除和空容量阶段；批大小 1/13/10000 专项测试通过 |
| M7 | 单卡 Thrust、sample、naive/hierarchical atomics、共享 bitonic/大桶回退、双卡排序/P2P 检测/host 中转；编译、运行与校验均通过；实测交换路径 pinned_host |
| M8 | E1–E13 脚本及 run_all --quick 完整通过，约 182 秒；154 组统计配置；E2–E13 共 12 组 PNG/PDF 图，含最小/最大误差线、Amdahl 拟合和 Karp–Flatt |
| M9 | code/README.md、简体中文 report/report.md、模块图/算法示意、报告章节、实测快速结果及正式实验 TODO 完整 |
| M10 | 未执行正式完整规模实验；需用户独占机器运行、补个人信息与截图、按教师模板排版后提交 |

## 回归记录

- 扩展 CPU：2048 组，n=0/1/2/7/1000/1000003，八种分布；大规模部分取 p=1/7，其余 p=1/2/3/7/8/16；另测三种调度和过分解。
- 扩展 MPI、Hybrid、Pipeline：656 组，覆盖 p=1/2/3/7/8、八种分布与多个边界规模，包含 Allgather、Scatter、重复键和不同批大小。
- E1 快速综合：728 组，CPU/MPI/GPU 全部通过，无跳过。每组含一次热身及一次正式运行，均开启校验。
- GPU 大规模专项：64 组，n=1000003 和 16777216，八种分布，单卡 Thrust、样本排序的两种原子路径及双卡；均 verified=1。
- 最终 /home/wcl/parasort 目录：构建检查、10 个主程序 --help、13 组 CPU/MPI/GPU 冒烟全部通过；见 final_smoke.csv。
- 无 nvcc 与无 mpicxx 的构建分支：打印 SKIP 并成功返回。
- 脚本通过 Python 语法和 Bash 语法检查；所有可执行程序支持 --help。

这些集合存在重叠，不能相加声称为互不重复的独立用例。原始日志在 code/results/validation/，E1_quick.csv、GPU_validation.csv 和 M2_acceptance.csv 保留对应原始记录。性能快速实验可能与其他回归并行运行，不能作为正式性能结论。

## 对设计细节的必要调整

1. 空块与短块只贡献有效样本，用 Gatherv/Allgatherv 避免最大键填充值的偏差。保留设计的等距采样索引和分位数选择规则，未暗改默认 oversample=1。
2. 经典 2n/p 均衡界带有等长块、样本充分和取整条件；均匀分布的 <1.1 也不是默认参数的无条件保证。本机以 oversample=8 的专项配置验证此目标。n<p 等情况优先确保严格正确性。
3. GPU 总桶数可到 2¹⁸，完整共享直方图不可行。改为每块 1024 槽共享哈希表聚合 512 输入，再更新全局。散射同样按活跃桶成批预留位置。
4. 48 KiB 不足以容纳把 12288 元素补齐至 16384 的 bitonic 缓冲。使用最多 8192 元素（32 KiB），大桶回退 Thrust。
5. MPI 接收缓冲大小在交换计数后确定，动态分配与 first-touch 包含在 exchange 时间，extra 标记 receive_allocation_timed=1。
6. --quick 除缩小 16 倍外，将通用规模限制为 2¹⁸、并行单元限制为 8、正式重复为 2；保留完整模式的设计参数，确保调试流程能在 15 分钟内完成。
7. 双卡混合使用设备事件计时本地 GPU 操作，主机墙钟计时选择分割点与跨卡控制，extra 标记 mixed_phase_clocks=1；总时间始终以主机墙钟记录。
8. quick/full、主机、种子及调度等调优参数分别统计，避免错误混算；无匹配串行基线的加速比保持空值。

## 正式实验前的检查

停止其他回归负载，激活项目 conda 环境；备份已有 results，执行 scripts/run_all.sh。核实 MPI 使用的绑定策略和 NUMA 放置。完整 E1 矩阵可能因 MPI 启动次数较多而耗时明显长于快速测试。最终报告数据仅使用正式结果，不将快速 n=2¹⁸ 外推为十亿规模表现。
