# ParaSort 并行排序系统

C++17 实现等宽桶排序、PSRS、Pthreads/OpenMP、MPI/混合排序、MPI 双缓冲流水线及单/双 CUDA GPU 排序。所有程序共用确定性输入、校验和、CSV 字段和校验计时规则。代码位于 `code/`，报告位于 `report/`。

## 环境与构建

依赖安装**只使用 conda**。当前机器已为本项目创建 `/home/wcl/parasort/.conda`，使用已有的系统 g++ 与 CUDA 12.6，不修改系统软件。新机器可在项目根目录执行：

```bash
conda env create --prefix ./.conda -f environment.yml
conda activate ./.conda
source code/scripts/activate.sh
cd code
make -j8
```

`activate.sh` 配置本项目 conda 环境的 PATH、MPI 动态库、MPICH_CXX、OMP 绑定和 matplotlib 缓存目录；每个实验脚本也会加载它。若使用外部 conda 环境，先激活并自行保证 MPI 的动态库可访问。无系统编译器时可用 `conda install -c conda-forge gxx_linux-64 make`，然后用 `make CXX="$CXX"`；CUDA 宿主编译器默认 `/usr/bin/g++`，需要时修改 Makefile 的 `-ccbin`。

目标：`all seq pthreads openmp mpi cuda test clean`。缺少 mpicxx 或 nvcc 时对应目标打印 SKIP，其余照常构建。CUDA 默认 `sm_89`，通过 `make CUDA_ARCH=sm_80` 等覆盖；Toolkit 由服务器提供，若缺失仍通过 conda 安装适配版本。构建时 `CUDA: enabled` 表示发现编译器，并不表示驱动可用。`make clean` 仅删除 bin/build，不删除实验结果。

本机已实测两块 RTX 4090 D，驱动 560.35.05，CUDA Toolkit 12.6。两个 CUDA 程序已通过 nvcc 编译，并在可访问 GPU 的执行环境中通过回归；受限沙箱内可能无法访问设备。程序报告具体 CUDA 错误，实验脚本在设备不可用时记录 SKIP，不生成虚假 GPU 数据。

## 快速开始

```bash
./bin/seq_sort --n 1e6 --dist uniform --repeat 3 --csv results/serial.csv
./bin/bucket_pthreads --n 2^24 --threads 8 --variant local
./bin/bucket_pthreads --n 2^20 --threads 8 --variant locked
./bin/bucket_omp --n 2^24 --threads 8 --dist normal
./bin/psrs_omp --n 2^24 --threads 8 --dist hotkeys
./bin/psrs_omp --n 2^24 --threads 8 --dist hotkeys --no-tiebreak
./bin/psrs_omp --n 2^24 --threads 8 --oversample 8 --buckets-per-thread 4 --schedule dynamic
./bin/ref_sorts --n 2^24 --threads 8 --variant gnu_parallel
./bin/ref_sorts --n 2^24 --threads 8 --variant omp_task_qsort
mpiexec -n 3 ./bin/psrs_mpi --n 1000003 --repeat 2
mpiexec -n 3 ./bin/psrs_mpi --n 1000003 --variant allgather --scatter
mpiexec -n 2 ./bin/psrs_hybrid --n 2^24 --threads 4
mpiexec -n 4 ./bin/pipeline_sort --n 4 --variant elem
mpiexec -n 3 ./bin/pipeline_sort --n 1000003 --variant block --batch 4096
CUDA_VISIBLE_DEVICES=0 ./bin/gpu_sort --n 2^24 --variant sample
CUDA_VISIBLE_DEVICES=0 ./bin/gpu_sort --n 2^24 --variant sample --naive-atomics
CUDA_VISIBLE_DEVICES=0,1 ./bin/gpu_multi --n 2^24
```

所有程序支持 `--help`。统一参数：`--n` 支持整数、`2^30`、`1e9`；`--dist` 支持 uniform/normal/power/hotkeys/sorted/reverse/nearly_sorted/few_unique；`--seed` 默认为 42；`--repeat` 默认为 5，每次另有一次不输出 CSV 的热身；`--verify` 默认开启，小于等于 100000 个元素时还与 std::sort 逐元素比较；关闭时 verified 为 -1，分析脚本不采用该行。

默认 PSRS 样本数为每块 p 个，支持 `--oversample`。均匀数据的失衡度不是必然小于 1.1：它随 n、p、样本数变化；需要更紧平衡时提高到 8 或 16，并以真实 CSV 为准。无 tie-break 的实验仍保证排序正确，用于观察重复键造成的失衡。

## MPI 与 NUMA

项目 conda 环境使用 **MPICH**；以下绑定参数专用于 **OpenMPI**，使用 OpenMPI 时另建 conda 环境，勿混用两种实现的编译器、启动器或动态库。

```bash
# OpenMPI：64 物理核上的标准运行
mpirun -np 64 --bind-to core --map-by core ./bin/psrs_mpi --n 2^30
# OpenMPI：128 逻辑 CPU，需要显式启用超线程或 --oversubscribe
mpirun -np 128 --use-hwthread-cpus --bind-to hwthread ./bin/psrs_mpi --n 2^30
# OpenMPI：每 socket 1 rank，每 rank 32 线程
mpirun -np 2 --map-by ppr:1:package:PE=32 --bind-to core \
  -x OMP_NUM_THREADS=32 -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
  ./bin/psrs_hybrid --n 2^30 --threads 32
```

自动实验可通过 `MPIEXEC=mpirun MPIEXEC_FLAGS='--use-hwthread-cpus --oversubscribe'` 传入启动参数。MPICH 支持的亲和参数请查看该版本的 `mpiexec -help`，脚本默认不附加厂商专用绑定选项。

本机 `numactl -H` 报告 8 个 NUMA 节点。混合实验应同时比较 2 rank（每 socket）和 8 rank（每 NUMA 节点）。默认 `OMP_PROC_BIND=close OMP_PLACES=cores`；`--variant serial_init` 对 PSRS/桶排序提供串行初始化对照。MPI 动态接收缓冲大小需交换计数后才确定，其分配与 first-touch 计入 exchange 时间并写入 extra。

每个 MPI 本地数量、count、displacement 均检查 `< INT_MAX`；默认本地生成可以处理全局 n > INT_MAX。可选 `--scatter` 是教学用 Scatterv 路径，因为旧式 Scatterv 全局 displacement 的限制，要求全局 n < INT_MAX，并在 extra 中单独记录 scatter_ms，不混入排序总时间。

## 验证与实验

```bash
make test                         # 快速回归，含可用的 MPI/GPU
bash tests/test_correctness.sh     # 完整参数矩阵，耗时较长
bash tests/test_correctness.sh --quick --cpu-only
bash scripts/run_all.sh --quick   # M8 冒烟实验及全部图表
bash scripts/run_all.sh           # 设计中的完整 E1–E13 实验
bash scripts/bench_e4.sh           # 单独执行分布与负载均衡实验
python3 analysis/summarize.py
python3 analysis/plot.py
```

快速模式为兼顾 8 核机器的调试耗时，将规模除以 16 后再上限限制为 2^18，正式重复 2 次，常规实验处理单元最多 8（E4b 保留 64/128 线程扫描）；E3 每单元 2^14，E11 采用 2^14 和 2^16。它只验证实验链路，不能替代设计中 2^30 的性能结论。完整模式使用设计规模，正式重复 5 次、串行基线 3 次。实验多次执行会追加 CSV，请在一轮干净正式实验开始前将旧 results 目录重命名备份；保留 env.txt 和原始 CSV。

E1 快速矩阵包含全部八种分布的 CPU 边界测试，并缩减 MPI 启动次数；完整矩阵覆盖 n={0,1,2,7,1000,1000003}、p={1,2,3,7,8,16}，GPU 在驱动可用时加入。elem 按算法定义只测试 n=p。核心测试另检验按块生成一致性、checksum 故障检测、CLI 和重复键切分边界。失败返回非零；所有实验脚本遇到实际运行错误立即停止，缺失可选硬件则写 status.json。

CSV 保持 DESIGN.md 的 22 列，单次重复一行。extra 保存 seed、schedule、oversample、tiebreak、桶数、批大小、GPU 路径等信息。`commands.jsonl` 保存实际执行命令。统计按主机、规模、分布、种子、并行配置和调优参数分组，快速/正式数据分开，所有加速比使用相匹配的 seq/std 基线。

图表写入 `../report/figures/`，表格写入 `../report/tables/`。每图同时保存 200 DPI PNG/PDF；无 GPU 时 E11 生成标有“measurements unavailable”的说明图，不伪造曲线。阶段时间在 MPI 中独立取各 rank 最大值，因此阶段最大值之和未必等于总时间最大值。GPU compute_ms 排除 H2D/D2H，total_ms 包括主机控制、内存申请和同步；两者并不简单相等。

## 实现中的边界处理

- 采样使用实际有效样本数：空块贡献 0 个，短块不重复抽样。MPI 使用 Gather(v)/Allgather(v) 传输有效样本，避免以 UINT32_MAX 伪造真实最大键样本。
- 复合键为 `(key, block/rank, sorted_position)`。统一采用包含分割点的 upper-bound 语义，不同桶的三元组边界互不重叠；输出仍只存 uint32 键。
- PSRS 的经典 2n/p 结论有等长块及足够规模等前提，不能将小 n、n<p、取整和过分解条件下的归一化失衡都断言为 ≤2；此时有序性、元素个数和 checksum 仍逐一校验。
- GPU 的 B 随 n 增长，不能把 B 项直方图整体放进 shared memory。因此用每 CUDA block 的 1024 槽共享哈希表聚合最多 512 个输入的活跃桶，再原子更新全局；分发按同样方式成批预留偏移，避免 O(grid×B) 内存。
- shared-memory bitonic 只处理不超过 8192 个元素的桶（32 KiB），更大桶回退 Thrust。12288 元素补齐到 16384 需要 64 KiB，不能套用“48 KiB 即可容纳”的判断。
- 双卡检测双向 P2P，支持时使用 peer async copy，否则使用 pinned host 中转，路径写入 CSV。实际交换路径与 GPU 校验结果见 CSV；正式 2^30 规模性能仍需完整实验。

## 报告与提交

`../report/report.md` 为简体中文报告初稿，引用真实快速测试图表。完整实验、GPU 性能、教师模板和个人信息保留明确 TODO。最终人工截取：构建成功、E1 PASS、128 线程 htop、双 GPU nvidia-smi、lscpu/numactl/topo；按模板排成 docx/pdf，填写学号姓名后将 `code/` 和 `report/` 打包为“学号-姓名.zip”。不应把 conda 环境、bin/build 编译产物或缓存放入课程提交包。

## MPI 绑定修复（T1）

所有 MPI 实验与回归通过 scripts/launch.py 集中配置启动器及环境。纯 MPI/流水线覆盖 OMP_PROC_BIND=false，MPICH 按 core 或 hwthread 映射；混合按 socket、NUMA 或成对 NUMA 映射，超出物理核数时使用 OMP_PLACES=threads。显式 MPIEXEC_FLAGS 优先并打印警告。

运行 `bash scripts/check_affinity.sh` 检查 rank CPU 集合、混合线程分布与 p=8/p=2 的 2²² 元素性能健全性。绑定前快速数据保留于 results/archive_pre_bindfix，不用于正式统计。extra 的 launcher、mpi_bind 区分实际启动策略。

## GPU NUMA 绑定（T2）

GPU 程序默认 `--gpu-affinity numa`，通过设备 PCI 地址读取 sysfs NUMA 节点，将驱动线程绑定到该节点 CPU；`--gpu-affinity none` 用于对照。每卡 pinned 输入、输出及中转由对应驱动线程在绑定后分配并首次写入，排序和采样默认参数保持不变。标准错误日志记录实际 CPU 集合；CSV 记录请求策略、实际绑定或失败原因。E11 包含双卡 numa/none 对照。

## 正式实验

正式数据应在服务器空闲、GPU 无其他计算任务时采集。开始前先完成进程绑定与性能健全性检查；快速模式只验证实验链路，不能代替十亿元素结果。

```bash
tmux new -s parasort
cd <项目目录>
conda activate ./.conda
source code/scripts/activate.sh
cd code
make -j16 all
bash scripts/check_affinity.sh --sanity
bash scripts/run_all.sh --dry-run
mkdir -p results/logs
bash scripts/run_all.sh 2>&1 | tee results/logs/run_all.log
```

`--only E2,E4` 只执行指定实验，`--force` 重跑成功项目，`--quick` 与正式结果分开维护续跑状态。每个实验写入临时 CSV，全部成功后原子替换对应结果；失败和 Ctrl-C 会保留失败状态与中间文件，再次运行从未成功的实验继续，避免将半轮数据混入统计。成功项目的状态在 `results/*_status.json`，日志在 `results/logs/<实验>.log`，命令及退出状态记录在 `results/commands.jsonl`。

`--dry-run` 列出实际参数和粗略耗时，仅用于安排机时，不生成 CSV、状态或环境记录；估时不能用于报告。起跑时记录机器信息、Git 提交和时间，检查 1 分钟负载及 GPU 计算进程。负载大于 2 时需要在终端确认，非交互环境应先检查负载来源，再明确使用 `--ignore-load`。机器信息和开始/结束时间保存在 `results/env.txt`。

中断后的典型续跑命令为 `bash scripts/run_all.sh`；希望整轮重采则使用 `bash scripts/run_all.sh --force`。运行前无需删除已归档的绑定修复前数据。


## 过采样与重复键扫描（T4）

`bash scripts/bench_e4b.sh` 执行 E4b；也可用 `bash scripts/run_all.sh --only E4b --dry-run` 检查完整命令。正式规模为 2²⁸，快速规模为 2¹⁸，快速模式保留相同的线程数和过采样系数，每配置分别测量 5 次和 2 次，均另有一次热身。

过采样扫描固定 OpenMP 64 线程、开启 tie-break，在 uniform、normal、hotkeys 分布上取 k={1,2,4,8,16,32}。重复键扫描固定 k=1，在 hotkeys、few_unique 分布上取 p={8,16,32,64,128}，分别开启或关闭 tie-break；128 线程使用 `OMP_PLACES=threads`。两个扫描中完全相同的 hotkeys、p=64、k=1、开启 tie-break 配置共用同一组实测数据，因此每轮共执行 37 个不同配置。

`E4b_oversample` 图按分布展示失衡度和总时间的双纵轴曲线，并单独显示分割点选择时间；`E4b_tiebreak` 图按分布展示失衡度随 p 的变化。正式图表不带后缀，快速图表和表格带 `_quick` 后缀，两种规模分别统计。失衡是否随 k 单调变化、p=64 时开启 tie-break 是否低于 2，均以实际 CSV 为准；脚本不会修改排序参数来匹配预期。

## 内存带宽（T5 / E14）

`bash scripts/bench_e14.sh` 测量 Copy 和 Triad：三个 double 数组并行首次写入，正式每数组 2²⁸ 个元素，快速每数组 2²⁰ 个元素。两种绑定 close/spread、1–128 线程均测量；每核热身一次、正式五次，保存最好和中位带宽，每次在计时后逐元素验证。快速数组可能命中缓存，其带宽不能作为 DRAM 性能结论。

CSV 使用十进制 GB/s，Copy 每元素读取和写入共 16 字节，Triad 两读一写共 24 字节。PSRS 只对归并阶段计算 `8*n/(merge_ms*1e6)` GB/s；不估算本地排序或划分的带宽。E14 正式图仅叠加完整 E2/E5 的归并数据，快速图只展示快速 STREAM。E14 在 E13 分析之前执行，以便同轮生成带宽图和数据包。

本轮单配置健全性检查使用 2²⁸、128 线程、close，Triad 最好带宽约 40 GB/s，原始结果保存在 `results/validation/T5_stream128.csv`。已核对 `-O3 -march=native`、全部 128 个线程的不同 CPU 绑定和并行 first-touch。`T5_residency.log` 显示 6 GiB 页面没有均匀驻留在八个节点：节点 6 没有页面、节点 7 约 6.5 MiB。当前部分节点的大量文件缓存、`zone_reclaim_mode=0` 和跨节点分配影响了本地性；[内核说明](https://www.kernel.org/doc/html/v6.14/admin-guide/sysctl/vm.html#zone-reclaim-mode) 允许这种回退。本次值是当前页面分配条件下的实测有效带宽，不能当作硬件峰值。没有修改系统设置或改变默认内存策略；该检查也不替代 E14 完整矩阵。

## 扩展性对照覆盖（T6）

E2 的 128 线程 OpenMP 使用 `OMP_PLACES=threads`，CSV 的 `places` 记录实际设置；p≥16 同时比较 2 rank 与 8 rank 混合。E3 增加等宽桶和 8 rank 混合；总处理单元不足 8 时，固定 8 rank 的配置明确跳过。E9 比较 2/4/8 rank 混合，并在 p=8、64 各包含本地生成与 Scatter 的配对配置；快速模式也保留这些通信配置。

Scatter 与输入生成发生在排序总计时之前，`scatter_ms` 单独记录。因此两种路径的 `total_ms` 比较仅代表排序，不能据此比较输入准备的完整成本。阶段表和数据包同时保留 Scatter 时间；`scatter_ms + total_ms` 也不包含 root 生成输入的时间。Amdahl 拟合区分混合 rank 数，并保留 64→128 的规定绑定切换对应的数据点。
