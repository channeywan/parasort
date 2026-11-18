# 並行程序設計課程設計 —— 設計文檔（供 coding 智能體直接實現）

> 題目：**面向大規模數據的多範式並行排序系統：從並行桶排序到負載均衡的樣本排序（PSRS）**
> 英文：*ParaSort: Load-Balanced Parallel Sample Sort on Multi-core, Message-Passing and GPU*
>
> 運行平台：2 × AMD EPYC 7542（共 64 物理核 / 128 邏輯 CPU，2 個 socket，NUMA）、256 GB 內存、Samsung 990 PRO 2TB（ext4）、2 × NVIDIA RTX 4090D（24 GB，Ada，sm_89）、Ubuntu 22.04.5 LTS。
>
> 課程要求（摘自 `0introduction2026.pdf` 第 58 頁）：獨立實現一並行程序並撰寫程序設計報告；介紹背景（總體介紹、模塊構成、主要算法或過程介紹——介紹思路，不貼大段代碼）、測試（截圖、性能對比數據等）；**重點介紹如何針對性用並行程序解決問題**；**代碼與報告分開放在不同文件夾**；壓縮後以「學號-姓名」命名提交。

---

## 0. 給 coding 智能體的總體指引

1. 按第 9 節的里程碑順序實現，每完成一個里程碑都要通過對應的驗收條件再進入下一個。
2. 代碼放 `code/`，報告放 `report/`，兩者不得混放（第 2 節給出完整目錄樹）。
3. 所有實現共用同一套「數據生成 → 計時 → 校驗 → CSV 輸出」框架（第 4 節），保證不同實現之間的數據完全相同、性能數據可以直接對比。
4. 報告用**簡體中文**撰寫（課程為華中科技大學課程，課件為簡體中文）；代碼註釋可用中文或英文。
5. 報告中**不要貼大段代碼**，算法用流程圖 / 示意圖 / 偽代碼片段（≤15 行）說明。
6. 所有性能數字都必須來自 `code/results/` 中真實運行得到的 CSV，由腳本生成圖表；不得編造數據。智能體若無法在服務器上運行，則把報告中的數據位置留成明確的 `TODO(待服務器實測)` 標記，並保證腳本一鍵可生成所需圖表。

---

## 1. 選題理由與課程內容的對應

### 1.1 為什麼選「並行排序」

- 排序是最基礎、應用最廣的計算之一（數據庫 ORDER BY / 歸併連接、MapReduce 的 Shuffle&Sort、搜索引擎倒排索引構建、TeraSort 基準）。數據規模達到 10⁹ 量級時，串行 `std::sort` 需要數十秒以上，存在明確的並行化需求。
- 它天然能串起課程幾乎所有內容，而且**正確性易驗證、規模易調節、加速比易測量**。
- 它有一個非常適合寫「針對性」分析的核心矛盾：**數據分佈 → 負載不均衡**。課件第 2 講的並行桶排序在均勻數據上表現好，但在偏斜數據上會嚴重失衡；本設計逐級給出針對性解決方案：

| 版本 | 劃分依據 | 在什麼數據上失效 | 針對性改進 |
|---|---|---|---|
| V1 等寬並行桶排序（課件第 2 講第 97–99 頁） | 按值域等寬切分 | 正態、冪律等偏斜分佈：某些桶極大 | → 用**正則採樣**選分割點（PSRS） |
| V2 PSRS | 按樣本分位數切分 | 大量重複鍵（熱點鍵）：相同鍵必然落入同一桶 | → **複合鍵 tie-break**（鍵, 塊號, 塊內位置） |
| V3 PSRS + tie-break | 按複合鍵分位數切分 | 理論上任意分佈每桶 ≤ 2n/p | — |

### 1.2 課件知識點 → 本項目中的落點（報告中要以表格形式呈現，體現「結合課程」）

| 課件 | 知識點 | 本項目中的體現 |
|---|---|---|
| 1 architecture | UMA / NUMA 存儲器存取模型、共享內存 vs 分佈式內存 | 雙路 EPYC 是 NUMA 機；first-touch 內存放置、線程綁定（`OMP_PROC_BIND`）、`numactl` 實驗 |
| 1 architecture | 加速比 S = T_serial / T_parallel（**必須與最優串行程序比較，不能與並行程序 p=1 比較**）、效率 E = S/n | 所有加速比以 `std::sort` 串行基線為分母 |
| 1 architecture | Amdahl 定律（含並行開銷 T_o）、Gustafson 定律 | 強擴展實驗擬合串行比例 f；弱擴展實驗對照 Gustafson |
| 2 algorithm | PCAM：分解、通信、組合、映射 | 報告第 2 章用 PCAM 四步推導 PSRS |
| 2 algorithm | 數據分解（輸入分解 / 輸出分解）、遞歸分解 | 塊劃分 = 輸入分解；桶 = 輸出（值域）分解；OpenMP task 快排 = 遞歸分解 |
| 2 algorithm | 映射：靜態映射、動態映射、負載均衡 | 等寬桶（靜態、失衡）vs 採樣分割（靜態但均衡）vs 過分解+dynamic 調度 |
| 2 algorithm | BSP 模型（超步 = 本地計算 + 通信 + 路障） | PSRS 的 4 個超步，用 BSP 代價模型估算時間 |
| 2 algorithm | 並行桶排序、「進一步優化」（每處理器 p 個小桶，桶 i 發給處理器 i）、全交換數據傳遞 | V1 實現；MPI 版本用 `MPI_Alltoallv` 實現「Sends data from each process to every other process」 |
| 2 algorithm | 第 101 頁「思考」：計算模型？劃分方法？映射方法？算法模型？挑戰？ | 報告總結章逐條回答 |
| 3 basics | SAS / MP / 混合編程模型、Pattern（Pipe-and-Filter、MapReduce） | Pthreads/OpenMP = SAS；MPI = MP；MPI+OpenMP = Hybrid；流水線排序 = Pipe-and-Filter |
| 4 pipeline | Type 2 流水線、流水線插入排序、「數列長度遠大於處理單元數量時如何設計流水線」 | 流水線排序模塊：逐元素版（課件原版）+ 分塊版（回答課件問題） |
| 5 shared memory | Pthreads 臨界區、鎖、路障、死鎖；false sharing | V1 帶鎖全局桶 vs 本地小桶無鎖版本；計數器 64B 對齊 |
| 5 shared memory | OpenMP：parallel for、schedule(static/dynamic/guided, chunk)、reduction、single/barrier、task | OpenMP PSRS；調度策略對比實驗；task 版並行快排 |
| 5 shared memory | TBB / Cilk 工作竊取、`parallel_sort` | `__gnu_parallel::sort` 與（可選）`tbb::parallel_sort` 作為參考上限 |
| 6 MPI | Scatter/Gather/Bcast/Allreduce/Alltoall(v)/Sendrecv、非阻塞 Isend/Irecv、Wtime | MPI PSRS 全流程；流水線版用非阻塞通信做雙緩衝 |
| 7 MapReduce | Partitioner、Shuffle & Sort、Spark `sortByKey` | 報告中說明 TeraSort / Spark RangePartitioner 本質就是樣本排序；（可選）PySpark 小實驗 |
| 8 GPU & CUDA | SIMT、Grid/Block/Thread、shared memory、`__syncthreads`、**層次化原子操作**、cudaEvent 計時、Stream 異步拷貝、Thrust（sort/scan/upper_bound/merge） | 單 GPU 樣本排序 + 雙 GPU 樣本排序（OpenMP 每線程驅動一塊 GPU） |

---

## 2. 目錄結構（提交時的最終形態）

```
學號-姓名/
├── code/                          # 代碼（與報告分開）
│   ├── README.md                  # 編譯、運行、復現實驗的說明
│   ├── Makefile                   # 頂層構建；CUDA 目標在找不到 nvcc 時自動跳過
│   ├── include/
│   │   ├── common.hpp             # Key 類型、Config、Timer、PhaseTimes、CSV 輸出聲明
│   │   ├── datagen.hpp            # 計數器式確定性數據生成
│   │   ├── verify.hpp             # 有序性 + 多重集校驗和
│   │   ├── psrs_core.hpp          # 與並行框架無關的 PSRS 公共函數（採樣、選分割點、劃分、多路歸併）
│   │   └── cli.hpp                # 命令行解析
│   ├── src/
│   │   ├── common/                # datagen.cpp verify.cpp cli.cpp csv.cpp psrs_core.cpp
│   │   ├── seq/seq_sort.cpp       # 串行基線：std::sort、串行桶排序
│   │   ├── pthreads/bucket_pthreads.cpp     # V1 等寬並行桶排序（帶鎖版 / 無鎖版）
│   │   ├── openmp/
│   │   │   ├── psrs_omp.cpp       # OpenMP PSRS（含 tie-break 開關、過分解 + 調度實驗）
│   │   │   ├── bucket_omp.cpp     # OpenMP 等寬桶排序（與 PSRS 公平對比用）
│   │   │   └── ref_sorts.cpp      # 參考：__gnu_parallel::sort、OpenMP task 快排、(可選) tbb::parallel_sort
│   │   ├── mpi/
│   │   │   ├── psrs_mpi.cpp       # 純 MPI PSRS
│   │   │   ├── psrs_hybrid.cpp    # MPI + OpenMP 混合
│   │   │   └── pipeline_sort.cpp  # 流水線插入排序（逐元素版 + 分塊版）
│   │   └── cuda/
│   │       ├── gpu_sort.cu        # 單 GPU：thrust::sort 基線 + 自寫樣本排序（層次化原子直方圖）
│   │       └── gpu_multi.cu       # 雙 GPU 樣本排序
│   ├── tests/
│   │   └── test_correctness.sh    # 正確性回歸測試（邊界規模 × 所有分佈 × 所有實現）
│   ├── scripts/
│   │   ├── env_info.sh            # 記錄 lscpu / numactl -H / nvidia-smi / 編譯器版本
│   │   ├── bench_*.sh             # 每個實驗一個腳本（E1–E13）
│   │   └── run_all.sh             # 依次執行全部實驗
│   ├── analysis/
│   │   ├── summarize.py           # 讀 results/*.csv → 中位數表格（Markdown）
│   │   └── plot.py                # 生成全部圖表到 ../report/figures/
│   └── results/                   # 原始 CSV + env.txt（實驗產物）
└── report/                        # 報告（與代碼分開）
    ├── report.md                  # 報告正文（簡體中文），最終按老師模板排版成 docx/pdf
    ├── figures/                   # 由 plot.py 生成的圖 + 手工截圖 + 示意圖
    └── tables/                    # 由 summarize.py 生成的 Markdown 表格
```

---

## 3. 環境與構建

### 3.1 依賴（Ubuntu 22.04）

```bash
sudo apt install build-essential numactl libnuma-dev hwloc openmpi-bin libopenmpi-dev python3-pandas python3-matplotlib
```
可選：`libtbb-dev`（tbb::parallel_sort 參考）、`pyspark`（第 7 講小實驗）。

CUDA：需要 CUDA Toolkit ≥ 11.8（RTX 4090D 為 sm_89；推薦 12.x），Thrust 隨 Toolkit 自帶。先用 `nvidia-smi`、`nvcc --version` 確認；若沒有 sudo 權限，在 README 中說明使用 conda 或用戶目錄安裝。

### 3.2 編譯選項

- C++17，`g++ -O3 -march=native -fopenmp`，MPI 用 `mpicxx`，CUDA 用 `nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler -fopenmp`。
- `Makefile` 目標：`all`、`seq`、`pthreads`、`openmp`、`mpi`、`cuda`、`test`、`clean`。輸出到 `code/bin/`。
- 編譯期打印 `CUDA: enabled/disabled`，找不到 `nvcc` 時跳過 `cuda` 目標而不是報錯。

### 3.3 運行注意事項（寫進 README）

- OpenMPI 默認以物理核計 slot（64 個），跑 128 進程要加 `--use-hwthread-cpus`（或 `--oversubscribe`），並說明這會用到超線程。
- 純 MPI：`mpirun -np P --bind-to core --map-by core ./bin/psrs_mpi ...`
- 混合：每 socket 1 個進程、每進程 32 線程：`mpirun -np 2 --map-by ppr:1:package:PE=32 --bind-to core -x OMP_NUM_THREADS=32 -x OMP_PROC_BIND=close -x OMP_PLACES=cores ./bin/psrs_hybrid ...`；另測每 NUMA 節點 1 進程（用 `numactl -H` 查看節點數，EPYC 可能是 NPS1/NPS2/NPS4）。
- OpenMP：默認 `OMP_PROC_BIND=close OMP_PLACES=cores`，線程數用 `--threads` 或 `OMP_NUM_THREADS`。
- GPU：用 `CUDA_VISIBLE_DEVICES` 選卡；`env_info.sh` 記錄 `nvidia-smi topo -m`（兩卡之間是 PCIe 還是同一 PCIe switch，GeForce 一般不支持 P2P/NVLink）。

---

## 4. 公共框架（所有實現共用）

### 4.1 數據類型

- `using Key = uint32_t;`（不要命名為 `key_t`，會與 `<sys/types.h>` 衝突）。
- 規模 n 用 `uint64_t`；MPI 中各進程本地數量必須 < 2³¹（`Alltoallv` 的 count 是 int），在代碼中斷言。

### 4.2 統一命令行（`cli.hpp`）

```
--n <N>            元素個數，支持 2^30、1e9、268435456 等寫法
--dist <name>      uniform | normal | power | hotkeys | sorted | reverse | nearly_sorted | few_unique
--seed <S>         默認 42
--threads <T>      線程數（OpenMP/Pthreads/混合中每進程線程數）
--variant <name>   實現內部變體（見各模塊）
--repeat <R>       重複次數，默認 5；另做 1 次不計入的熱身
--oversample <k>   採樣過採樣係數，默認 1（每塊取 k·p 個樣本）
--buckets-per-thread <k>   過分解係數，默認 1
--schedule <s>     static | dynamic | guided（OpenMP 歸併階段）
--no-tiebreak      關閉複合鍵 tie-break
--csv <file>       追加結果到 CSV
--verify / --no-verify     默認開啟校驗
```

### 4.3 確定性、可並行的數據生成（`datagen`）

關鍵要求：**第 g 個元素的值只取決於 (seed, g)**，與線程數、進程數無關。這樣：
- 所有實現、所有 p 下排序的是完全相同的數據；
- 每個線程/進程可以只生成自己那一塊（MPI 不需要 rank 0 生成再 Scatter；OpenMP 可以用 first-touch 並行生成）。

做法：`h(g) = splitmix64(seed ^ (g * 0x9E3779B97F4A7C15))`，`u = (h >> 11) * 2^-53 ∈ [0,1)`。各分佈：

| 名稱 | 定義 | 作用 |
|---|---|---|
| uniform | `h >> 32` | 理想情況 |
| normal | Box–Muller（用 h(g) 與 splitmix64(h(g)) 兩個均勻數），均值 2³¹、標準差 2²⁷，截斷到 [0, 2³²) | 中度偏斜，等寬桶失衡 |
| power | `(Key)(u^8 · 2³²)` | 強偏斜（大量集中在小值） |
| hotkeys | 若 `u < 0.5` 取 8 個熱點值之一（`hot[h%8]`），否則 uniform | 大量重複鍵，PSRS 無 tie-break 時失衡 |
| sorted | `(Key)(g · 2³² / n)` | 已有序 |
| reverse | `(Key)((n-1-g) · 2³² / n)` | 逆序 |
| nearly_sorted | sorted，但 1% 的位置（`h % 100 == 0`）替換為隨機值 | 近似有序 |
| few_unique | `(h % 16) · (2³² / 16)` | 只有 16 種值 |

### 4.4 校驗（`verify`）

- **有序性**：並行檢查 `a[i] <= a[i+1]`；MPI 中每個 rank 檢查本地有序，並用 `MPI_Sendrecv` 把本地最大值發給右鄰居，檢查跨進程邊界。
- **排列不變（多重集相等）**：排序前後分別計算順序無關的校驗和 `sum = Σ mix64(x) mod 2⁶⁴` 和 `xr = XOR mix64(x)`，並比較元素總數；OpenMP 用 `reduction(+:sum) reduction(^:xr)`，MPI 用 `MPI_Allreduce`。
- 測試中再對小規模數據與 `std::sort` 結果逐元素比對。
- 校驗時間**不計入**排序時間。

### 4.5 計時與 CSV

- CPU 計時用 `std::chrono::steady_clock`；MPI 用 `MPI_Barrier` + `MPI_Wtime`，記錄**各 rank 的最大值**（`MPI_Reduce(MPI_MAX)`）；GPU 用 `cudaEvent`。
- 計時範圍：從「數據已在內存中」到「有序結果已寫入輸出緩衝」。**數據生成不計入**。GPU 版本分別記錄 H2D、kernel/sort、D2H，並給出「含拷貝」與「不含拷貝」兩種時間。
- 分階段計時 `PhaseTimes`：`local_sort, sampling, splitters, partition, exchange, merge, h2d, d2h, other`（不適用的填 0）。
- 失衡度 `imbalance = max_bucket_size / (n / p)`（理想 = 1.0；PSRS 理論上界 2.0）。
- 每次運行向 CSV 追加一行，字段固定：

```
timestamp,host,impl,variant,dist,n,procs,threads,gpus,repeat_id,total_ms,local_sort_ms,sampling_ms,splitters_ms,partition_ms,exchange_ms,merge_ms,h2d_ms,d2h_ms,imbalance,verified,extra
```
`impl` 取值：`seq, pthreads, omp, mpi, hybrid, pipeline, gpu, gpu_multi, ref`。`extra` 放鍵值對（如 `schedule=dynamic;bind=close`）。

---

## 5. 模塊設計與算法

> 記號：n 為總元素數，p 為處理單元（線程 / 進程 / GPU）數，m = n/p。塊 i = 下標區間 `[i·n/p, (i+1)·n/p)`（用 64 位整數計算，n 不能被 p 整除時同樣正確）。

### 5.1 串行基線（`seq_sort`）

- `--variant std`：`std::sort`。**這是所有加速比的分母 T_serial**（課件第 1 講第 35 頁：應比較串行程序在單處理器上的性能與並行程序在 n 個處理器上的性能）。
- `--variant bucket`：串行桶排序（課件第 2 講第 93–96 頁），桶數 B 默認 1024，等寬，桶內 `std::sort`。用於說明桶排序本身的思路。

### 5.2 Pthreads：等寬並行桶排序（V1）

對應課件第 2 講「並行 bucket sort」與第 5 講 Pthreads 鎖/路障。

- 預處理：各線程求本塊 min/max，歸約得全局 [lo, hi]；桶 j 的值域為 `[lo + j·w, lo + (j+1)·w)`，w = ⌈(hi-lo+1)/p⌉，共 p 個桶，桶 j 映射到線程 j（靜態映射）。
- `--variant locked`（反面教材，課件「簡單方法：每一桶分配一處理器」的直接實現）：p 個共享 `std::vector<Key>` 桶，每桶一把 `pthread_mutex_t`；每個線程掃描本塊，每個元素加鎖後 push 到目標桶。用於展示**臨界區與鎖競爭**的代價，只在較小規模（如 2²⁴）運行。
- `--variant local`（課件「進一步優化：每一處理器維護 p 個小桶，bucket i 發給 processor i」的共享內存版）：
  1. 計數：線程 t 統計本塊落入各桶的數量 `cnt[t][j]`（每行 `alignas(64)` 避免 false sharing）。
  2. `pthread_barrier_wait`。
  3. 前綴和：`off[t][j] = Σ_{j'<j} Σ_t' cnt[t'][j'] + Σ_{t'<t} cnt[t'][j]`（可由線程 0 計算後再 barrier，p² 規模很小）。
  4. 分發：線程 t 把本塊元素寫到輸出數組 `out[off[t][j]++]`，**無鎖**。
  5. barrier；線程 j 對桶 j 所在區間 `std::sort`。
- 線程只創建一次（`pthread_create` / `pthread_join`），參數用結構體傳入；記錄 imbalance。

### 5.3 OpenMP：PSRS（Parallel Sorting by Regular Sampling）—— 本項目的核心算法

PSRS（Shi & Schaeffer, 1992）是 BSP 風格的 4 超步算法。每一步都要能對應到 PCAM：

```
超步 1  本地排序      ：線程 i 對塊 i std::sort                         （輸入分解，O(m log m)）
超步 2  正則採樣      ：線程 i 從有序塊中取 k·p 個等間距樣本               （k = oversample）
        選分割點      ：single 線程合併 k·p² 個樣本、排序，取 p-1 個分割點
超步 3  劃分          ：線程 i 用二分查找把有序塊切成 p 段，得到 p×p 的段邊界矩陣
        （共享內存中「交換」= 計算偏移，前綴和得每個輸出桶在結果數組中的位置）
超步 4  多路歸併      ：線程 j 把 p 段（各來自一個塊、都已有序）歸併為輸出桶 j
```

細節：

- **正則採樣**：塊 i 長 m_i，取下標 `⌊s·m_i/(k·p)⌋`，s = 0..k·p-1。m_i = 0 時不貢獻樣本（共享內存版可直接跳過）。
- **選分割點**：樣本總數 S = k·p²（實際數量），排序後取下標 `⌊(j·S)/p⌋`，j = 1..p-1，作為分割點 `spl[j-1]`。在報告中說明經典 PSRS 結論：k=1 時每個桶 ≤ 2m（鍵互不相同時）。
- **正確性不變量（寫進報告）**：只要分割點非遞減，且每個元素被唯一地放進「分割點區間」對應的桶、桶內歸併正確，結果就一定有序；分割點的好壞只影響負載均衡，不影響正確性。這也是為什麼退化情況（樣本不足）可以安全處理。
- **劃分**：段邊界 `b[i][j] = upper_bound(塊 i, spl[j-1])`（`b[i][0]=0, b[i][p]=m_i`）。
- **複合鍵 tie-break（默認開啟，`--no-tiebreak` 關閉）**：把每個元素視為三元組 (key, block_id, pos_in_sorted_block)，三元組兩兩不同，因此 2m 上界對任意重複程度都成立。實現時樣本攜帶 (key, block, pos)；劃分時對塊 i 和分割點 (k, sb, sp)：鍵 < k 的去左側、> k 的去右側，鍵 == k 的那一段（在有序塊中連續）再按 `(i, pos) < (sb, sp)` 字典序二分切開。熱點鍵實驗用來證明它的效果。
- **輸出偏移**：`dst[j] = Σ_i Σ_{j'<j} (b[i][j'+1]-b[i][j'])`，桶 j 在輸出數組中佔 `[dst[j], dst[j+1])`。
- **多路歸併**：p 路歸併用二叉小頂堆（或敗者樹），O(m log p)。p=1 時直接拷貝。
- **過分解 + 調度實驗**（對應課件 OpenMP schedule 子句 和 第 2 講動態映射）：`--buckets-per-thread k` 時選 k·p-1 個分割點、形成 k·p 個輸出桶，歸併階段 `#pragma omp parallel for schedule(<s>)` 遍歷桶。比較 static / dynamic / guided 在偏斜數據上的效果。
- **NUMA 處理**：輸入數組和輸出數組都用與排序階段相同的 `schedule(static)` 並行初始化（first-touch），使每個線程的數據落在本地 NUMA 節點；`--variant serial_init` 用單線程初始化作為對照。
- 內存：輸入 n + 輸出 n + O(p²) 元數據；結果在輸出數組中。

`bucket_omp.cpp`：用 OpenMP 重寫 5.2 的 local 版本（等寬桶），與 PSRS 在同一框架下公平比較，證明「失衡是劃分策略造成的，而不是 Pthreads 與 OpenMP 的區別」。

`ref_sorts.cpp`（參考上限，報告中作為「工業實現」對照，不作為本人的算法貢獻）：
- `--variant gnu_parallel`：`__gnu_parallel::sort`（libstdc++ 並行模式，基於 OpenMP 的多路歸併排序）。
- `--variant omp_task_qsort`：OpenMP task 並行快排（遞歸分解，課件第 2 講；與 Cilk/TBB 工作竊取的思想對應），子問題小於 cutoff（如 2¹⁶）時串行 `std::sort`。
- `--variant tbb`（可選，有 libtbb-dev 時）：`tbb::parallel_sort`。

### 5.4 MPI：PSRS（分佈式內存）

將 5.3 映射到消息傳遞模型（課件第 3 講 MP 模型、第 6 講集合通信）：

| 步驟 | 操作 | MPI 調用 |
|---|---|---|
| 0 生成 | 每個 rank 用計數器式生成器直接生成自己的塊（`--scatter` 時改為 rank 0 生成全量後 `MPI_Scatterv` 分發，並單獨計時，用來展示 Scatter 與其代價） | (`MPI_Scatterv`) |
| 1 本地排序 | `std::sort` | — |
| 2 採樣 | 每 rank 取 k·p 個樣本（塊為空或不足時用 `UINT32_MAX` 填充——依據 5.3 的不變量，只影響均衡不影響正確性） | `MPI_Gather` 到 root |
| 3 分割點 | root 排序樣本、選 p-1 個分割點 | `MPI_Bcast` |
| 4 劃分 | 二分查找得到發往各 rank 的段 | — |
| 5 交換計數 | 每 rank 告知其他 rank 將發送多少元素 | `MPI_Alltoall` |
| 6 數據全交換 | 段 j 發給 rank j（課件第 2 講「Sends data from each process to every other process」） | `MPI_Alltoallv` |
| 7 歸併 | p 個有序段多路歸併 | — |
| 8 校驗/統計 | 邊界檢查、校驗和、桶大小、各階段最大時間 | `MPI_Sendrecv`、`MPI_Allreduce`、`MPI_Gather`、`MPI_Reduce(MAX)` |

- tie-break 在 MPI 中同樣可做：樣本攜帶 (key, rank, pos)，用 `MPI_Type_create_struct` 或打包成 3×uint64 的 contiguous 類型（順帶展示課件中的派生數據類型）。
- 報告中給出 BSP 代價分析：T ≈ (m log m) + k·p² log(k·p²)（root）+ g·(m)（全交換，g 為單位數據通信代價）+ m log p + 若干次 L（路障/集合通信延遲）。解釋 p 增大時 root 的 O(p² log p) 和 Alltoall 的 O(p) 延遲項會成為 Amdahl 意義上的串行/開銷部分。
- `--variant allgather`：用 `MPI_Allgather` 讓每個 rank 自己選分割點（省一次 Bcast、但每個 rank 都排序 p² 樣本），作為小對比。

### 5.5 MPI + OpenMP 混合（`psrs_hybrid`）

對應課件第 3 講 Hybrid parallel programming 與第 1 講 NUMA：

- `MPI_Init_thread(MPI_THREAD_FUNNELED)`；每個 socket（或每個 NUMA 節點）1 個 rank，rank 內 T 個 OpenMP 線程。
- rank 內：用 5.3 的共享內存 PSRS（或 `__gnu_parallel::sort`）完成本地排序；樣本、分割點、劃分與 5.4 相同；Alltoallv 只由主線程調用；最後的 p 路歸併改為「按值域把本 rank 結果再切成 T 份，T 個線程並行歸併」（即在 rank 內再做一次基於分割點的輸出劃分）。
- 意義：進程數從 128 降到 2–8，Alltoall 的消息數從 O(p²)=16384 降到很少，root 樣本量從 p² 降低；同時每個 rank 的內存都在本地 NUMA 節點。

### 5.6 流水線排序（`pipeline_sort`，課件第 4 講）

- `--variant elem`（課件第 26–29 頁原版，Type 2 流水線）：p 個進程排成線性流水線，P0 依次送入 n=p 個數；Pi 收到 number 後若 number > x 則把 x 送給 Pi+1 並保留 number，否則把 number 送給 Pi+1。最終 P0 保留最大值……Pn-1 保留最小值，`MPI_Gather` 收集後倒序即有序。只用於演示和驗證（n = p ≤ 128）。
- `--variant block`（回答課件第 30 頁的問題「如果數列長度遠大於處理單元數量時，如何設計流水線」）：每個 Pi 維護一個容量 m=n/p 的有序緩衝；P0 把輸入按批（大小 b，默認 b=m/8）送入；每個階段把收到的批與緩衝歸併，**保留最大的 m 個、把其餘（較小的）作為一批發給下一階段**；輸入結束後發送結束標誌。最後 Pi 持有第 i 大的一段，收集即得結果。
  - 用 `MPI_Isend`/`MPI_Irecv` 雙緩衝讓「接收下一批」與「歸併當前批」重疊（課件第 6 講非阻塞通信）。
  - 報告中畫空間-時間圖，用課件公式分析：流水線不減少單個任務的時間，總時間 ≈ (p-1)·t_stage + 批數·t_max_stage；說明它在排序上不如 PSRS 的原因（數據必須流經全部 p 個階段、每元素被處理 O(p) 次 → 總工作量 O(n·p) 量級而非 O(n log n)），只適合流式輸入/在線排序等場景——這是「針對性選擇並行模式」的反面論證。
  - 規模：n ≤ 2²⁴，p ∈ {2,4,8,16,32}。

### 5.7 CUDA：單 GPU 樣本排序（`gpu_sort.cu`，課件第 8 講）

- `--variant thrust`：`thrust::sort`（在 GPU 上對 uint32 是基數排序）作為 GPU 基線和上限參考。
- `--variant sample`（本人實現，展示 CUDA 編程要點）：
  1. H2D：用 pinned memory（`cudaMallocHost`）+ `cudaMemcpyAsync`。
  2. 採樣：從設備數組等距取 S = 32·B 個樣本（B 為桶數，取值見第 6 步），`thrust::sort` 樣本後取 B-1 個分割點，存於全局內存（B 大時放不進 `__constant__`），桶號用二分查找求得。
  3. **桶計數 kernel**：每個線程對若干元素做二分查找求桶號，先在 `__shared__` 直方圖上 `atomicAdd`，`__syncthreads()` 後每個 block 把局部直方圖 `atomicAdd` 到全局直方圖——這正是課件第 85 頁「Hierarchical Atomics」的應用；加一個 `--naive-atomics` 開關直接對全局內存做 atomicAdd，對比資源競爭（課件第 84 頁）。
  4. `thrust::exclusive_scan` 求各桶起始偏移。
  5. **分發 kernel**：用 block 級前綴偏移 + `atomicAdd` 取得寫位置，把元素寫入目標桶（桶內順序無所謂）。
  6. 桶內排序：每桶各調一次 `thrust::sort` 會有 B 次 kernel 啟動，太慢。**做法**：一個 CUDA block 負責一個桶，桶大小 ≤ shared memory 容量（48KB / 4B = 12288 個元素，pad 到 2 的冪）時把桶讀入 `__shared__`，用雙調排序（bitonic sort，配合 `__syncthreads()`）排好後寫回；超出容量的少數大桶記錄下來，事後逐個用 `thrust::sort` 處理。B 取 n/4096 向上取 2 的冪（n=2³⁰ 時 B = 2¹⁸，平均每桶 4096 個），使絕大多數桶都放得進 shared memory；報告中說明這一參數選擇的理由，並統計大桶個數。
  7. D2H。
- 線程組織：Grid/Block 維度、warp=32、避免分支發散（二分查找是少量分支）；在報告中畫線程映射圖。
- 計時：`cudaEventRecord` 分階段記錄；報告中說明 PCIe 拷貝（H2D + D2H）在總時間中的佔比——GPU 加速受 Amdahl 意義上「拷貝」這一串行部分限制。
- 規模：單卡 24GB，n=2³⁰ 的 uint32 為 4GB，加上輸出緩衝與臨時空間足夠；n=2³¹ 也可嘗試。

### 5.8 CUDA：雙 GPU 樣本排序（`gpu_multi.cu`）

把 PSRS 思想用到 p=2 塊 GPU 上，體現「同一算法在不同層次的並行硬件上的映射」：

1. 主機用 OpenMP 開 2 個線程，線程 g 調用 `cudaSetDevice(g)` 並使用自己的 stream（OpenMP + CUDA 協同）。
2. 每個 GPU 拷入一半數據並 `thrust::sort`（本地排序）。
3. 每個 GPU 取 k·2 個正則樣本拷回主機，主機選 1 個分割點（p=2 時就是兩組樣本合併後的中位數）。
4. 每個 GPU 用 `thrust::upper_bound` 得到切分位置，把「屬於另一塊卡」的一段經主機 pinned 緩衝傳過去（GeForce 一般無 P2P；先用 `cudaDeviceCanAccessPeer` 檢測，可用時走 `cudaMemcpyPeerAsync`，否則經主機中轉，報告中記錄實際走的路徑）。
5. 每個 GPU 用 `thrust::merge` 歸併「本地保留段 + 收到的段」。
6. 各自 D2H 寫回輸出數組的對應位置。

報告中分析：雙卡相比單卡的加速比、交換開銷佔比、為什麼未必接近 2（PCIe 中轉、主機選分割點的同步點）。

### 5.9 MapReduce/Spark 視角（報告章節；代碼可選）

- 不必實現，但報告中要有一節說明：Hadoop TeraSort 用 `TotalOrderPartitioner`（先採樣得分割點）、Spark `sortByKey` 用 `RangePartitioner`（蓄水池採樣得分割點）——**兩者本質上就是樣本排序**；Map 階段 = 本地處理，Shuffle = Alltoallv，Reduce 端 sort/merge = 多路歸併。用一張對應表說明本項目各階段與 MapReduce 各階段的對應（課件第 7 講 Partitioner / Shuffle and Sort）。
- 可選代碼：`code/src/mapreduce/spark_sort.py`，用 PySpark 在本機 `local[*]` 對 10⁷ 個數 `sortByKey`，只作為功能演示，不做性能比較（JVM 啟動開銷等不可比）。

---

## 6. 實驗設計（scripts/bench_*.sh 一一對應）

通用規則：熱身 1 次 + 正式 R=5 次，報告**中位數**，誤差條用最小/最大值；串行基線 R=3。每個腳本先調用 `env_info.sh`，把結果追加到 `results/<實驗名>.csv`。默認 `OMP_PROC_BIND=close OMP_PLACES=cores`。

| 編號 | 實驗 | 參數 | 輸出圖表 | 報告中要說明的結論 |
|---|---|---|---|---|
| E1 | 正確性 | n ∈ {0,1,2,7,1000,10⁶+3}，所有 dist，p ∈ {1,2,3,7,8,16}，所有實現 | 通過/失敗表 + 終端截圖 | 所有實現在邊界情況下正確 |
| E2 | 強擴展 | n = 2³⁰（4GB），uniform；p ∈ {1,2,4,8,16,32,64,128}；impl：bucket_omp, psrs_omp, psrs_mpi, hybrid, gnu_parallel, omp_task_qsort | 加速比曲線（含理想線）、效率曲線 | 加速比 S、效率 E；64→128（超線程）收益 |
| E3 | 弱擴展 | 每處理單元 2²⁴ 個元素，p 同上（n 最大 2³¹=8GB） | 時間 vs p、Gustafson 縮放加速比 | 與 Gustafson 定律對照 |
| E4 | 數據分佈與負載均衡 | n = 2²⁸，p = 64，全部 8 種分佈；bucket_omp vs psrs_omp(--no-tiebreak) vs psrs_omp | 分組柱狀圖（時間）+ imbalance 柱狀圖 | **核心「針對性」結論**：等寬桶在 normal/power 上失衡；PSRS 解決；hotkeys 需 tie-break |
| E5 | 階段時間分解 | n = 2³⁰，p ∈ {8,32,128}，psrs_omp 與 psrs_mpi | 堆疊柱狀圖 | 哪一階段成為瓶頸；p 大時分割點選擇/通信佔比上升 |
| E6 | OpenMP 調度與過分解 | n = 2²⁸，normal 和 power，p = 64；buckets-per-thread ∈ {1,4,16}，schedule ∈ {static,dynamic,guided} | 熱力圖或分組柱狀圖 | 動態映射緩解剩餘失衡，但過分解增加歸併路數 |
| E7 | NUMA 與綁定 | n = 2³⁰，p = 64；`OMP_PROC_BIND` ∈ {false, close, spread}；first-touch vs serial_init；`numactl --interleave=all` | 分組柱狀圖 | NUMA 遠程訪問代價（課件 UMA/NUMA） |
| E8 | 鎖競爭 | n = 2²⁴，p ∈ {1,2,4,8,16,32,64}；pthreads locked vs local | 時間曲線（對數軸） | 臨界區串行化使「加線程反而變慢」 |
| E9 | MPI 通信方式 | n = 2³⁰；純 MPI p ∈ {2..128} vs 混合（2、4、8 個 rank × 相應線程）；Gather+Bcast vs Allgather | 柱狀圖 + exchange 階段時間 | 混合模型減少消息數與通信開銷 |
| E10 | 流水線 | elem：n=p ∈ {4..64}；block：n ∈ {2²⁰,2²²,2²⁴}，p ∈ {2..32}，批大小 b ∈ {m/32, m/8, m/2} | 時間曲線 + 空間-時間示意圖 | 流水線適用條件；與 PSRS 對比 |
| E11 | GPU | n ∈ {2²⁴,2²⁶,2²⁸,2³⁰}；thrust vs sample（含/不含拷貝）；naive vs hierarchical atomics；單卡 vs 雙卡 | 時間曲線 + 階段堆疊圖 | GPU 計算極快，但 PCIe 拷貝佔大頭；層次化原子的作用 |
| E12 | 全局對比 | n = 2³⁰ uniform：seq、最佳 CPU（128 線程/進程）、單 GPU、雙 GPU | 一張總覽柱狀圖（對數軸）+ 總表 | 綜合結論 |
| E13 | 定律擬合 | 由 E2 數據用最小二乘擬合 Amdahl 模型 T(p) = T₁(f + (1-f)/p) + T_o(p)，並計算 Karp–Flatt 實驗串行比例 e(p) = (1/S − 1/p)/(1 − 1/p) | 擬合曲線 + e(p) 曲線 | f 的估計值；e(p) 隨 p 上升說明開銷而非固有串行部分在限制擴展（Karp–Flatt 為課件外的補充指標，報告中註明） |

預計耗時：串行 2³⁰ 約 1–2 分鐘/次，全部實驗在服務器上約 2–4 小時；`run_all.sh` 支持 `--quick`（所有規模縮小 16 倍）用於調試。

**需要用戶手工截的圖**（在報告中預留位置並在 README 中列出）：
1. `make` 編譯成功的終端截圖；
2. E1 測試全部 PASS 的終端截圖；
3. 128 線程運行時的 `htop`（所有 CPU 滿載）；
4. GPU 運行時的 `nvidia-smi`（兩卡均有負載）；
5. `lscpu` / `numactl -H` / `nvidia-smi topo -m` 的輸出（環境說明）。

---

## 7. 分析與繪圖腳本

- `analysis/summarize.py`：讀取 `results/*.csv`，按 (impl, variant, dist, n, procs, threads, gpus) 分組取中位數，計算 speedup（分母為同 n、同 dist 的 `seq/std` 中位數）、efficiency、imbalance，輸出到 `report/tables/*.md`。
- `analysis/plot.py`：生成 E2–E13 的圖到 `report/figures/`，PNG（dpi 200）+ PDF 各一份；中文字體用 `Noto Sans CJK SC` 或 `SimHei`，找不到時退回英文標籤；圖的標題、軸標籤、圖例齊全，加速比圖畫理想線 y=x。
- 兩個腳本都只依賴 pandas + matplotlib，可在任何機器上離線運行。

---

## 8. 報告大綱（`report/report.md`，簡體中文；最終按老師提供的模板排版）

> 按課程要求的順序組織：背景 → 總體介紹 → 模塊構成 → 主要算法 → 測試；全文主線是「如何針對性地用並行程序解決問題」。正文建議 12–20 頁。

**摘要**（200–300 字）：問題、方法（PSRS + 多範式實現）、主要結果（最佳加速比、負載均衡改善倍數、GPU 結果）。

**1 背景與總體介紹**
- 1.1 排序在大數據處理中的地位（數據庫、MapReduce Shuffle、TeraSort）；串行排序在 10⁹ 規模下的耗時（引用 E2 的串行實測值）。
- 1.2 並行化動機：多核/眾核/GPU 已成為主流（引用課件第 0、1 講的技術趨勢）。
- 1.3 本文工作與貢獻（3–4 條）。
- 1.4 實驗平台（表格：CPU、內存、GPU、OS、編譯器、MPI、CUDA 版本）。

**2 問題分析與並行算法設計（PCAM）**
- 2.1 分解：輸入分解（塊）+ 輸出/值域分解（桶）；任務依賴圖與關鍵路徑；並行度。
- 2.2 通信：排序的本質困難——任意元素可能去往任意位置 → 需要全交換；通信量分析。
- 2.3 組合：為什麼按塊組合而不是按元素（粒度與通信開銷）。
- 2.4 映射：靜態映射與負載均衡；等寬桶的失衡問題的定量分析（以正態分佈為例計算最大桶比例）→ 引出採樣。
- 2.5 計算模型：用 BSP 描述 PSRS 的超步並給出代價公式。

**3 系統總體設計與模塊構成**
- 系統架構圖（公共框架 + 6 個實現模塊 + 實驗/分析腳本）。
- 各模塊職責表；統一的數據生成/校驗/計時設計（說明為什麼要用計數器式生成器保證公平對比）。

**4 主要算法與實現思路**（每節：思路 + 示意圖 + ≤15 行偽代碼 + 複雜度）
- 4.1 等寬並行桶排序（Pthreads）：帶鎖版與無鎖「本地小桶」版。
- 4.2 PSRS（OpenMP）：四個超步的圖示；2n/p 上界；正確性不變量；tie-break。
- 4.3 MPI PSRS：通信步驟時序圖（Gather → Bcast → Alltoall → Alltoallv）。
- 4.4 MPI+OpenMP 混合：進程/線程到 socket/NUMA 的映射圖。
- 4.5 流水線排序：逐元素版與分塊版、空間-時間圖。
- 4.6 GPU 樣本排序：線程組織、層次化原子直方圖、shared memory 雙調排序；雙 GPU 協同流程圖。
- 4.7 與 MapReduce/Spark 的對應關係。

**5 針對性優化**（這是報告的重點章，每條都要「問題現象 → 原因分析 → 措施 → 實驗證據」）
- 5.1 數據偏斜導致的負載失衡 → 正則採樣（E4）。
- 5.2 重複鍵導致的失衡 → 複合鍵 tie-break（E4 hotkeys）。
- 5.3 剩餘失衡 → 過分解 + 動態調度（E6）。
- 5.4 鎖競爭 → 計數-前綴和-分發的無鎖方案（E8）；false sharing → 緩存行對齊。
- 5.5 NUMA 遠程訪問 → first-touch + 線程綁定（E7）。
- 5.6 大規模進程的通信開銷 → 混合編程（E9）。
- 5.7 GPU 全局原子競爭 → 層次化原子（E11）；PCIe 拷貝 → pinned memory + 異步拷貝。

**6 測試**
- 6.1 正確性測試（E1 表格 + 截圖）。
- 6.2 強擴展與效率（E2）；6.3 弱擴展（E3）；6.4 階段分解（E5）；6.5 流水線（E10）；6.6 GPU（E11）；6.7 總覽對比（E12）。
- 每張圖下面寫 2–4 句「觀察 + 原因」。

**7 性能分析：Amdahl 與 Gustafson**
- 用 E13 的擬合給出 f 與 T_o 的估計；解釋為什麼 64→128 線程收益有限（超線程共享執行單元、內存帶寬飽和）；排序是訪存密集型——估算內存帶寬上限並與實測對照。

**8 總結與思考**
- 逐條回答課件第 2 講第 101 頁「思考」：本例採用的並行計算模型（BSP）、劃分方法（輸入分解 + 值域分解）、映射方法（靜態映射 + 採樣均衡；可選動態調度）、算法模型（數據並行模型 + 全交換；流水線版為 Pipeline 模型；OpenMP task 版為任務圖/工作竊取）、主要分解方法的挑戰（負載均衡、全交換通信、重複鍵）。
- 不足與展望（外部排序/超出內存的數據、多節點集群、GPU 基數排序等）。

**參考文獻**：課程課件；Shi & Schaeffer 1992（PSRS）；Li et al. 1993（PSRS 分析）；Valiant 1990（BSP）；Dean & Ghemawat 2004（MapReduce）；O'Malley 2008（TeraSort）；NVIDIA CUDA C++ Programming Guide；OpenMP 5.x / MPI 4.0 標準；Karp & Flatt 1990。

**附錄**：代碼目錄說明、復現步驟（指向 `code/README.md`）。

---

## 9. 實現里程碑與驗收條件

| 里程碑 | 內容 | 驗收條件 |
|---|---|---|
| M0 | 目錄骨架、Makefile、`common`（Key/CLI/datagen/verify/timer/CSV）、`seq_sort` | `make seq` 成功；`seq_sort --n 1e6 --dist uniform` 輸出 verified=1 的 CSV 行；同一 seed 多次生成結果相同；按塊生成與整體生成結果逐元素一致 |
| M1 | Pthreads 桶排序（locked/local）、OpenMP 桶排序 | E1 對這兩個實現全部通過；normal 分佈下 imbalance 明顯 > 1.5 |
| M2 | OpenMP PSRS（含 oversample、tie-break、過分解、調度、first-touch） | E1 通過；uniform 下 imbalance < 1.1；hotkeys 下關閉 tie-break 時 imbalance 大、開啟後 ≤ 2.0 |
| M3 | 參考實現 `ref_sorts`（gnu_parallel、omp_task_qsort、可選 tbb） | E1 通過 |
| M4 | MPI PSRS（含 `--scatter`、`allgather` 變體） | `mpirun -np {1,2,3,7,8}` 下 E1 通過；邊界檢查與校驗和生效 |
| M5 | 混合 MPI+OpenMP | E1 通過；與純 MPI 結果校驗和一致 |
| M6 | 流水線排序（elem、block） | elem：n=p 時正確；block：n 不被 p 整除時正確 |
| M7 | CUDA 單卡（thrust、sample、naive/hierarchical atomics）+ 雙卡 | 與 CPU 結果校驗和一致；無 GPU 的機器上編譯被跳過而不報錯 |
| M8 | `tests/test_correctness.sh`、`scripts/bench_*.sh`、`run_all.sh --quick`、`summarize.py`、`plot.py` | `run_all.sh --quick` 在 8 核機器上 15 分鐘內跑完並生成全部圖表 |
| M9 | `code/README.md`、`report/report.md` 初稿（數據處留 TODO） | 報告結構完整，與第 8 節一致；圖表引用路徑正確 |
| M10 | 用戶在服務器上運行 `run_all.sh`，回填數據與截圖，完成報告 | 報告中所有 TODO 已替換為實測數據；按模板排版；整體打包為「學號-姓名.zip」 |

---

## 10. 編碼規範與常見陷阱（給智能體）

- 所有下標與規模用 `uint64_t`/`size_t`；`n*p`、`i*n/p` 等運算先轉 64 位，防止溢出。
- 採樣、分割點、劃分代碼放在 `psrs_core` 中被 OpenMP/MPI/混合複用，避免三份不一致的實現。
- 計時區間內不得有 I/O、內存分配以外的無關操作；大數組在計時前分配並 first-touch（輸出緩衝同理），否則測到的是缺頁開銷。
- 不要在 OpenMP 並行區內使用 `std::vector::push_back` 寫共享容器。
- 每線程計數器、每線程統計結構 `alignas(64)`。
- MPI：所有 rank 都必須調用集合通信；Alltoallv 的 displacement 用 int，斷言不溢出；結束前 `MPI_Finalize`。
- CUDA：每次 API 調用用宏檢查錯誤；kernel 後 `cudaGetLastError()`；設備內存不足時給出清晰錯誤信息。
- 輸出的每個可執行文件都支持 `--help`。
- 不要把大段代碼複製到報告裡。
