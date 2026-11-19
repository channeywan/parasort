# ParaSort 改進任務書（第二階段）

> 本文檔交給 coding 智能體執行，用來在 `channeywan/parasort`（提交 `9696046`）的基礎上繼續完善。原設計見 `DESIGN.md`，已完成的部分見 `IMPLEMENTATION.md`。
>
> **目標**：讓 E1–E13 能在服務器上以完整規模跑出**可信**的數據；把實驗結果整理成一個「報告數據包」；最後把代碼和提交包整理到可以交給老師的程度。
>
> **分工**：智能體**只負責編碼**，包括修改程序、實驗腳本、分析和繪圖腳本、打包腳本。**報告正文（`report/report.md`）由另一位撰寫者完成，智能體不得修改它。** 智能體與報告撰寫者之間的唯一接口是 T7 導出的數據包 `report/data_bundle/`。
>
> **運行環境**：與第一階段相同，即 2×EPYC 7542（8 個 NUMA 節點，NPS4）、2×RTX 4090D、Ubuntu 22.04，項目 conda 環境（MPICH、CUDA 12.6）。依賴仍然**只用 conda 安裝**。

---

## 0. 總體規則（每個任務都要遵守）

1. **按優先級順序執行**：P0 → P1 → P2。每個任務都有「驗收」一節，通過了才能進入下一個任務。
2. **不得改變排序算法的語義和默認參數**。PSRS、tie-break、桶排序、流水線、GPU 各算法的行為保持不變。本輪只修**實驗方法**、補**實驗項**、改**可讀性**、導出**報告數據包**。
3. **數據紀律**：
   - 數據包中的每個數字都必須來自 `code/results/` 中 `verified=1` 的 CSV，**不得估算、外推或編造**；缺數據時對應字段填 `null`，並在 `missing` 列表中說明原因。
   - 快速數據（`*_quick.csv`）和完整數據（`E*.csv`）不得混用；數據包默認只導出完整數據。
   - 綁核修復前產生的 MPI、混合、流水線數據一律作廢，見 T1。
4. 智能體如果自己不能在服務器上獨佔運行完整實驗，就把命令準備好並寫進 `code/README.md` 的「正式實驗」一節，由用戶執行。
5. 每完成一個任務都單獨提交一次 commit，提交信息寫清任務編號，例如 `T1: bind MPI ranks to distinct cores`。
6. **不要修改 `report/report.md`**，也不要新建其他報告正文文件。智能體能寫入 `report/` 的只有三處：`report/figures/`、`report/tables/`、`report/data_bundle/`，都由腳本生成。
7. 圖表中的文字用**簡體中文**（報告是簡體中文）；本文檔和代碼註釋不受此限。

---

## P0 —— 不做就沒有可信數據

### T1 修復 MPI / 混合 / 流水線的進程綁核（最高優先級）

**問題**

`scripts/activate.sh` 對所有程序都導出了 `OMP_PROC_BIND=close` 和 `OMP_PLACES=cores`。MPI 程序也以 `-fopenmp` 鏈接，`datagen.cpp`、`verify.cpp` 裡有 `#pragma omp parallel`。libgomp 在 `OMP_PROC_BIND` 不為 false 時，會在啟動時把初始線程綁到 place 列表的第一個 place，也就是 core 0。而 `bench.py` 用 MPICH 的 `mpiexec` 啟動時不帶任何綁定參數（`MPIEXEC_FLAGS` 默認為空），每個 rank 看到的都是整機的 cpuset，於是**所有 rank 都被綁到 core 0**。混合模式同理：每個 rank 的線程都從 core 0 開始排，不同 rank 之間互相重疊。

快速 E2 的數據支持這個判斷：純 MPI 在 p=2、4、8 時分別是 13.4、53.8、88.5 ms，進程越多越慢；混合 2×4 為 10.1 ms，而 OpenMP 8 線程只要 4.5 ms。

**步驟**

1. **先寫一個診斷工具** `code/src/tools/affinity_probe.cpp`（MPI + OpenMP），在 Makefile 中加入 `mpi` 目標。
   - 每個 rank、每個 OpenMP 線程打印一行：`rank thread sched_getcpu() cpuset(列表) hostname`。
   - 程序最後由 rank 0 匯總，並檢查兩件事：
     - 純 MPI（每 rank 1 線程）時，各 rank 的 cpuset 兩兩不相交；
     - 混合模式時，不同 rank 的 cpuset 不相交，且同一 rank 內各線程分佈在該 rank 的 cpuset 內。
   - 輸出 `AFFINITY OK` 或 `AFFINITY BAD`，並以相應的退出碼返回。
2. **復現問題**：新建 `scripts/check_affinity.sh`，分別用「當前默認啟動方式」和「修復後的啟動方式」運行 `affinity_probe`。把修復前的輸出保存到 `results/validation/affinity_before.log`，作為報告附錄的證據。
3. **在 `bench.py` 中集中生成啟動參數**。新增函數 `launch_flags(exe, procs, threads)`：
   - 先識別 MPI 實現：`mpiexec --version` 輸出含 `HYDRA` 時為 MPICH，含 `Open MPI` 時為 OpenMPI。
   - MPICH（Hydra）的映射規則如下。具體語法要先用 `mpiexec -bind-to -help` 或 `mpiexec -help` 核對當前版本，以 `affinity_probe` 的實際結果為準：

     | 場景 | 參數 | 環境變量 |
     |---|---|---|
     | 純 MPI 和流水線，procs ≤ 64 | `-bind-to core -map-by core` | `OMP_NUM_THREADS=1`、`OMP_PROC_BIND=false` |
     | 純 MPI 和流水線，procs > 64 | `-bind-to hwthread -map-by hwthread` | 同上 |
     | 混合，ranks = 2 | `-bind-to socket -map-by socket` | `OMP_PROC_BIND=close`、`OMP_PLACES=cores` |
     | 混合，ranks = 8 | `-bind-to numa -map-by numa` | 同上 |
     | 混合，ranks = 4 | `-bind-to numa:2 -map-by numa:2`（每 rank 2 個 NUMA 節點） | 同上 |
     | 混合，每 rank 線程數 > 該 rank 的物理核數（超線程） | 同上 | 把 `OMP_PLACES` 改為 `threads` |

   - OpenMPI：使用 `code/README.md` 中已有的 `--bind-to/--map-by` 寫法。
   - 用戶顯式設置了 `MPIEXEC_FLAGS` 時，以用戶設置為準，並在日誌中打印警告。
   - **純 MPI 的環境變量不要依賴全局 activate**：在 `run()` 中按場景顯式傳入，覆蓋 activate.sh 的默認值。
4. **在 CSV 中記錄綁定信息**：在 `extra` 中追加 `launcher=<mpich|openmpi>;mpi_bind=<core|hwthread|socket|numa|numa:2|user>`。`summarize.py` 的 `TUNING` 列表加上 `mpi_bind`，避免不同綁定方式的數據被混在一起統計。
5. **作廢舊數據**：把 `results/` 中現有的 `E2/E3/E5/E9/E10/E12_quick.csv` 移到 `results/archive_pre_bindfix/`，並在其中放一個 `README.txt` 說明作廢原因。然後重跑 `run_all.sh --quick`。

**驗收**

- 在 p ∈ {2,4,8,16} 和混合 2×8 時，`check_affinity.sh` 都輸出 `AFFINITY OK`。
- 新增一個健全性檢查：純 MPI 在 n = 2²² 時，p=8 的總時間應**低於** p=2。不滿足就停下來排查，不要繼續跑完整實驗。
- 重跑後的快速 E2 中，混合 2×4 不再明顯慢於 OpenMP 8 線程（差距在 2 倍以內）。

### T2 讓 GPU 驅動線程綁到 GPU 所在的 NUMA 節點

**問題**

`gpu_multi.cu` 用 `#pragma omp parallel for num_threads(2)` 驅動兩塊卡。在 `OMP_PROC_BIND=close` 下，這兩個線程會落在 core 0 和 core 1 上，而 GPU0 掛在 NUMA 2、GPU1 掛在 NUMA 7（見 `env.txt`）。結果是 pinned 緩衝區和主機中轉都跨了 NUMA。`gpu_sort.cu` 的單線程也有同樣的問題。

**步驟**

1. 在 `gpu_support.cuh` 中加入 `bind_to_gpu_numa(int dev)`：
   - 用 `cudaDeviceGetPCIBusId` 取得 PCI 地址；
   - 轉成小寫後讀 `/sys/bus/pci/devices/<addr>/numa_node`；
   - 讀 `/sys/devices/system/node/node<k>/cpulist` 並解析成 cpu 集合；
   - 用 `pthread_setaffinity_np` 綁定當前線程；
   - 讀取失敗或 numa_node 為 -1 時不綁定，並把 `gpu_affinity=none(reason)` 寫入 extra。
2. 新增命令行參數 `--gpu-affinity numa|none`，默認 `numa`。在 `cli.cpp` 中加上，`--help` 裡也要寫明。
3. 綁定必須發生在 `cudaMallocHost` 之前，並由同一線程對 pinned 緩衝區做 first-touch，這樣頁面才會落在 GPU 本地的 NUMA 節點上。
4. 在 E11 中加一組對比：`gpu_multi` 在 n = 2³⁰ 時分別用 `numa` 和 `none`。`plot.py` 的 E11 圖增加這一組。

**驗收**

- `nvidia-smi topo -m` 顯示的 CPU Affinity 與程序實際綁定的 CPU 一致（在日誌中打印出來）。
- GPU_validation 回歸測試仍然全部 `verified=1`。

### T3 讓完整實驗可以穩定地跑完

完整規模預計要 4–6 小時，必須支持斷點續跑，並能事後追溯每一次運行。

**步驟**

1. `run_all.sh` 增加以下參數：
   - `--only E2,E4,...`：只跑指定的實驗；
   - `--force`：重跑已成功的實驗。默認跳過 `*_status.json` 中 `success=true` 且非 quick 的實驗。
   - `--dry-run`：只打印將要執行的命令和估計時間。估計時間根據串行 `std::sort` 約 90 s / 2³⁰ 元素的經驗值，按 n·log n 縮放；對並行運行按 p 粗略折算。這個估計只用來提示，不寫進報告。
2. 每個實驗的標準輸出和錯誤輸出寫到 `results/logs/<E>.log`。
3. **起跑前檢查**：
   - 讀 `/proc/loadavg`，1 分鐘負載大於 2 時打印警告，要求用戶確認或加 `--ignore-load` 才繼續；
   - 用 `nvidia-smi --query-compute-apps` 檢查 GPU 上是否有其他進程；
   - 把 `lscpu`、`numactl -H`、`nvidia-smi topo -m`、負載和 git commit id 寫入 `results/env.txt`。
4. 在 `code/README.md` 中新增「正式實驗」一節，寫出完整的運行流程：
   ```bash
   tmux new -s parasort
   conda activate <项目环境>
   source code/scripts/activate.sh
   cd code
   make -j16 all
   bash scripts/check_affinity.sh
   bash scripts/run_all.sh --dry-run
   bash scripts/run_all.sh 2>&1 | tee results/logs/run_all.log
   ```

**驗收**

- `run_all.sh --quick` 中途用 Ctrl-C 中斷後再次運行，能從中斷的實驗繼續。
- `--dry-run` 列出全部命令，並且不產生任何 CSV。

---

## P1 —— 讓「針對性」結論更有說服力

### T4 擴展 E4：過採樣係數，以及小 p 時 tie-break 的反常

**問題**

快速 E4 在 p=8 時，開啟 tie-break 後 hotkeys 的失衡度從 1.05 升到了 1.44，few_unique 從 1.01 升到 1.43。原因是 p=8 恰好等於熱點值的個數，按鍵切分時巧合地對齊了桶邊界。p=64 時就是另一回事了：M2 專項測得 4.10 → 1.38。數據上需要把這兩種情況都體現出來。另外，PSRS 的均衡程度取決於過採樣係數 k，目前沒有任何實驗展示這一取捨。

**步驟**

在 `bench.py` 中新增 `E4b`（實驗編號擴展為 E1–E14 再加 E4b；`run_all.sh` 和 `plot.py` 同步修改）：

- **E4b-1 過採樣掃描**：n = 2²⁸，p = 64，dist ∈ {uniform, normal, hotkeys}，`--oversample` ∈ {1, 2, 4, 8, 16, 32}，tie-break 開啟。
  - 圖：橫軸 k，左縱軸 imbalance，右縱軸 total_ms；另畫一張 `splitters_ms` 隨 k 變化的圖。
  - 要說明的結論：k 越大越均衡，但選分割點的開銷按 O(k·p² log(k·p²)) 增長，存在最優區間。
- **E4b-2 tie-break 與 p 的關係**：n = 2²⁸，dist ∈ {hotkeys, few_unique}，p ∈ {8, 16, 32, 64, 128}，tie-break 開啟與關閉各跑一次。
  - 圖：imbalance 隨 p 變化，兩條線（開 / 關）。
  - 要說明的結論：當 p 小於等於不同熱點值的個數時，按鍵切分可能巧合地均衡；p 越大，不開 tie-break 時失衡越嚴重（熱點值必然落在單個桶內），開啟後保持在 2 以下。

**驗收**

- 兩張新圖生成到 `report/figures/E4b_*.png/pdf`，對應的表格生成到 `report/tables/E4b.md`。
- p = 64 時，hotkeys 不開 tie-break 的失衡度大於 3，開啟後小於等於 2。若實測不符，如實記錄在數據包中，不要改代碼去湊結果。

### T5 新增 E14：內存帶寬測量與有效帶寬分析

**問題**

報告需要解釋 64→128 線程的收益為什麼有限，以及排序是否已經碰到內存帶寬上限。目前沒有實測帶寬。

**步驟**

1. 新增 `code/src/tools/stream_bw.cpp`（OpenMP 版 STREAM，只實現 Copy 和 Triad 兩個核）：
   - 數組長度默認 2²⁸ 個 double，並行 first-touch；
   - 線程數 ∈ {1, 2, 4, 8, 16, 32, 64, 128}，分別在 `OMP_PROC_BIND=close` 和 `spread` 下測試；
   - 每組重複 5 次，取最好的一次（STREAM 的慣例）；
   - 輸出到 `results/E14.csv`，字段為 `threads, bind, kernel, best_GBps, median_GBps`。
   - 不要另寫一套 CSV 框架，可以複用 `csv.cpp` 中的工具函數，也可以單獨寫一個簡單的 CSV 輸出。
2. 在 `summarize.py` 中計算 PSRS 各階段的**有效帶寬**：
   - 歸併階段每個元素讀一次、寫一次：`8n 字節 / merge_ms`；
   - 本地排序和劃分階段不做帶寬估計：它們不是訪存受限的單遍掃描，按字節數估帶寬沒有意義。在 T7 的 `methods.md` 中記錄這一點。
3. `plot.py` 新增 E14 圖：STREAM 帶寬隨線程數的曲線，疊加 PSRS 歸併階段的有效帶寬（取 E2、E5 的完整數據）。

**驗收**

- 128 線程 close 綁定的 Triad 帶寬落在合理量級：雙路 8 通道 DDR4-3200 的理論峰值約 409 GB/s，實測一般在百 GB/s 量級。結果明顯不合理時要檢查 first-touch 和編譯選項。
- E14 的數據寫入 T7 數據包的 `e14_bandwidth` 字段。

### T6 確保 E2、E3、E5、E9 覆蓋課件要求的對比點

**步驟**

1. **E2 中 p = 128 的 OpenMP 設置**：`OMP_PLACES` 改用 `threads`（`cores` 下 128 個線程也會每核排兩個，但寫成 `threads` 語義更清楚），並記錄在 extra 中。
2. **E2 中的混合實現**：目前固定為 2 個 rank。在 p ≥ 16 時增加 8 rank × (p/8) 線程的配置，即每 NUMA 節點一個 rank，並使用 T1 的綁定規則。
3. **E3 弱擴展**：補上 `psrs_hybrid`（8 rank）和 `bucket_omp`，用來對比負載均衡在弱擴展下的影響。
4. **E9**：確認混合配置 2、4、8 rank 都用了 T1 的綁定規則；`--scatter` 對照也要在 p ∈ {8, 64} 時各跑一次，證明「各 rank 本地生成」比「root 生成後 Scatter」更好。

**驗收**

- `run_all.sh --dry-run` 的輸出中包含上述所有新增配置。

---

## P1 —— 為報告撰寫準備數據（等完整數據到位後再做）

> 報告正文不由智能體撰寫。本節的任務是把撰寫報告所需的全部事實整理成一個**自包含的數據包**，使撰寫者不必閱讀代碼或原始 CSV，也不必登錄服務器。

### T7 導出報告數據包 `report/data_bundle/`

新增腳本 `code/analysis/export_for_report.py`，依賴只用 pandas，由 `run_all.sh` 在 E13 之後自動調用，也可以單獨運行。它讀取 `code/results/` 和 `report/tables/`，生成以下文件：

**1. `key_numbers.json`：撰寫摘要和結論要用的關鍵數字**

每個數字都要附帶出處，格式為：

```json
{"value": 12.3, "unit": "x", "source": "tables/E2.md", "config": {"impl": "omp", "variant": "psrs", "n": 1073741824, "procs": 1, "threads": 64, "dist": "uniform"}}
```

必須包含以下鍵；缺數據時 `value` 填 `null`，並寫入 `missing` 列表說明原因：

| 鍵 | 含義 |
|---|---|
| `seq_std_2p30_ms` | n=2³⁰、uniform 時串行 `std::sort` 的中位時間 |
| `best_cpu` | CPU 上的最佳實現：impl、variant、p、time_ms、speedup、efficiency |
| `best_by_impl` | 每種 CPU 實現在 E2 中的最佳 p 及其 time_ms、speedup、efficiency |
| `speedup_64_vs_128` | 每種實現在 p=64 和 p=128 時的加速比，以及兩者之比 |
| `e4_skew` | E4 中 normal 和 power 分佈下，等寬桶和 PSRS 的 time_ms、imbalance，以及兩者之比 |
| `e4_hotkeys_tiebreak` | E4 中 hotkeys 和 few_unique 分佈下，PSRS 開 / 關 tie-break 的 imbalance 和 time_ms |
| `e4b_oversample_best` | E4b-1 中每種分佈下 total_ms 最低的 k，以及對應的 imbalance |
| `e4b_tiebreak_vs_p` | E4b-2 中每個 p 的開 / 關 tie-break imbalance |
| `e5_phase_share` | E5 中每種配置各階段佔總時間的百分比 |
| `e6_best_schedule` | E6 中每種分佈的最佳 (buckets_per_thread, schedule) 組合，及其相對 static、k=1 的提升 |
| `e7_numa` | E7 中 first-touch 對 serial_init、close / spread / false、interleave 的時間 |
| `e8_lock` | E8 中 locked 和 local 在各 p 下的時間，以及 locked 時間最短時的 p |
| `e9_comm` | E9 中純 MPI 各 p 的 exchange_ms 和 total_ms、混合 2/4/8 rank 的對應值、scatter 和本地生成的對比 |
| `e10_pipeline` | E10 中 elem 版各 p 的時間、block 版最佳批大小，以及與同規模 PSRS-MPI 的時間比 |
| `e11_gpu` | E11 中各規模下 thrust、sample（naive / hierarchical）、雙卡的 total_ms 和 compute_ms、拷貝佔比、雙卡相對單卡加速比、numa 與 none 綁定對比、實際交換路徑（peer 或 pinned_host） |
| `e12_overview` | E12 中各方案的 total_ms，以及相對串行的加速比 |
| `e13_amdahl` | 擬合得到的 f、α、擬合誤差、使用的數據點，以及 Karp–Flatt e(p) 序列 |
| `e14_bandwidth` | STREAM Triad 和 Copy 在各線程數下的最佳帶寬，以及 PSRS 歸併階段的有效帶寬 |
| `correctness` | E1 的總組數、通過數、覆蓋的 n / 分佈 / p / 實現列表 |
| `affinity_fix` | 綁核修復前後純 MPI p=2/4/8 的時間對比（修復前取自 archive_pre_bindfix 的快速數據，修復後取自修復後重跑的快速數據，在 `note` 中註明規模），以及 `affinity_before.log` 的摘要（各 rank 所綁的 CPU） |

**2. `tables/`：給報告直接使用的精簡表格**

這是從 `report/tables/` 精選出來的版本：每張表不超過 8 列，列名用簡體中文，數字保留 3 位有效數字。每個實驗一張，文件名為 `E2.md` … `E14.md`、`E4b.md`。

**3. `figures_manifest.md`：圖表清單**

每張圖一行，寫出文件路徑（相對 `report/`）、圖題（簡體中文）、對應實驗、數據來源表、圖中每條曲線 / 每組柱子的含義。

**4. `environment.md`：實驗環境信息**

從 `results/env.txt` 提取，整理成「項目 | 值」的表格。要包含：CPU 型號、核數 / 線程數、NUMA 節點數及每節點 CPU 列表、內存、GPU 型號和顯存、GPU 所在 NUMA 節點、`nvidia-smi topo -m` 中的卡間連接類型、OS / 內核、gcc / nvcc / MPICH / Python 版本、git commit id、實驗開始和結束時間。

**5. `methods.md`：實驗方法的事實記錄**

只寫事實，不寫分析，條目式：
- 每個實驗的參數矩陣、重複次數、統計方法（中位數、min / max）；
- MPI 和 OpenMP 的實際綁定方式（取自 T1 的 extra 字段）；
- 計時起止點；
- 被跳過的配置及原因（取自 `*_status.json` 的 skipped）。

**6. `implementation_notes.md`：撰寫算法章節要用的實現事實**

由智能體手寫，不要求腳本生成，只寫事實、不寫評價。內容包括：
- 第一階段 `IMPLEMENTATION.md` 中「對設計細節的必要調整」8 條的要點；
- 本輪 T1–T6 中所有影響實驗結果的改動（綁定規則、GPU NUMA 綁定、新增實驗）；
- 每個實現的關鍵參數默認值（oversample、bitonic 容量、GPU 桶數公式、block 大小、流水線默認批大小等）；
- 已知局限（例如 MPI int count 上限、大桶回退 Thrust）。

**7. `README.md`：數據包說明**

說明數據包內各文件的用途，以及生成時間、所用 CSV 文件列表（含行數）。

**驗收**

- 在只有快速數據時運行腳本，不報錯，並把所有完整數據字段標為 `null`、寫進 `missing`。用參數 `--include-quick` 可以改為導出快速數據，此時每個值都帶 `"quick": true` 標記。
- 完整數據到位後，`key_numbers.json` 的 `missing` 為空，或只包含確實跳過的配置。
- 隨機抽查 5 個數字，都能在 `report/tables/` 的原始表中找到。

### T8 圖表質量

1. `plot.py` 全部圖使用中文標籤（找不到 CJK 字體時先 `conda install -c conda-forge font-ttf-noto-cjk` 或同類包，**不要退回英文**）。軸標籤要寫單位，例如「时间 (ms)」「加速比」「处理单元数 p」。
2. 圖例裡的 `omp/psrs` 這類內部標籤，要映射成可讀的中文名稱，例如「OpenMP PSRS」「OpenMP 等宽桶」「MPI PSRS」「MPI+OpenMP 混合」「GNU parallel sort（参考）」。在 `plot.py` 頂部集中定義一張映射表。
3. 加速比圖要同時有對數 x 軸（p = 1…128）和理想線 y = p。
4. E5 的堆疊柱狀圖：在 MPI 的圖註中說明「各阶段取各 rank 最大值，堆叠高度可能大于总时间」。
5. 每張圖都要登記在 T7 的 `figures_manifest.md` 中；舊的 `*_quick` 圖移到 `report/figures/quick/`，不要留在 `report/figures/` 頂層。

**驗收**

- 用戶目視檢查全部圖：沒有方框亂碼，沒有重疊的圖例。

---

## P2 —— 可讀性與提交整理

### T9 代碼可讀性重排（不改行為）

**問題**

源碼是一行塞多條語句的緊湊寫法，例如 `psrs_omp.cpp` 全部只有 3 行、`psrs_core.cpp` 最長一行有 278 個字符，關鍵步驟幾乎沒有註釋。課程要求提交代碼，老師可能會讀。

**步驟**

1. 在 `code/` 下新增 `.clang-format`：`BasedOnStyle: LLVM`、`IndentWidth: 4`、`ColumnLimit: 100`、`AllowShortFunctionsOnASingleLine: None`、`AllowShortIfStatementsOnASingleLine: Never`、`AllowShortLoopsOnASingleLine: false`。用 conda 安裝 `clang-format`，對所有 `.cpp/.hpp/.cu/.cuh` 執行格式化。
2. clang-format 不會拆開用分號連接的多條語句，需要**手工**改成一行一條語句。重點文件：`psrs_core.cpp`、`mpi_engine.cpp`、`pipeline_sort.cpp`、`gpu_sort.cu`、`gpu_multi.cu`、`bucket_pthreads.cpp`、`cli.cpp`。
3. 在以下位置加**中文塊註釋**，每處 2–6 行，說明「做什麼、為什麼」，不要逐行翻譯代碼：
   - PSRS 的四個超步入口（OpenMP、MPI、混合各一處）；
   - `split_position` 中複合鍵 tie-break 的三種情況（源塊號小於、大於、等於分割點所在塊）；
   - 多路歸併的堆結構；
   - 無鎖桶排序的「計數 → 前綴和 → 分發」；
   - 流水線的雙緩衝槽：何時可以重新發布 Irecv，何時可以重用發送緩衝；
   - GPU 共享哈希直方圖（層次化原子）和按桶批量預留寫入位置；
   - bitonic 排序的 shared memory 容量選擇（8192 元素）；
   - 雙卡交換中檢測 P2P、回退到 pinned host 中轉的邏輯。
4. Python 腳本（`bench.py`、`summarize.py`、`plot.py`）用 `black`（conda 安裝）格式化，並把一行多語句拆開。
5. 每個源文件的開頭加 3–5 行文件頭註釋，說明模塊作用，以及它對應 `DESIGN.md` 的哪一節。

**驗收（必須證明行為沒變）**

- 重排前後各跑一次 `tests/test_correctness.sh --quick` 和 GPU 回歸測試，全部通過。
- 用固定的 seed 和 n（2²⁰），對所有實現比較重排前後輸出的校驗和，必須逐一相同。把比較腳本放在 `tests/compare_fingerprints.sh`。
- `git diff --stat` 中只出現格式和註釋改動，以及 T1–T6 中明確列出的功能改動。

### T10 提交包整理

**步驟**

1. **去掉絕對路徑**：
   - `code/README.md`、`scripts/activate.sh` 中的 `/home/wcl/...` 改為相對路徑或 `<项目目录>` 佔位符；
   - `bench.py` 寫入 `commands.jsonl` 時，把命令中的路徑轉成相對 `code/` 的路徑；
   - 已有的 `results/commands.jsonl` 和 `results/validation/*.log` 用腳本做一次路徑替換。
2. **新增打包腳本** `code/scripts/package.sh <学号> <姓名>`：
   - 在倉庫外生成 `<学号>-<姓名>.zip`，內含 `<学号>-<姓名>/code/` 和 `<学号>-<姓名>/report/`；
   - 排除 `bin/`、`build/`、`*.o`、`.conda/`、`results/.mpl-cache/`、`results/archive_pre_bindfix/`、`__pycache__/`；
   - `report/` 中包含 `report.md`、`figures/`、`tables/`，以及用戶排版後的 `report.docx` 或 `report.pdf`（存在時才打包）；
   - 打包後做兩項檢查：用 `zipinfo` 確認不含 `/home/`、`wcl`、`.conda` 字樣；打印壓縮包大小，超過 50 MB 時給出警告。
3. **根目錄的 `DESIGN.md`、`IMPLEMENTATION.md` 不打包**；`report/data_bundle/` 也不打包，它只是給報告撰寫者用的中間材料。
4. **截圖清單**：新建目錄 `report/figures/screenshots/`，在 `code/README.md` 中列出需要用戶手工截取的 5 張圖及文件名：
   - `01_build.png`：`make` 編譯成功；
   - `02_tests_pass.png`：E1 全部 PASS；
   - `03_htop_128.png`：128 線程運行時的 htop；
   - `04_nvidia_smi_dual.png`：雙卡同時有負載；
   - `05_env.png`：`lscpu`、`numactl -H`、`nvidia-smi topo -m`。

   報告撰寫者會在正文中按這些文件名引用截圖，文件名不要改。

**驗收**

- `package.sh 000 测试` 生成的壓縮包解壓後只有 `code/` 和 `report/` 兩個目錄，並且能在新目錄中執行 `make` 編譯。

---

## 執行順序總覽

```
T1 綁核修復 ──► T2 GPU NUMA ──► T3 續跑基礎設施 ──► T4/T5/T6 補實驗（只改 bench/plot/新工具）
      │                                                     │
      └──► run_all.sh --quick 全綠 ─────────────────────────┘
                                                            ▼
                         【用戶在服務器上獨佔運行完整實驗：run_all.sh】
                                                            ▼
                        T8 圖表 ──► T7 導出數據包 ──►【交給報告撰寫者】
                                                            ▼
                    T9 代碼重排（含行為一致性驗證）──► T10 打包（等報告完成後執行）
```

- T9 可以在等待完整實驗期間進行，但必須在**同一個 commit 範圍內**通過行為一致性驗證，並且不能和正在運行的實驗爭搶 CPU。最好在實驗結束後再執行 T9 的驗證步驟。
- 完整實驗開始之前，必須先通過 T1 的健全性檢查和 `check_affinity.sh`。

## 不要做的事

- 不要引入 TBB、Spark 或其他新的運行時依賴（clang-format、black、CJK 字體這類開發工具除外，都用 conda 安裝）。
- 不要為了讓數據「好看」去調整算法默認參數；參數掃描只放在 E4b、E6 這類專門的實驗裡。
- 不要修改 `report/report.md`，也不要用小規模結果推斷十億規模的表現。
- 不要刪除 `results/archive_pre_bindfix/`。它是綁核修復前後對比的證據（T7 的 `affinity_fix` 字段要用），只是不打包提交。
