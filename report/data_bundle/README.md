# 报告数据包

- `key_numbers.json`：关键数字、原始 CSV 行号、完整配置、统计方法、推导公式和缺失原因；`source` 相对 report，`raw_sources.file` 相对仓库根目录。
- `tables/`：每实验一张精简表，最多八列，三位有效数字；JSON 保留计算精度。
- `figures_manifest.md`：图题、实验、来源及曲线/柱组说明。
- `environment.md`：保存的环境快照及实验时间范围；不采集导出机器的当前配置。
- `methods.md`：实际参数矩阵、重复数、计时口径、绑定和跳过配置。
- `implementation_notes.md`：算法与参数的实现事实。

本包不会改写 report/report.md。缺失数值为 null；无测量时不外推、不补零。默认完整模式；--include-quick 只选择快速模式，二者不混合。affinity_fix 是单独标记 quick=true 的修复诊断，不构成完整性能结论。

生成时间：2026-11-19T17:58:24.970036+09:00

所选模式：full

读取来源：

所选模式暂无 CSV。

独立的修复快速诊断来源（不混入完整性能统计）：

| CSV 文件 | 总行数 | verified=1 行数 | 排除行数 | SHA256 |
| --- | --- | --- | --- | --- |
| code/results/archive_pre_bindfix/E2_quick.csv | 50 | 50 | 0 | 792f0eea984a250f7544f5e2eee920362164208f18760453c82514a3cb9d66a3 |
| code/results/E2_quick.csv | 50 | 50 | 0 | e278aaa65b4de52f153e4fc2b746691dbec9c1ca9edb72ca90585b7956e36ac9 |

原始行号从 2 开始，包含 CSV 表头；只对该快照哈希有效。raw_sources 引用的行均为 verified=1。排序配置缺少修复后的 MPI 绑定标记时会排除，并列入读取问题。
