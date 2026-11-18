# ParaSort 并行排序课程项目

项目依据随附 DESIGN.md 实现。代码、运行说明、实验脚本和原始 CSV 位于 `code/`，简体中文报告与生成图表位于 `report/`。

本项目依赖只通过 conda 管理，项目环境为 `.conda/`。从项目根目录执行：

```bash
conda activate /home/wcl/parasort/.conda
source code/scripts/activate.sh
cd code
make -j8
bash scripts/run_all.sh --quick
```

完整说明见 `code/README.md`。`IMPLEMENTATION.md` 记录已验证范围和设计中需要调整的实现细节。CPU、MPI、单/双 GPU 均已编译并进行了真实回归；完整规模性能实验及最终课程排版属于设计中的 M10。
