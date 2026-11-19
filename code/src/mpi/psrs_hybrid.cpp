// MPI 与 OpenMP 混合排序入口。
// 对应 DESIGN.md 第 5.5：混合并行 节。
// 实验配置与校验均沿用公共框架，算法默认参数保持不变。
#include "mpi_support.hpp"
int main(int argc, char **argv) {
    return parasort::mpi_main(argc, argv, true);
}
