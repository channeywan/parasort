// 纯 MPI PSRS 的命令行入口。
// 对应 DESIGN.md 第 5.4：MPI 样本排序 节。
// 实验配置与校验均沿用公共框架，算法默认参数保持不变。
#include "mpi_support.hpp"
int main(int argc, char **argv) {
    return parasort::mpi_main(argc, argv, false);
}
