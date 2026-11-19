/*
 * OpenMP PSRS 程序入口，复用公共 PSRS 排序步骤。
 * 对应 DESIGN.md §5.3，接受普通模式和串行初始化对照。
 * 输入生成、重复次数、CSV 与结果校验交给统一 CPU 执行框架。
 */
#include "psrs_core.hpp"
using namespace parasort;
int main(int argc, char **argv) {
    return run_cpu(argc, argv, "omp", "psrs", [](Key *a, Key *o, uint64_t n, const Config &c) {
        require_variant(c, {"psrs", "serial_init"});
        return psrs_openmp(a, o, n, c);
    });
}
