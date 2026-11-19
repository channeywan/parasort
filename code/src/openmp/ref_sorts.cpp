/*
 * GNU 并行排序和 OpenMP 任务快速排序两个参考实现。
 * 对应 DESIGN.md §5 的排序对照及 §6 的扩展性实验。
 * 任务快速排序用三路划分跳过重复键区间，并保留原有递归阈值。
 */
#include "common.hpp"
#include <parallel/algorithm>
#include <omp.h>
using namespace parasort;
static void task_sort(Key *a, uint64_t n, int depth) {
    if (n < 65536 || depth <= 0) {
        std::sort(a, a + n);
        return;
    }
    Key x = a[n / 2], y = a[0], z = a[n - 1];
    Key pivot = std::max(std::min(x, y), std::min(std::max(x, y), z));
    Key *mid = std::partition(a, a + n, [pivot](Key k) {
        return k < pivot;
    });
    Key *right = std::partition(mid, a + n, [pivot](Key k) {
        return k == pivot;
    });
    uint64_t left_n = mid - a, right_n = a + n - right;
#pragma omp task firstprivate(a, left_n, depth)
    task_sort(a, left_n, depth - 1);
#pragma omp task firstprivate(right, right_n, depth)
    task_sort(right, right_n, depth - 1);
#pragma omp taskwait
}
static SortResult reference(Key *a, Key *, uint64_t n, const Config &c) {
    require_variant(c, {"gnu_parallel", "omp_task_qsort"});
    SortResult r;
    r.data = a;
    Timer t;
    if (c.variant == "gnu_parallel")
        __gnu_parallel::sort(a, a + n);
    else {
#pragma omp parallel num_threads(c.threads)
        {
#pragma omp single
            task_sort(a, n, 64);
        }
    }
    r.phase.local_sort = t.ms();
    return r;
}
int main(int argc, char **argv) {
    return run_cpu(argc, argv, "ref", "gnu_parallel", reference);
}
