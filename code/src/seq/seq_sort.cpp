/*
 * 串行 std::sort 与固定等宽桶排序基线。
 * 对应 DESIGN.md §5.1，为各并行实现提供同规模、同分布的实测基线。
 * 统一框架负责输入生成、预热、重复测量和逐项结果校验。
 */
#include "common.hpp"
#include <numeric>
using namespace parasort;
static SortResult sort_seq(Key *a, Key *out, uint64_t n, const Config &c) {
    require_variant(c, {"std", "bucket"});
    SortResult r;
    Timer t;
    if (c.variant == "std") {
        std::sort(a, a + n);
        r.data = a;
        r.phase.local_sort = t.ms();
        return r;
    }
    constexpr uint64_t B = 1024;
    std::vector<uint64_t> count(B), offset(B + 1);
    for (uint64_t i = 0; i < n; ++i)
        ++count[uint64_t(a[i]) * B >> 32];
    for (uint64_t b = 0; b < B; ++b)
        offset[b + 1] = offset[b] + count[b];
    auto pos = offset;
    for (uint64_t i = 0; i < n; ++i)
        out[pos[uint64_t(a[i]) * B >> 32]++] = a[i];
    r.phase.partition = t.lap();
    for (uint64_t b = 0; b < B; ++b)
        std::sort(out + offset[b], out + offset[b + 1]);
    r.phase.local_sort = t.ms();
    r.data = out;
    r.imbalance = imbalance(*std::max_element(count.begin(), count.end()), n, B);
    return r;
}
int main(int argc, char **argv) {
    return run_cpu(argc, argv, "seq", "std", sort_seq);
}
