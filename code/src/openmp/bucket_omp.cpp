/*
 * OpenMP 等宽桶排序，使用线程本地计数与不相交输出区间。
 * 对应 DESIGN.md §5.2、§5.3，作为负载均衡及初始化方式的对照。
 * 桶边界由输入值域决定；所有排序参数与公共计时规则保持统一。
 */
#include "psrs_core.hpp"
using namespace parasort;
static SortResult bucket(Key *a, Key *out, uint64_t n, const Config &c) {
    require_variant(c, {"local", "serial_init"});
    uint64_t p = c.threads, stride = (p + 7) / 8 * 8;
    /*
     * 每个线程独占一行桶计数，行宽补齐到缓存行大小。
     * C++17 的过对齐分配保持行首对齐，减少相邻线程计数时的伪共享。
     */
    struct alignas(64) Counter {
        uint64_t v[8] = {};
    };
    std::vector<Counter> rows(p * stride / 8);
    auto count = [&](uint64_t t, uint64_t b) -> uint64_t & {
        uint64_t i = t * stride + b;
        return rows[i / 8].v[i % 8];
    };
    std::vector<uint64_t> offsets(p * stride), dst(p + 1);
    SortResult r;
    r.data = out;
    Timer timer;
    Key lo = UINT32_MAX, hi = 0;
#pragma omp parallel for num_threads(c.threads) reduction(min : lo) reduction(max : hi)            \
    schedule(static)
    for (uint64_t i = 0; i < n; ++i) {
        lo = std::min(lo, a[i]);
        hi = std::max(hi, a[i]);
    }
    if (!n) {
        lo = hi = 0;
    }
    uint64_t width = (uint64_t(hi) - lo + 1 + p - 1) / p;
    r.phase.other = timer.lap();
/*
 * 先统计线程本地桶计数，再求每个桶及每个线程片段的前缀和。
 * 最终分发只写本线程拥有的输出区间，因此整个分发过程不需要桶锁。
 */
#pragma omp parallel for num_threads(c.threads) schedule(static)
    for (uint64_t t = 0; t < p; ++t)
        for (uint64_t i = cut(n, t, p); i < cut(n, t + 1, p); ++i)
            ++count(t, bucket_for(a[i], lo, width, p));
    for (uint64_t b = 0; b < p; ++b) {
        uint64_t pos = dst[b];
        for (uint64_t t = 0; t < p; ++t) {
            offsets[t * stride + b] = pos;
            pos += count(t, b);
        }
        dst[b + 1] = pos;
    }
    r.phase.partition = timer.lap();
#pragma omp parallel for num_threads(c.threads) schedule(static)
    for (uint64_t t = 0; t < p; ++t)
        for (uint64_t i = cut(n, t, p); i < cut(n, t + 1, p); ++i) {
            auto b = bucket_for(a[i], lo, width, p);
            out[offsets[t * stride + b]++] = a[i];
        }
    r.phase.exchange = timer.lap();
#pragma omp parallel for num_threads(c.threads) schedule(static)
    for (uint64_t b = 0; b < p; ++b)
        std::sort(out + dst[b], out + dst[b + 1]);
    r.phase.local_sort = timer.ms();
    uint64_t largest = 0;
    for (uint64_t b = 0; b < p; ++b)
        largest = std::max(largest, dst[b + 1] - dst[b]);
    r.imbalance = imbalance(largest, n, p);
    return r;
}
int main(int argc, char **argv) {
    return run_cpu(argc, argv, "omp", "local", bucket);
}
