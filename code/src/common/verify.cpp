// 检查顺序、元素数量和输出指纹。
// 对应 DESIGN.md 第 4.3：正确性校验 节。
// 实验配置与校验均沿用公共框架，算法默认参数保持不变。
#include "verify.hpp"
#include "datagen.hpp"
#include <cstdlib>
#include <cstring>
namespace parasort {
bool output_fingerprint_requested() {
    const char *value = std::getenv("PARASORT_FINGERPRINT");
    return value && std::strcmp(value, "1") == 0;
}
void append_output_fingerprint(SortResult &result, const Fingerprint &output) {
    if (!result.extra.empty())
        result.extra += ';';
    result.extra += "output_count=" + std::to_string(output.count) +
                    ";output_sum=" + std::to_string(output.sum) +
                    ";output_xor=" + std::to_string(output.xr);
}
Fingerprint fingerprint(const Key *a, uint64_t n, int threads) {
    uint64_t sum = 0, xr = 0;
#pragma omp parallel for num_threads(threads) reduction(+ : sum) reduction(^ : xr) schedule(static)
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t h = mix64(a[i]);
        sum += h;
        xr ^= h;
    }
    return {n, sum, xr};
}
bool ordered(const Key *a, uint64_t n, int threads) {
    int bad = 0;
#pragma omp parallel for num_threads(threads) reduction(| : bad) schedule(static)
    for (uint64_t i = 1; i < n; ++i)
        bad |= a[i - 1] > a[i];
    return !bad;
}
bool verify_result(const Key *a, uint64_t n, const Fingerprint &before,
                   const std::vector<Key> &exact, int threads) {
    return ordered(a, n, threads) && fingerprint(a, n, threads) == before &&
           (exact.empty() || std::equal(exact.begin(), exact.end(), a));
}
} // namespace parasort
