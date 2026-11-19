// 排序输出和指纹校验接口。
// 对应 DESIGN.md 第 4.3：正确性校验 节。
// 实验配置与校验均沿用公共框架，算法默认参数保持不变。
#pragma once
#include "common.hpp"
namespace parasort {
struct Fingerprint {
    uint64_t count = 0, sum = 0, xr = 0;
    bool operator==(const Fingerprint &o) const {
        return count == o.count && sum == o.sum && xr == o.xr;
    }
};
Fingerprint fingerprint(const Key *, uint64_t, int threads = 1);
bool output_fingerprint_requested();
void append_output_fingerprint(SortResult &, const Fingerprint &);
bool ordered(const Key *, uint64_t, int threads = 1);
bool verify_result(const Key *, uint64_t, const Fingerprint &, const std::vector<Key> &exact,
                   int threads);
} // namespace parasort
