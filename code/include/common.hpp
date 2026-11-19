// 公共配置、排序结果和计时类型。
// 对应 DESIGN.md 第 4：统一运行框架 节。
// 实验配置与校验均沿用公共框架，算法默认参数保持不变。
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace parasort {
using Key = uint32_t;
struct Config {
    uint64_t n = 1000000, seed = 42, batch = 0;
    int threads = 1, repeat = 5, oversample = 1, buckets_per_thread = 1;
    std::string dist = "uniform", variant, schedule = "static", csv, gpu_affinity = "numa";
    bool verify = true, tiebreak = true, scatter = false, naive_atomics = false, help = false;
};
struct PhaseTimes {
    double local_sort = 0, sampling = 0, splitters = 0, partition = 0, exchange = 0, merge = 0,
           h2d = 0, d2h = 0, other = 0;
};
struct Timer {
    using Clock = std::chrono::steady_clock;
    Clock::time_point start = Clock::now();
    double ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    double lap() {
        double elapsed = ms();
        start = Clock::now();
        return elapsed;
    }
};
struct SortResult {
    Key *data = nullptr;
    PhaseTimes phase;
    double imbalance = 1;
    std::string extra;
};
using Sorter = std::function<SortResult(Key *, Key *, uint64_t, const Config &)>;
using Buffer = std::unique_ptr<Key[]>;
inline Buffer buffer(uint64_t n) {
    return Buffer(new Key[std::max<uint64_t>(1, n)]);
}
inline uint64_t cut(uint64_t n, uint64_t i, uint64_t p) {
    return uint64_t((__uint128_t(n) * i) / p);
}
inline double imbalance(uint64_t largest, uint64_t n, uint64_t p) {
    return n ? double(largest) * double(p) / double(n) : 1.;
}
void emit_csv(const Config &, const std::string &impl, const SortResult &, int repeat, double total,
              int verified, int procs = 1, int gpus = 0);
int run_cpu(int argc, char **argv, const std::string &impl, const std::string &default_variant,
            const Sorter &);
void require_variant(const Config &, std::initializer_list<const char *>);
} // namespace parasort
