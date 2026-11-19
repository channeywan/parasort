// 输出实测阶段时间和实际运行配置。
// 对应 DESIGN.md 第 4.4：CSV 与计时 节。
// 实验配置与校验均沿用公共框架，算法默认参数保持不变。
#include "common.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <ctime>
#include <unistd.h>
namespace parasort {
static const char *header = "timestamp,host,impl,variant,dist,n,procs,threads,gpus,repeat_id,total_"
                            "ms,local_sort_ms,sampling_ms,splitters_ms,partition_ms,exchange_ms,"
                            "merge_ms,h2d_ms,d2h_ms,imbalance,verified,extra";
static std::string quote(std::string x) {
    std::string r = "\"";
    for (char c : x) {
        r += c;
        if (c == '\"')
            r += c;
    }
    return r + '\"';
}
void emit_csv(const Config &c, const std::string &impl, const SortResult &r, int repeat,
              double total, int verified, int procs, int gpus) {
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    auto &p = r.phase;
    std::ostringstream s;
    std::string extras =
        "seed=" + std::to_string(c.seed) + ";oversample=" + std::to_string(c.oversample) +
        ";buckets_per_thread=" + std::to_string(c.buckets_per_thread) + ";schedule=" + c.schedule +
        ";tiebreak=" + std::to_string(c.tiebreak) + ";verify=" + std::to_string(c.verify);
    if (const char *bind = std::getenv("OMP_PROC_BIND"))
        extras += ";bind=" + std::string(bind);
    if (const char *places = std::getenv("OMP_PLACES"))
        extras += ";places=" + std::string(places);
    if (const char *launcher = std::getenv("PARASORT_LAUNCHER"))
        extras += ";launcher=" + std::string(launcher);
    if (const char *binding = std::getenv("PARASORT_MPI_BIND"))
        extras += ";mpi_bind=" + std::string(binding);
    if (const char *policy = std::getenv("PARASORT_MEMPOLICY"))
        extras += ";mempolicy=" + std::string(policy);
    if (!r.extra.empty())
        extras += ';' + r.extra;
    s << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ") << ',' << quote(host) << ',' << impl << ','
      << quote(c.variant) << ',' << c.dist << ',' << c.n << ',' << procs << ',' << c.threads << ','
      << gpus << ',' << repeat << std::setprecision(12) << ',' << total << ',' << p.local_sort
      << ',' << p.sampling << ',' << p.splitters << ',' << p.partition << ',' << p.exchange << ','
      << p.merge << ',' << p.h2d << ',' << p.d2h << ',' << r.imbalance << ',' << verified << ','
      << quote(extras);
    static bool first = true;
    if (first) {
        std::cout << header << '\n';
        first = false;
    }
    std::cout << s.str() << '\n';
    if (!c.csv.empty()) {
        std::filesystem::path path(c.csv);
        if (path.has_parent_path())
            std::filesystem::create_directories(path.parent_path());
        bool empty = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
        std::ofstream f(c.csv, std::ios::app);
        if (!f)
            throw std::runtime_error("cannot open CSV: " + c.csv);
        if (empty) {
            f << header << '\n';
        }
        f << s.str() << '\n';
        if (!f)
            throw std::runtime_error("CSV write failed");
    }
}
} // namespace parasort
