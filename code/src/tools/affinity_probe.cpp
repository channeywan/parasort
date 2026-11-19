// 检查 MPI rank 与 OpenMP 线程的实际 CPU 集合。
// 对应 DESIGN.md 第 6：实验方法；IMPROVEMENTS.md T1 节。
// 实验配置与校验均沿用公共框架，算法默认参数保持不变。
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {
using CpuSet = std::set<int>;

CpuSet current_affinity() {
    // CPU_ALLOC also handles machines with CPU identifiers above CPU_SETSIZE.
    int size = std::max(128L, sysconf(_SC_NPROCESSORS_CONF));
    for (; size <= (1 << 20); size *= 2) {
        cpu_set_t *mask = CPU_ALLOC(size);
        if (!mask)
            return {};
        const std::size_t bytes = CPU_ALLOC_SIZE(size);
        CPU_ZERO_S(bytes, mask);
        const int result = sched_getaffinity(0, bytes, mask);
        const int error = errno;
        CpuSet cpus;
        if (result == 0)
            for (int cpu = 0; cpu < size; ++cpu)
                if (CPU_ISSET_S(cpu, bytes, mask))
                    cpus.insert(cpu);
        CPU_FREE(mask);
        if (result == 0 || error != EINVAL)
            return cpus;
    }
    return {};
}

CpuSet rank_domain() {
    // libgomp can bind the initial thread before main(). Its current affinity
    // would then describe just one place, not the launcher's complete rank
    // allocation. The union of all rank-local places preserves that allocation.
    CpuSet cpus;
    for (int place = 0; place < omp_get_num_places(); ++place) {
        const int count = omp_get_place_num_procs(place);
        std::vector<int> ids(count);
        omp_get_place_proc_ids(place, ids.data());
        cpus.insert(ids.begin(), ids.end());
    }
    return cpus.empty() ? current_affinity() : cpus;
}

std::string cpu_list(const CpuSet &cpus) {
    std::ostringstream out;
    for (auto it = cpus.begin(); it != cpus.end(); ++it) {
        if (it != cpus.begin())
            out << ',';
        out << *it;
    }
    return cpus.empty() ? "empty" : out.str();
}

void encode_set(std::ostream &out, const CpuSet &cpus) {
    out << cpus.size();
    for (int cpu : cpus)
        out << ' ' << cpu;
    out << '\n';
}

CpuSet decode_set(std::istream &in) {
    int count = 0;
    in >> count;
    CpuSet cpus;
    for (int i = 0, cpu = 0; i < count; ++i) {
        in >> cpu;
        cpus.insert(cpu);
    }
    return cpus;
}

struct Thread {
    int cpu = -1;
    CpuSet cpus;
};
struct Rank {
    std::string host;
    CpuSet domain;
    std::vector<Thread> threads;
};
} // namespace

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    char hostname[MPI_MAX_PROCESSOR_NAME + 1] = {};
    int hostname_length = 0;
    MPI_Get_processor_name(hostname, &hostname_length);
    const CpuSet domain = rank_domain();
    std::vector<Thread> threads;
#pragma omp parallel
    {
#pragma omp single
        threads.resize(omp_get_num_threads());
        auto &thread = threads[omp_get_thread_num()];
        thread.cpus = current_affinity();
        thread.cpu = sched_getcpu();
    }
    std::ostringstream encoded;
    encoded << hostname << '\n';
    encode_set(encoded, domain);
    encoded << threads.size() << '\n';
    for (const auto &thread : threads) {
        encoded << thread.cpu << '\n';
        encode_set(encoded, thread.cpus);
    }
    const std::string payload = encoded.str();
    const int length = static_cast<int>(payload.size());
    std::vector<int> lengths(ranks), offsets(ranks);
    MPI_Gather(&length, 1, MPI_INT, lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    int total = 0;
    if (rank == 0)
        for (int i = 0; i < ranks; ++i) {
            offsets[i] = total;
            total += lengths[i];
        }
    std::vector<char> all(total);
    MPI_Gatherv(payload.data(), length, MPI_CHAR, all.data(), lengths.data(), offsets.data(),
                MPI_CHAR, 0, MPI_COMM_WORLD);
    int valid = 1;
    if (rank == 0) {
        std::vector<Rank> records(ranks);
        for (int r = 0; r < ranks; ++r) {
            auto &record = records[r];
            std::istringstream in(std::string(all.data() + offsets[r], lengths[r]));
            in >> record.host;
            record.domain = decode_set(in);
            int thread_count = 0;
            in >> thread_count;
            record.threads.resize(thread_count);
            std::cout << "rank=" << r << " rank_cpuset=" << cpu_list(record.domain)
                      << " hostname=" << record.host << '\n';
            if (record.domain.empty() || thread_count == 0)
                valid = 0;
            for (int t = 0; t < thread_count; ++t) {
                auto &thread = record.threads[t];
                in >> thread.cpu;
                thread.cpus = decode_set(in);
                std::cout << "rank=" << r << " thread=" << t << " cpu=" << thread.cpu
                          << " cpuset=" << cpu_list(thread.cpus) << " hostname=" << record.host
                          << '\n';
                if (thread.cpus.empty() || !thread.cpus.count(thread.cpu) ||
                    !std::includes(record.domain.begin(), record.domain.end(), thread.cpus.begin(),
                                   thread.cpus.end())) {
                    valid = 0;
                    std::cout << "ERROR rank=" << r << " thread=" << t
                              << " lies outside its rank domain\n";
                }
            }
            if (!in)
                valid = 0;
        }
        for (int a = 0; a < ranks; ++a)
            for (int b = a + 1; b < ranks; ++b) {
                if (records[a].host != records[b].host)
                    continue;
                CpuSet overlap;
                std::set_intersection(records[a].domain.begin(), records[a].domain.end(),
                                      records[b].domain.begin(), records[b].domain.end(),
                                      std::inserter(overlap, overlap.end()));
                if (!overlap.empty()) {
                    valid = 0;
                    std::cout << "ERROR ranks=" << a << ',' << b << " overlap=" << cpu_list(overlap)
                              << " hostname=" << records[a].host << '\n';
                }
            }
        std::cout << (valid ? "AFFINITY OK" : "AFFINITY BAD") << std::endl;
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
