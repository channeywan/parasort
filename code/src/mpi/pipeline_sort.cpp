/*
 * MPI 逐元素与分块流水线排序，使用双接收槽和双发送槽。
 * 对应 DESIGN.md §5.6；每个 rank 保留当前最大的容量份额。
 * 小键持续流向下游，最终按相反的 rank 位移收集为升序输出。
 */
#include "mpi_support.hpp"
using namespace parasort;
/*
 * 每个 rank 保留目前最大的 capacity 个键，将更小的溢出部分发往下游。
 * 接收和发送各有两个轮换槽，槽内向量在对应 MPI 请求结束前必须保持有效。
 */
static std::vector<Key> pipeline(const Key *input, const Config &c, int rank, int p) {
    uint64_t capacity = cut(c.n, p - rank, p) - cut(c.n, p - rank - 1, p);
    uint64_t min_capacity = c.n / uint64_t(p);
    uint64_t batch =
        c.variant == "elem" ? 1 : (c.batch ? c.batch : std::max<uint64_t>(1, min_capacity / 8));
    mpi_int(batch);
    std::vector<Key> held;
    held.reserve(capacity);
    std::vector<Key> recv[2], send[2];
    for (int i = 0; i < 2; ++i) {
        recv[i].resize(batch);
        send[i].reserve(batch);
    }
    MPI_Request receives[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL},
                sends[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    if (rank > 0)
        for (int i = 0; i < 2; ++i)
            mpi_check(MPI_Irecv(recv[i].data(), mpi_int(batch), MPI_UINT32_T, rank - 1, 22,
                                MPI_COMM_WORLD, &receives[i]));
    uint64_t consumed = 0;
    int slot = 0;
    while (true) {
        uint64_t count = 0;
        if (rank == 0) {
            count = std::min(batch, c.n - consumed);
            if (count)
                std::copy(input + consumed, input + consumed + count, recv[slot].begin());
            consumed += count;
        } else {
            /*
             * 只有当前接收槽的请求完成后才读取其中的键。
             * 随后将收到的键复制到 packet，避免后续预取覆盖本轮仍需使用的数据。
             */
            MPI_Status status;
            mpi_check(MPI_Wait(&receives[slot], &status));
            int size = 0;
            mpi_check(MPI_Get_count(&status, MPI_UINT32_T, &size));
            count = size;
        }
        if (!count)
            break; // Zero-length packet is the ordered end marker.
        std::vector<Key> packet(recv[slot].begin(), recv[slot].begin() + count);
        std::sort(packet.begin(), packet.end());
        std::vector<Key> combined;
        combined.reserve(held.size() + count);
        std::merge(held.begin(), held.end(), packet.begin(), packet.end(),
                   std::back_inserter(combined));
        uint64_t spill = combined.size() > capacity ? combined.size() - capacity : 0;
        if (rank < p - 1) {
            /*
             * 重用发送槽前先等待上一次 Isend 完成，再替换该槽向量中的键。
             * 否则 assign 可能改写或重新分配 MPI 尚在读取的发送缓冲区。
             */
            mpi_check(MPI_Wait(&sends[slot], MPI_STATUS_IGNORE));
            send[slot].assign(combined.begin(), combined.begin() + spill);
            if (spill)
                mpi_check(MPI_Isend(send[slot].data(), mpi_int(spill), MPI_UINT32_T, rank + 1, 22,
                                    MPI_COMM_WORLD, &sends[slot]));
        } else if (spill)
            throw std::runtime_error("pipeline capacity invariant failed");
        held.assign(combined.begin() + spill, combined.end());
        /*
         * 当前槽的数据已复制并完成本轮处理，此时才能在原接收向量上重新发布 Irecv。
         * 另一个槽可同时接收下一包；轮换槽号实现预取与本地处理的交叠。
         */
        if (rank > 0)
            mpi_check(MPI_Irecv(recv[slot].data(), mpi_int(batch), MPI_UINT32_T, rank - 1, 22,
                                MPI_COMM_WORLD, &receives[slot]));
        slot ^= 1;
    }
    /*
     * 零长度包是有序的结束标记；收到后收回未完成的预取，并等待全部在途发送。
     * 向下游传递结束标记前保持发送缓冲有效，函数退出时不留下引用局部向量的 MPI 请求。
     */
    if (rank > 0)
        for (auto &request : receives)
            if (request != MPI_REQUEST_NULL) {
                MPI_Cancel(&request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
    for (auto &request : sends)
        mpi_check(MPI_Wait(&request, MPI_STATUS_IGNORE));
    if (rank < p - 1) {
        MPI_Request done;
        mpi_check(MPI_Isend(nullptr, 0, MPI_UINT32_T, rank + 1, 22, MPI_COMM_WORLD, &done));
        mpi_check(MPI_Wait(&done, MPI_STATUS_IGNORE));
    }
    return held;
}
int main(int argc, char **argv) {
    int provided, rank, p;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    try {
        Config c = parse_cli(argc, argv, "block");
        if (c.help) {
            if (!rank)
                std::cout << help_text();
            MPI_Finalize();
            return 0;
        }
        require_variant(c, {"elem", "block"});
        c.threads = 1;
        mpi_int(c.n);
        if (c.variant == "elem" && c.n != uint64_t(p))
            throw std::invalid_argument("elem requires n == number of MPI ranks");
        auto input = buffer(rank == 0 ? c.n : 0), out = buffer(rank == 0 ? c.n : 0);
        if (!rank)
            first_touch(out.get(), c.n, 1);
        std::vector<int> counts(p), offsets(p);
        for (int r = 0; r < p; ++r) {
            offsets[r] = mpi_int(cut(c.n, p - r - 1, p));
            counts[r] = mpi_int(cut(c.n, p - r, p) - cut(c.n, p - r - 1, p));
        }
        for (int rep = -1; rep < c.repeat; ++rep) {
            Fingerprint before;
            std::vector<Key> exact;
            if (!rank) {
                generate(input.get(), c.n, 0, c);
                if (c.verify) {
                    before = fingerprint(input.get(), c.n);
                    if (c.n <= 100000) {
                        exact.assign(input.get(), input.get() + c.n);
                        std::sort(exact.begin(), exact.end());
                    }
                }
            }
            mpi_check(MPI_Barrier(MPI_COMM_WORLD));
            double start = MPI_Wtime();
            auto held = pipeline(input.get(), c, rank, p);
            // Descending rank order becomes ascending output order through reversed displacements.
            mpi_check(MPI_Gatherv(held.data(), mpi_int(held.size()), MPI_UINT32_T, out.get(),
                                  counts.data(), offsets.data(), MPI_UINT32_T, 0, MPI_COMM_WORLD));
            double elapsed = (MPI_Wtime() - start) * 1000, total = 0;
            mpi_check(MPI_Reduce(&elapsed, &total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
            int verified = -1;
            if (!rank && c.verify)
                verified = verify_result(out.get(), c.n, before, exact, 1);
            mpi_check(MPI_Bcast(&verified, 1, MPI_INT, 0, MPI_COMM_WORLD));
            if (verified == 0)
                throw std::runtime_error("pipeline verification FAILED");
            if (!rank && rep >= 0) {
                SortResult result;
                result.data = out.get();
                result.phase.merge = total;
                result.extra = "batch=" + std::to_string(c.batch) + ";pipeline_includes_gather=1";
                if (output_fingerprint_requested())
                    append_output_fingerprint(result, fingerprint(out.get(), c.n, 1));
                emit_csv(c, "pipeline", result, rep, total, verified, p);
            }
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "rank " << rank << ": " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
