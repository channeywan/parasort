/*
 * 双 GPU 模块将 PSRS 的排序、采样、分区、交换和归并映射到两块卡。
 * 对应 DESIGN.md §5.8，各驱动线程绑定到目标 GPU 所在的 NUMA 节点。
 * 根据实际能力选择 P2P 或 pinned host 中转，并记录所用交换路径。
 */
#include "gpu_support.cuh"
#include "psrs_core.hpp"
#include <exception>
#include <omp.h>
using namespace parasort;
struct GpuState {
    Stream stream;
    DeviceBuffer<Key> input;
    std::unique_ptr<DeviceBuffer<Key>> received, output;
    explicit GpuState(uint64_t n) : input(n) {
    }
};
template <class F> static void on_two(GpuHostBatch &batch, const Config &c, F operation) {
    on_gpu_devices(batch, c, operation);
}
static SortResult multi_sort(GpuHostBatch &batch, const Config &c) {
    require_variant(c, {"sample"});
    SortResult r;
    r.data = batch.outputs[0];
    uint64_t n = c.n;
    batch.output_sizes[0] = batch.output_sizes[1] = 0;
    if (!n)
        return r;
    uint64_t sizes[2] = {batch.input_sizes[0], batch.input_sizes[1]}, cuts[2] = {};
    std::unique_ptr<GpuState> state[2];
    double h2d[2] = {}, sortms[2] = {}, d2h[2] = {};
    on_two(batch, c, [&](int g) {
        state[g] = std::make_unique<GpuState>(sizes[g]);
        auto &s = *state[g];
        EventTimer t(s.stream.value);
        CUDA_CHECK(cudaMemcpyAsync(s.input.data, batch.inputs[g], sizes[g] * sizeof(Key),
                                   cudaMemcpyHostToDevice, s.stream.value));
        h2d[g] = t.lap();
        thrust::sort(thrust::cuda::par.on(s.stream.value), s.input.data, s.input.data + sizes[g]);
        sortms[g] = t.lap();
    });
    r.phase.h2d = std::max(h2d[0], h2d[1]);
    r.phase.local_sort = std::max(sortms[0], sortms[1]);
    Timer timer;
    std::vector<Sample> per_gpu[2];
    on_two(batch, c, [&](int g) {
        uint64_t count = std::min<uint64_t>(sizes[g], 2 * c.oversample);
        auto &s = *state[g];
        per_gpu[g].resize(count);
        for (uint64_t i = 0; i < count; ++i) {
            uint64_t pos = cut(sizes[g], i, count);
            Key key;
            CUDA_CHECK(cudaMemcpyAsync(&key, s.input.data + pos, sizeof(Key),
                                       cudaMemcpyDeviceToHost, s.stream.value));
            CUDA_CHECK(cudaStreamSynchronize(s.stream.value));
            per_gpu[g][i] = {key, uint64_t(g), pos};
        }
    });
    r.phase.sampling = timer.lap();
    std::vector<Sample> samples = per_gpu[0];
    samples.insert(samples.end(), per_gpu[1].begin(), per_gpu[1].end());
    Sample splitter = select_splitters(std::move(samples), 2)[0];
    r.phase.splitters = timer.lap();
    on_two(batch, c, [&](int g) {
        auto &s = *state[g];
        auto policy = thrust::cuda::par.on(s.stream.value);
        uint64_t lo =
            thrust::lower_bound(policy, s.input.data, s.input.data + sizes[g], Key(splitter.key)) -
            s.input.data;
        uint64_t hi =
            thrust::upper_bound(policy, s.input.data, s.input.data + sizes[g], Key(splitter.key)) -
            s.input.data;
        cuts[g] = !c.tiebreak || uint64_t(g) < splitter.block ? hi
                  : uint64_t(g) > splitter.block ? lo
                                                 : std::clamp(splitter.pos + 1, lo, hi);
    });
    uint64_t sending[2] = {sizes[0] - cuts[0], cuts[1]}, received[2] = {sending[1], sending[0]},
             outsize[2] = {cuts[0] + cuts[1], n - cuts[0] - cuts[1]};
    on_two(batch, c, [&](int g) {
        batch.output_sizes[g] = outsize[g];
        state[g]->received = std::make_unique<DeviceBuffer<Key>>(received[g]);
        state[g]->output = std::make_unique<DeviceBuffer<Key>>(outsize[g]);
    });
    r.phase.partition = timer.lap();
    /*
     * 只有两卡都允许对端访问才启用 P2P，否则由各源卡先复制到本地 pinned 缓冲。
     * 两次并行阶段之间的屏障保证所有 D2H 完成后，目标卡再从对端缓冲发起 H2D。
     * extra 记录实际交换路径，避免将不同互联方式的结果混在一起。
     */
    int access[2] = {};
    for (int g = 0; g < 2; ++g)
        CUDA_CHECK(cudaDeviceCanAccessPeer(&access[g], g, 1 - g));
    bool peer = access[0] && access[1];
    if (peer)
        on_two(batch, c, [&](int g) {
            cudaError_t error = cudaDeviceEnablePeerAccess(1 - g, 0);
            if (error != cudaSuccess && error != cudaErrorPeerAccessAlreadyEnabled)
                CUDA_CHECK(error);
            else if (error == cudaErrorPeerAccessAlreadyEnabled)
                cudaGetLastError();
        });
    if (peer) {
        on_two(batch, c, [&](int g) {
            int src = 1 - g;
            const Key *ptr = state[src]->input.data + (src == 0 ? cuts[0] : 0);
            CUDA_CHECK(cudaMemcpyPeerAsync(state[g]->received->data, g, ptr, src,
                                           received[g] * sizeof(Key), state[g]->stream.value));
            CUDA_CHECK(cudaStreamSynchronize(state[g]->stream.value));
        });
    } else {
        std::unique_ptr<PinnedHostBuffer> staging[2];
        on_two(batch, c, [&](int g) {
            staging[g] = std::make_unique<PinnedHostBuffer>(sending[g]);
            auto &s = *state[g];
            CUDA_CHECK(cudaMemcpyAsync(staging[g]->data, s.input.data + (g == 0 ? cuts[0] : 0),
                                       sending[g] * sizeof(Key), cudaMemcpyDeviceToHost,
                                       s.stream.value));
            CUDA_CHECK(cudaStreamSynchronize(s.stream.value));
        });
        on_two(batch, c, [&](int g) {
            auto &s = *state[g];
            CUDA_CHECK(cudaMemcpyAsync(s.received->data, staging[1 - g]->data,
                                       received[g] * sizeof(Key), cudaMemcpyHostToDevice,
                                       s.stream.value));
            CUDA_CHECK(cudaStreamSynchronize(s.stream.value));
        });
    }
    r.phase.exchange = timer.lap();
    double mergems[2] = {};
    on_two(batch, c, [&](int g) {
        auto &s = *state[g];
        EventTimer t(s.stream.value);
        Key *first = s.input.data + (g == 0 ? 0 : cuts[1]);
        uint64_t keep = sizes[g] - received[1 - g];
        thrust::merge(thrust::cuda::par.on(s.stream.value), first, first + keep, s.received->data,
                      s.received->data + received[g], s.output->data);
        mergems[g] = t.lap();
        CUDA_CHECK(cudaMemcpyAsync(batch.outputs[g], s.output->data, outsize[g] * sizeof(Key),
                                   cudaMemcpyDeviceToHost, s.stream.value));
        d2h[g] = t.lap();
    });
    r.phase.merge = std::max(mergems[0], mergems[1]);
    r.phase.d2h = std::max(d2h[0], d2h[1]);
    r.imbalance = imbalance(std::max(outsize[0], outsize[1]), n, 2);
    r.extra =
        std::string("exchange_path=") + (peer ? "peer" : "pinned_host") + ";mixed_phase_clocks=1";
    return r;
}
int main(int argc, char **argv) {
    return run_gpu(argc, argv, "gpu_multi", "sample", 2, multi_sort);
}
