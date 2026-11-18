#include "gpu_support.cuh"
#include "psrs_core.hpp"
#include <exception>
#include <omp.h>
using namespace parasort;
struct GpuState {
    Stream stream; DeviceBuffer<Key> input;
    std::unique_ptr<DeviceBuffer<Key>> received,output;
    explicit GpuState(uint64_t n):input(n) {}
};
template<class F> static void on_two(F operation) {
    std::exception_ptr errors[2];
    #pragma omp parallel for num_threads(2) schedule(static)
    for(int g=0;g<2;++g) { try { CUDA_CHECK(cudaSetDevice(g)); operation(g); } catch(...) { errors[g]=std::current_exception(); } }
    for(auto& error:errors) if(error) std::rethrow_exception(error);
}
struct PinnedTransfer {
    Key* data=nullptr;
    explicit PinnedTransfer(uint64_t n) { CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data),std::max<uint64_t>(1,n)*sizeof(Key))); }
    ~PinnedTransfer() { if(data) cudaFreeHost(data); }
};
static SortResult multi_sort(Key* host,Key* output,uint64_t n,const Config& c) {
    require_variant(c,{"sample"}); SortResult r; r.data=output; if(!n) return r;
    uint64_t sizes[2]={n/2,n-n/2},begin[2]={0,n/2},cuts[2]={};
    std::unique_ptr<GpuState> state[2]; double h2d[2]={},sortms[2]={},d2h[2]={};
    on_two([&](int g) {
        state[g]=std::make_unique<GpuState>(sizes[g]); auto& s=*state[g]; EventTimer t(s.stream.value);
        CUDA_CHECK(cudaMemcpyAsync(s.input.data,host+begin[g],sizes[g]*sizeof(Key),cudaMemcpyHostToDevice,s.stream.value)); h2d[g]=t.lap();
        thrust::sort(thrust::cuda::par.on(s.stream.value),s.input.data,s.input.data+sizes[g]); sortms[g]=t.lap();
    });
    r.phase.h2d=std::max(h2d[0],h2d[1]); r.phase.local_sort=std::max(sortms[0],sortms[1]); Timer timer;
    std::vector<Sample> per_gpu[2];
    on_two([&](int g) {
        uint64_t count=std::min<uint64_t>(sizes[g],2*c.oversample); auto& s=*state[g]; per_gpu[g].resize(count);
        for(uint64_t i=0;i<count;++i) { uint64_t pos=cut(sizes[g],i,count); Key key; CUDA_CHECK(cudaMemcpyAsync(&key,s.input.data+pos,sizeof(Key),cudaMemcpyDeviceToHost,s.stream.value)); CUDA_CHECK(cudaStreamSynchronize(s.stream.value)); per_gpu[g][i]={key,uint64_t(g),pos}; }
    });
    r.phase.sampling=timer.lap(); std::vector<Sample> samples=per_gpu[0]; samples.insert(samples.end(),per_gpu[1].begin(),per_gpu[1].end());
    Sample splitter=select_splitters(std::move(samples),2)[0]; r.phase.splitters=timer.lap();
    on_two([&](int g) {
        auto& s=*state[g]; auto policy=thrust::cuda::par.on(s.stream.value);
        uint64_t lo=thrust::lower_bound(policy,s.input.data,s.input.data+sizes[g],Key(splitter.key))-s.input.data;
        uint64_t hi=thrust::upper_bound(policy,s.input.data,s.input.data+sizes[g],Key(splitter.key))-s.input.data;
        cuts[g]=!c.tiebreak || uint64_t(g)<splitter.block ? hi:uint64_t(g)>splitter.block ? lo:std::clamp(splitter.pos+1,lo,hi);
    });
    uint64_t sending[2]={sizes[0]-cuts[0],cuts[1]},received[2]={sending[1],sending[0]},outsize[2]={cuts[0]+cuts[1],n-cuts[0]-cuts[1]};
    on_two([&](int g) { state[g]->received=std::make_unique<DeviceBuffer<Key>>(received[g]); state[g]->output=std::make_unique<DeviceBuffer<Key>>(outsize[g]); });
    r.phase.partition=timer.lap(); int access[2]={};
    for(int g=0;g<2;++g) CUDA_CHECK(cudaDeviceCanAccessPeer(&access[g],g,1-g));
    bool peer=access[0] && access[1];
    if(peer) on_two([&](int g) { cudaError_t error=cudaDeviceEnablePeerAccess(1-g,0); if(error!=cudaSuccess && error!=cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(error); else if(error==cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); });
    if(peer) {
        on_two([&](int g) { int src=1-g; const Key* ptr=state[src]->input.data+(src==0 ? cuts[0]:0); CUDA_CHECK(cudaMemcpyPeerAsync(state[g]->received->data,g,ptr,src,received[g]*sizeof(Key),state[g]->stream.value)); CUDA_CHECK(cudaStreamSynchronize(state[g]->stream.value)); });
    } else {
        std::unique_ptr<PinnedTransfer> staging[2]; for(int g=0;g<2;++g) staging[g]=std::make_unique<PinnedTransfer>(sending[g]);
        on_two([&](int g) { auto& s=*state[g]; CUDA_CHECK(cudaMemcpyAsync(staging[g]->data,s.input.data+(g==0 ? cuts[0]:0),sending[g]*sizeof(Key),cudaMemcpyDeviceToHost,s.stream.value)); CUDA_CHECK(cudaStreamSynchronize(s.stream.value)); });
        on_two([&](int g) { auto& s=*state[g]; CUDA_CHECK(cudaMemcpyAsync(s.received->data,staging[1-g]->data,received[g]*sizeof(Key),cudaMemcpyHostToDevice,s.stream.value)); CUDA_CHECK(cudaStreamSynchronize(s.stream.value)); });
    }
    r.phase.exchange=timer.lap(); double mergems[2]={};
    on_two([&](int g) {
        auto& s=*state[g]; EventTimer t(s.stream.value); Key* first=s.input.data+(g==0 ? 0:cuts[1]); uint64_t keep=sizes[g]-received[1-g];
        thrust::merge(thrust::cuda::par.on(s.stream.value),first,first+keep,s.received->data,s.received->data+received[g],s.output->data); mergems[g]=t.lap();
        CUDA_CHECK(cudaMemcpyAsync(output+(g==0 ? 0:outsize[0]),s.output->data,outsize[g]*sizeof(Key),cudaMemcpyDeviceToHost,s.stream.value)); d2h[g]=t.lap();
    });
    r.phase.merge=std::max(mergems[0],mergems[1]); r.phase.d2h=std::max(d2h[0],d2h[1]); r.imbalance=imbalance(std::max(outsize[0],outsize[1]),n,2);
    r.extra=std::string("exchange_path=")+(peer ? "peer":"pinned_host")+";mixed_phase_clocks=1"; return r;
}
int main(int argc,char** argv) { return run_gpu(argc,argv,"gpu_multi","sample",2,multi_sort); }
