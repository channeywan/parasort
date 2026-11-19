#include "gpu_support.cuh"
using namespace parasort;
using Count=unsigned long long;
constexpr int BLOCK=256,TILE=512,HASH=1024;
__device__ unsigned bucket_id(Key key,const Key* splits,unsigned B) { unsigned lo=0,hi=B-1; while(lo<hi) { unsigned m=lo+(hi-lo)/2; if(key<=splits[m]) hi=m; else lo=m+1; } return lo; }
__global__ void sample_kernel(const Key* a,Key* samples,uint64_t n,uint64_t S) { for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<S;i+=uint64_t(gridDim.x)*blockDim.x) samples[i]=a[(i*n)/S]; }
__global__ void splitter_kernel(const Key* samples,Key* splits,uint64_t S,unsigned B) { for(unsigned j=blockIdx.x*blockDim.x+threadIdx.x;j<B-1;j+=gridDim.x*blockDim.x) splits[j]=samples[(uint64_t(j+1)*S)/B]; }
__device__ unsigned hash_insert(unsigned key,unsigned* keys) {
    unsigned slot=(key*2654435761U)&(HASH-1);
    while(true) { unsigned old=atomicCAS(keys+slot,UINT32_MAX,key); if(old==UINT32_MAX || old==key) return slot; slot=(slot+1)&(HASH-1); }
}
__global__ void count_kernel(const Key* a,uint64_t n,const Key* splits,unsigned B,Count* counts,bool naive) {
    __shared__ unsigned keys[HASH],counts_local[HASH];
    for(uint64_t base=uint64_t(blockIdx.x)*TILE;base<n;base+=uint64_t(gridDim.x)*TILE) {
        if(!naive) { for(int j=threadIdx.x;j<HASH;j+=BLOCK) { keys[j]=UINT32_MAX; counts_local[j]=0; } }
        __syncthreads();
        for(int k=0;k<TILE/BLOCK;++k) { uint64_t i=base+threadIdx.x+k*BLOCK; if(i<n) { unsigned b=bucket_id(a[i],splits,B); if(naive) atomicAdd(counts+b,Count(1)); else { unsigned slot=hash_insert(b,keys); atomicAdd(counts_local+slot,1U); } } }
        __syncthreads();
        if(!naive) for(int j=threadIdx.x;j<HASH;j+=BLOCK) if(keys[j]!=UINT32_MAX) atomicAdd(counts+keys[j],Count(counts_local[j]));
        __syncthreads();
    }
}
__global__ void scatter_kernel(const Key* a,Key* out,uint64_t n,const Key* splits,unsigned B,Count* cursor,bool naive) {
    __shared__ unsigned keys[HASH],counts_local[HASH]; __shared__ Count bases[HASH];
    for(uint64_t base=uint64_t(blockIdx.x)*TILE;base<n;base+=uint64_t(gridDim.x)*TILE) {
        if(!naive) for(int j=threadIdx.x;j<HASH;j+=BLOCK) { keys[j]=UINT32_MAX; counts_local[j]=0; }
        __syncthreads(); unsigned slots[TILE/BLOCK],ranks[TILE/BLOCK];
        for(int k=0;k<TILE/BLOCK;++k) { uint64_t i=base+threadIdx.x+k*BLOCK; if(i<n) { unsigned b=bucket_id(a[i],splits,B); if(naive) out[atomicAdd(cursor+b,Count(1))]=a[i]; else { unsigned slot=hash_insert(b,keys); slots[k]=slot; ranks[k]=atomicAdd(counts_local+slot,1U); } } }
        __syncthreads();
        if(!naive) { for(int j=threadIdx.x;j<HASH;j+=BLOCK) if(keys[j]!=UINT32_MAX) bases[j]=atomicAdd(cursor+keys[j],Count(counts_local[j])); }
        __syncthreads();
        if(!naive) for(int k=0;k<TILE/BLOCK;++k) { uint64_t i=base+threadIdx.x+k*BLOCK; if(i<n) out[bases[slots[k]]+ranks[k]]=a[i]; }
        __syncthreads();
    }
}
__global__ void bitonic_buckets(Key* a,const Count* offsets,unsigned B) {
    extern __shared__ Key shared[]; unsigned b=blockIdx.x; if(b>=B) return;
    Count begin=offsets[b],size=offsets[b+1]-begin; if(size==0 || size>8192) return;
    unsigned padded=1; while(padded<size) padded<<=1;
    for(unsigned i=threadIdx.x;i<padded;i+=blockDim.x) shared[i]=i<size ? a[begin+i]:UINT32_MAX;
    __syncthreads();
    for(unsigned k=2;k<=padded;k<<=1) for(unsigned j=k>>1;j;j>>=1) {
        for(unsigned i=threadIdx.x;i<padded;i+=blockDim.x) { unsigned other=i^j; if(other>i) { bool ascending=(i&k)==0; Key x=shared[i],y=shared[other]; if((x>y)==ascending) { shared[i]=y; shared[other]=x; } } }
        __syncthreads();
    }
    for(unsigned i=threadIdx.x;i<size;i+=blockDim.x) a[begin+i]=shared[i];
}
static SortResult gpu_sort(Key* host,Key* output,uint64_t n,const Config& c) {
    require_variant(c,{"thrust","sample"}); SortResult r; r.data=output;
    if(!n) return r;
    CUDA_CHECK(cudaSetDevice(0)); Stream stream; DeviceBuffer<Key> input(n); EventTimer t(stream.value);
    CUDA_CHECK(cudaMemcpyAsync(input.data,host,n*sizeof(Key),cudaMemcpyHostToDevice,stream.value)); r.phase.h2d=t.lap(); Key* sorted=input.data;
    if(c.variant=="thrust") { thrust::sort(thrust::cuda::par.on(stream.value),input.data,input.data+n); r.phase.local_sort=t.lap(); }
    else {
        unsigned B=1; while(uint64_t(B)<(n+4095)/4096) { if(B>=1U<<29) throw std::length_error("too many GPU buckets"); B<<=1; }
        uint64_t S=std::min(n,uint64_t(32)*B); unsigned grid=std::min<uint64_t>(4096,(n+TILE-1)/TILE);
        DeviceBuffer<Key> samples(S),splits(B-1),out(n); DeviceBuffer<Count> counts(B),offsets(B+1),cursor(B);
        sample_kernel<<<std::min<uint64_t>(4096,(S+BLOCK-1)/BLOCK),BLOCK,0,stream.value>>>(input.data,samples.data,n,S); CUDA_CHECK(cudaGetLastError());
        thrust::sort(thrust::cuda::par.on(stream.value),samples.data,samples.data+S); r.phase.sampling=t.lap();
        if(B>1) { splitter_kernel<<<(B+BLOCK-2)/BLOCK,BLOCK,0,stream.value>>>(samples.data,splits.data,S,B); CUDA_CHECK(cudaGetLastError()); } r.phase.splitters=t.lap();
        CUDA_CHECK(cudaMemsetAsync(counts.data,0,B*sizeof(Count),stream.value));
        count_kernel<<<grid,BLOCK,0,stream.value>>>(input.data,n,splits.data,B,counts.data,c.naive_atomics); CUDA_CHECK(cudaGetLastError());
        thrust::exclusive_scan(thrust::cuda::par.on(stream.value),counts.data,counts.data+B,offsets.data);
        Count total=n; CUDA_CHECK(cudaMemcpyAsync(offsets.data+B,&total,sizeof(total),cudaMemcpyHostToDevice,stream.value)); r.phase.partition=t.lap();
        CUDA_CHECK(cudaMemcpyAsync(cursor.data,offsets.data,B*sizeof(Count),cudaMemcpyDeviceToDevice,stream.value));
        scatter_kernel<<<grid,BLOCK,0,stream.value>>>(input.data,out.data,n,splits.data,B,cursor.data,c.naive_atomics); CUDA_CHECK(cudaGetLastError()); r.phase.exchange=t.lap();
        std::vector<Count> sizes(B),positions(B+1);
        CUDA_CHECK(cudaMemcpyAsync(sizes.data(),counts.data,B*sizeof(Count),cudaMemcpyDeviceToHost,stream.value));
        CUDA_CHECK(cudaMemcpyAsync(positions.data(),offsets.data,(B+1)*sizeof(Count),cudaMemcpyDeviceToHost,stream.value)); CUDA_CHECK(cudaStreamSynchronize(stream.value));
        bitonic_buckets<<<B,BLOCK,8192*sizeof(Key),stream.value>>>(out.data,offsets.data,B); CUDA_CHECK(cudaGetLastError());
        unsigned large=0; for(unsigned b=0;b<B;++b) if(sizes[b]>8192) { ++large; thrust::sort(thrust::cuda::par.on(stream.value),out.data+positions[b],out.data+positions[b+1]); }
        r.phase.merge=t.lap(); r.imbalance=imbalance(*std::max_element(sizes.begin(),sizes.end()),n,B);
        r.extra="buckets="+std::to_string(B)+";large_buckets="+std::to_string(large)+";naive_atomics="+std::to_string(c.naive_atomics)+";histogram=sparse_shared_hash";
        CUDA_CHECK(cudaMemcpyAsync(output,out.data,n*sizeof(Key),cudaMemcpyDeviceToHost,stream.value)); r.phase.d2h=t.lap(); return r;
    }
    CUDA_CHECK(cudaMemcpyAsync(output,sorted,n*sizeof(Key),cudaMemcpyDeviceToHost,stream.value)); r.phase.d2h=t.lap(); return r;
}
int main(int argc,char** argv) {
    return run_gpu(argc,argv,"gpu","thrust",1,[](GpuHostBatch& batch,const Config& c) {
        batch.output_sizes[0]=c.n;
        return gpu_sort(batch.inputs[0],batch.outputs[0],c.n,c);
    });
}
