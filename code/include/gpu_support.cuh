#pragma once
#include "cli.hpp"
#include "datagen.hpp"
#include "verify.hpp"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/merge.h>
#include <iostream>
#include <stdexcept>
namespace parasort {
inline void cuda_check(cudaError_t code,const char* expr,const char* file,int line) { if(code!=cudaSuccess) throw std::runtime_error(std::string(file)+":"+std::to_string(line)+" "+expr+": "+cudaGetErrorString(code)); }
#define CUDA_CHECK(x) ::parasort::cuda_check((x),#x,__FILE__,__LINE__)
template<class T> struct DeviceBuffer {
    T* data=nullptr; size_t count; int device;
    explicit DeviceBuffer(size_t n):count(n) { CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data),std::max<size_t>(n,1)*sizeof(T))); }
    ~DeviceBuffer() { if(data) { cudaSetDevice(device); cudaFree(data); } }
    DeviceBuffer(const DeviceBuffer&)=delete; DeviceBuffer& operator=(const DeviceBuffer&)=delete;
};
struct Stream {
    cudaStream_t value; int device;
    Stream() { CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaStreamCreate(&value)); }
    ~Stream() { cudaSetDevice(device); cudaStreamDestroy(value); }
};
struct EventTimer {
    cudaEvent_t begin,end; cudaStream_t stream;
    explicit EventTimer(cudaStream_t s):stream(s) { CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end)); CUDA_CHECK(cudaEventRecord(begin,stream)); }
    ~EventTimer() { cudaEventDestroy(begin); cudaEventDestroy(end); }
    double lap() { CUDA_CHECK(cudaEventRecord(end,stream)); CUDA_CHECK(cudaEventSynchronize(end)); float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end)); CUDA_CHECK(cudaEventRecord(begin,stream)); return ms; }
};
inline int run_gpu(int argc,char** argv,const std::string& impl,const std::string& variant,int needed,const Sorter& sorter) {
    Key *a=nullptr,*out=nullptr;
    try {
        Config c=parse_cli(argc,argv,variant); if(c.help) { std::cout<<help_text(); return 0; }
        c.threads=needed; int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices));
        if(devices<needed) throw std::runtime_error("requires "+std::to_string(needed)+" usable CUDA GPU(s); check driver and CUDA_VISIBLE_DEVICES");
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&a),std::max<uint64_t>(1,c.n)*sizeof(Key)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&out),std::max<uint64_t>(1,c.n)*sizeof(Key)));
        first_touch(out,c.n,c.threads);
        for(int rep=-1;rep<c.repeat;++rep) {
            generate(a,c.n,0,c); Fingerprint before; std::vector<Key> exact;
            if(c.verify) { before=fingerprint(a,c.n,c.threads); if(c.n<=100000) { exact.assign(a,a+c.n); std::sort(exact.begin(),exact.end()); } }
            Timer timer; auto result=sorter(a,out,c.n,c); double elapsed=timer.ms();
            int verified=c.verify ? int(verify_result(result.data,c.n,before,exact,c.threads)):-1;
            if(!verified) throw std::runtime_error("GPU verification FAILED");
            result.extra+=";compute_ms="+std::to_string(result.phase.local_sort+result.phase.sampling+result.phase.splitters+result.phase.partition+result.phase.exchange+result.phase.merge);
            if(rep>=0) emit_csv(c,impl,result,rep,elapsed,verified,1,needed);
        }
        CUDA_CHECK(cudaFreeHost(a)); a=nullptr; CUDA_CHECK(cudaFreeHost(out)); return 0;
    } catch(const std::exception& e) { if(a) cudaFreeHost(a); if(out) cudaFreeHost(out); std::cerr<<"error: "<<e.what()<<'\n'; return 1; }
}
}
