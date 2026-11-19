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
#include <cctype>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <stdexcept>
namespace parasort {
inline void cuda_check(cudaError_t code,const char* expr,const char* file,int line) { if(code!=cudaSuccess) throw std::runtime_error(std::string(file)+":"+std::to_string(line)+" "+expr+": "+cudaGetErrorString(code)); }
#define CUDA_CHECK(x) ::parasort::cuda_check((x),#x,__FILE__,__LINE__)

struct GpuNumaInfo {
    int node=-1;
    std::string pci,reason;
    cpu_set_t cpus{};
};

inline bool parse_gpu_cpulist(const std::string& text,cpu_set_t& cpus) {
    CPU_ZERO(&cpus);
    std::istringstream input(text);
    std::string token;
    try {
        while(std::getline(input,token,',')) {
            size_t dash=token.find('-'),used=0;
            int first=std::stoi(token.substr(0,dash),&used);
            if(used!=(dash==std::string::npos ? token.size():dash)) return false;
            int last=first;
            if(dash!=std::string::npos) {
                std::string end=token.substr(dash+1);
                last=std::stoi(end,&used);
                if(used!=end.size()) return false;
            }
            if(first<0 || last<first || last>=CPU_SETSIZE) return false;
            for(int cpu=first;cpu<=last;++cpu) CPU_SET(cpu,&cpus);
        }
    } catch(const std::exception&) { return false; }
    return CPU_COUNT(&cpus)>0;
}

inline std::string gpu_cpu_list(const cpu_set_t& cpus) {
    std::ostringstream output;
    bool first=true;
    for(int cpu=0;cpu<CPU_SETSIZE;++cpu) {
        if(!CPU_ISSET(cpu,&cpus)) continue;
        int begin=cpu;
        while(cpu+1<CPU_SETSIZE && CPU_ISSET(cpu+1,&cpus)) ++cpu;
        if(!first) output<<',';
        first=false;
        output<<begin;
        if(cpu!=begin) output<<'-'<<cpu;
    }
    return first ? "empty":output.str();
}

inline GpuNumaInfo gpu_numa_info(int dev) {
    // 拓扑信息在同一进程中保持不变；计时阶段只使用缓存的 CPU 集合。
    static std::mutex mutex;
    static std::map<int,GpuNumaInfo> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto found=cache.find(dev);
    if(found!=cache.end()) return found->second;
    GpuNumaInfo info;
    char address[64]={};
    cudaError_t error=cudaDeviceGetPCIBusId(address,sizeof(address),dev);
    if(error!=cudaSuccess) {
        info.reason="pci_query_"+std::string(cudaGetErrorName(error));
    } else {
        info.pci=address;
        for(char& ch:info.pci) ch=char(std::tolower(static_cast<unsigned char>(ch)));
        unsigned domain=0,bus=0,slot=0,function=0;
        if(std::sscanf(info.pci.c_str(),"%x:%x:%x.%x",&domain,&bus,&slot,&function)==4) {
            std::snprintf(address,sizeof(address),"%04x:%02x:%02x.%x",domain,bus,slot,function);
            info.pci=address;
        }
        std::ifstream node_file("/sys/bus/pci/devices/"+info.pci+"/numa_node");
        if(!(node_file>>info.node)) info.reason="numa_node_unreadable";
        else if(info.node<0) info.reason="numa_node_unknown";
        else {
            std::ifstream cpus_file("/sys/devices/system/node/node"+std::to_string(info.node)+"/cpulist");
            std::string cpulist;
            if(!(cpus_file>>cpulist)) info.reason="cpulist_unreadable";
            else if(!parse_gpu_cpulist(cpulist,info.cpus)) info.reason="cpulist_invalid_or_unsupported";
        }
    }
    cache.emplace(dev,info);
    return info;
}

struct GpuAffinityResult {
    int node=-1;
    std::string pci,actual="none(uninitialized)",cpus="unknown";
};

class GpuAffinityGuard {
    cpu_set_t previous_{};
    bool restore_=false;
    GpuAffinityResult result_;
public:
    explicit GpuAffinityGuard(int dev,const std::string& requested="numa") {
        auto info=gpu_numa_info(dev);
        result_.node=info.node;
        result_.pci=info.pci;
        int error=pthread_getaffinity_np(pthread_self(),sizeof(previous_),&previous_);
        if(error) {
            result_.actual="none(getaffinity_"+std::to_string(error)+")";
            return;
        }
        result_.cpus=gpu_cpu_list(previous_);
        if(requested=="none") {
            result_.actual="none(disabled)";
            return;
        }
        if(!info.reason.empty()) {
            result_.actual="none("+info.reason+")";
            return;
        }
        // 当前掩码可能已经被 libgomp 限制到 CPU 0，不能用它与目标节点求交。
        // 内核负责遵守 cgroup 的有效 CPU 限制；记录它实际接受的 CPU 集合。
        error=pthread_setaffinity_np(pthread_self(),sizeof(info.cpus),&info.cpus);
        if(error) {
            result_.actual="none(setaffinity_"+std::to_string(error)+")";
            return;
        }
        restore_=true;
        cpu_set_t actual{};
        error=pthread_getaffinity_np(pthread_self(),sizeof(actual),&actual);
        if(error) {
            result_.actual="none(getaffinity_after_"+std::to_string(error)+")";
            int restore_error=pthread_setaffinity_np(pthread_self(),sizeof(previous_),&previous_);
            restore_=restore_error!=0;
            return;
        }
        result_.actual="numa";
        result_.cpus=gpu_cpu_list(actual);
    }
    ~GpuAffinityGuard() {
        if(restore_) {
            int error=pthread_setaffinity_np(pthread_self(),sizeof(previous_),&previous_);
            if(error) std::fprintf(stderr,"GPU affinity restore failed: %s\n",std::strerror(error));
        }
    }
    GpuAffinityGuard(const GpuAffinityGuard&)=delete;
    GpuAffinityGuard& operator=(const GpuAffinityGuard&)=delete;
    const GpuAffinityResult& result() const { return result_; }
};

inline GpuAffinityGuard bind_to_gpu_numa(int dev,const std::string& requested="numa") {
    return GpuAffinityGuard(dev,requested);
}

struct PinnedHostBuffer {
    Key* data=nullptr;
    uint64_t capacity;
    explicit PinnedHostBuffer(uint64_t n):capacity(n) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data),std::max<uint64_t>(1,n)*sizeof(Key)));
        // 分配和首次写入均由已绑定的所有者线程执行，不另开 OpenMP 团队。
        std::fill_n(data,std::max<uint64_t>(1,n),Key(0));
    }
    ~PinnedHostBuffer() { if(data) cudaFreeHost(data); }
    PinnedHostBuffer(const PinnedHostBuffer&)=delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&)=delete;
};

struct GpuHostBatch {
    int devices;
    std::vector<std::unique_ptr<PinnedHostBuffer>> input_buffers,output_buffers;
    std::vector<Key*> inputs,outputs;
    std::vector<uint64_t> input_sizes,output_sizes;
    std::vector<GpuAffinityResult> affinity;
    explicit GpuHostBatch(int count):devices(count),input_buffers(count),output_buffers(count),
        inputs(count,nullptr),outputs(count,nullptr),input_sizes(count),output_sizes(count),affinity(count) {}
};

template<class F> inline void on_gpu_devices(GpuHostBatch& batch,const Config& c,F operation) {
    std::vector<std::exception_ptr> errors(batch.devices);
    #pragma omp parallel for num_threads(batch.devices) schedule(static) if(batch.devices>1)
    for(int g=0;g<batch.devices;++g) {
        try {
            auto binding=bind_to_gpu_numa(g,c.gpu_affinity);
            batch.affinity[g]=binding.result();
            CUDA_CHECK(cudaSetDevice(g));
            operation(g);
        } catch(...) { errors[g]=std::current_exception(); }
    }
    for(const auto& error:errors) if(error) std::rethrow_exception(error);
}

inline Fingerprint gpu_batch_fingerprint(const GpuHostBatch& batch,bool output) {
    Fingerprint combined;
    for(int g=0;g<batch.devices;++g) {
        auto part=fingerprint(output ? batch.outputs[g]:batch.inputs[g],
                              output ? batch.output_sizes[g]:batch.input_sizes[g],1);
        combined.count+=part.count;
        combined.sum+=part.sum;
        combined.xr^=part.xr;
    }
    return combined;
}

inline bool verify_gpu_batch(const GpuHostBatch& batch,const Fingerprint& before,
                             const std::vector<Key>& exact) {
    uint64_t offset=0;
    bool have_previous=false;
    Key previous=0;
    for(int g=0;g<batch.devices;++g) {
        uint64_t n=batch.output_sizes[g];
        const Key* output=batch.outputs[g];
        if(n>batch.output_buffers[g]->capacity || n>before.count-offset) return false;
        if(!ordered(output,n,1)) return false;
        if(n) {
            if(have_previous && previous>output[0]) return false;
            previous=output[n-1];
            have_previous=true;
        }
        if(!exact.empty() && !std::equal(output,output+n,exact.begin()+offset)) return false;
        offset+=n;
    }
    return offset==before.count && gpu_batch_fingerprint(batch,true)==before;
}

inline std::string gpu_affinity_extra(const GpuHostBatch& batch,const Config& c) {
    std::string actual=batch.affinity.front().actual;
    for(const auto& state:batch.affinity) if(state.actual!=actual) actual="mixed";
    std::string extra="gpu_affinity="+c.gpu_affinity+";gpu_affinity_actual="+actual;
    for(int g=0;g<batch.devices;++g) {
        const auto& state=batch.affinity[g];
        std::string prefix=";gpu"+std::to_string(g);
        extra+=prefix+"_affinity="+state.actual+prefix+"_numa="+std::to_string(state.node);
        extra+=prefix+"_pci="+state.pci+prefix+"_cpus="+state.cpus;
    }
    return extra+";gpu_host_layout=per_device;gpu_first_touch=owner_thread";
}

using GpuSorter=std::function<SortResult(GpuHostBatch&,const Config&)>;

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
inline int run_gpu(int argc,char** argv,const std::string& impl,const std::string& variant,int needed,const GpuSorter& sorter) {
    try {
        Config c=parse_cli(argc,argv,variant); if(c.help) { std::cout<<help_text(); return 0; }
        c.threads=needed; int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices));
        if(devices<needed) throw std::runtime_error("requires "+std::to_string(needed)+" usable CUDA GPU(s); check driver and CUDA_VISIBLE_DEVICES");
        GpuHostBatch batch(needed);
        // 每个设备的输出可能在分区后增长至 n；预留容量保持主机分配在计时之外。
        on_gpu_devices(batch,c,[&](int g) {
            batch.input_sizes[g]=cut(c.n,g+1,needed)-cut(c.n,g,needed);
            batch.input_buffers[g]=std::make_unique<PinnedHostBuffer>(batch.input_sizes[g]);
            batch.output_buffers[g]=std::make_unique<PinnedHostBuffer>(c.n);
            batch.inputs[g]=batch.input_buffers[g]->data;
            batch.outputs[g]=batch.output_buffers[g]->data;
            batch.output_sizes[g]=batch.input_sizes[g];
        });
        for(int g=0;g<needed;++g) {
            const auto& state=batch.affinity[g];
            std::cerr<<"GPU affinity: device="<<g<<" requested="<<c.gpu_affinity
                     <<" actual="<<state.actual<<" pci="<<state.pci<<" numa="<<state.node
                     <<" cpus="<<state.cpus<<'\n';
        }
        for(int rep=-1;rep<c.repeat;++rep) {
            on_gpu_devices(batch,c,[&](int g) {
                uint64_t begin=cut(c.n,g,needed);
                for(uint64_t i=0;i<batch.input_sizes[g];++i) batch.inputs[g][i]=generate_key(c,begin+i);
            });
            Fingerprint before;
            std::vector<Key> exact;
            if(c.verify) {
                before=gpu_batch_fingerprint(batch,false);
                if(c.n<=100000) {
                    for(int g=0;g<needed;++g) exact.insert(exact.end(),batch.inputs[g],batch.inputs[g]+batch.input_sizes[g]);
                    std::sort(exact.begin(),exact.end());
                }
            }
            std::unique_ptr<GpuAffinityGuard> single_binding;
            if(needed==1) {
                single_binding=std::make_unique<GpuAffinityGuard>(0,c.gpu_affinity);
                batch.affinity[0]=single_binding->result();
            }
            Timer timer; auto result=sorter(batch,c); double elapsed=timer.ms();
            single_binding.reset();
            int verified=c.verify ? int(verify_gpu_batch(batch,before,exact)):-1;
            if(!verified) throw std::runtime_error("GPU verification FAILED");
            if(!result.extra.empty()) result.extra+=';';
            result.extra+=gpu_affinity_extra(batch,c);
            result.extra+=";compute_ms="+std::to_string(result.phase.local_sort+result.phase.sampling+result.phase.splitters+result.phase.partition+result.phase.exchange+result.phase.merge);
            if(rep>=0) emit_csv(c,impl,result,rep,elapsed,verified,1,needed);
        }
        return 0;
    } catch(const std::exception& e) { std::cerr<<"error: "<<e.what()<<'\n'; return 1; }
}
}
