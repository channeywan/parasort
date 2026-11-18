#include "psrs_core.hpp"
#include <pthread.h>
#include <atomic>
#include <thread>
#include <stdexcept>
using namespace parasort;
struct alignas(64) Stats { Key lo=UINT32_MAX,hi=0; };
struct Context {
    Key *a,*out; uint64_t n,p,stride; bool locked;
    struct alignas(64) Counters { uint64_t v[8]={}; };
    std::vector<Stats> stats; std::vector<Counters> counts; std::vector<uint64_t> offsets,dst;
    uint64_t& count(uint64_t t,uint64_t b) { uint64_t i=t*stride+b; return counts[i/8].v[i%8]; }
    std::vector<std::vector<Key>> buckets; std::vector<pthread_mutex_t> locks;
    pthread_barrier_t barrier; std::atomic<int> gate{0}; Key lo=0; uint64_t width=1; SortResult result; Timer timer;
    Context(Key* a_,Key* o,uint64_t n_,uint64_t p_,bool l):a(a_),out(o),n(n_),p(p_),stride((p+7)/8*8),locked(l),stats(p),counts(p*stride/8),offsets(p*stride),dst(p+1),buckets(p),locks(p) {
        if(pthread_barrier_init(&barrier,nullptr,p)) throw std::runtime_error("pthread_barrier_init failed");
        for(auto& lock:locks) pthread_mutex_init(&lock,nullptr);
    }
    ~Context() { for(auto& lock:locks) pthread_mutex_destroy(&lock); pthread_barrier_destroy(&barrier); }
    void sync() { pthread_barrier_wait(&barrier); }
};
struct Arg { Context* c; uint64_t t; };
static void* worker(void* ptr) {
    auto arg=*static_cast<Arg*>(ptr); auto& c=*arg.c; auto t=arg.t;
    while(c.gate.load(std::memory_order_acquire)==0) std::this_thread::yield();
    if(c.gate.load()<0) return nullptr;
    uint64_t begin=cut(c.n,t,c.p),end=cut(c.n,t+1,c.p);
    for(uint64_t i=begin;i<end;++i) { c.stats[t].lo=std::min(c.stats[t].lo,c.a[i]); c.stats[t].hi=std::max(c.stats[t].hi,c.a[i]); }
    c.sync();
    if(t==0) { Key hi=0; c.lo=UINT32_MAX; for(auto& s:c.stats) { c.lo=std::min(c.lo,s.lo); hi=std::max(hi,s.hi); } if(!c.n) c.lo=hi=0; c.width=(uint64_t(hi)-c.lo+1+c.p-1)/c.p; c.result.phase.other=c.timer.lap(); }
    c.sync();
    for(uint64_t i=begin;i<end;++i) {
        auto b=bucket_for(c.a[i],c.lo,c.width,c.p);
        if(c.locked) { pthread_mutex_lock(&c.locks[b]); c.buckets[b].push_back(c.a[i]); pthread_mutex_unlock(&c.locks[b]); }
        else ++c.count(t,b);
    }
    c.sync();
    if(t==0) {
        for(uint64_t b=0;b<c.p;++b) { uint64_t pos=c.dst[b]; for(uint64_t s=0;s<c.p;++s) { c.offsets[s*c.stride+b]=pos; if(!c.locked) pos+=c.count(s,b); } c.dst[b+1]=c.locked ? c.dst[b]+c.buckets[b].size():pos; }
        c.result.phase.partition=c.timer.lap();
    }
    c.sync();
    if(!c.locked) for(uint64_t i=begin;i<end;++i) { auto b=bucket_for(c.a[i],c.lo,c.width,c.p); c.out[c.offsets[t*c.stride+b]++]=c.a[i]; }
    c.sync(); if(t==0) c.result.phase.exchange=c.timer.lap(); c.sync();
    if(c.locked) { std::sort(c.buckets[t].begin(),c.buckets[t].end()); std::copy(c.buckets[t].begin(),c.buckets[t].end(),c.out+c.dst[t]); }
    else std::sort(c.out+c.dst[t],c.out+c.dst[t+1]);
    c.sync(); if(t==0) c.result.phase.local_sort=c.timer.ms(); return nullptr;
}
static SortResult bucket(Key* a,Key* out,uint64_t n,const Config& config) {
    require_variant(config,{"locked","local"}); Context c(a,out,n,config.threads,config.variant=="locked");
    std::vector<pthread_t> ids(c.p); std::vector<Arg> args(c.p); uint64_t created=0;
    for(;created<c.p;++created) { args[created]={&c,created}; if(pthread_create(&ids[created],nullptr,worker,&args[created])) break; }
    c.gate.store(created==c.p ? 1:-1,std::memory_order_release);
    for(uint64_t i=0;i<created;++i) pthread_join(ids[i],nullptr);
    if(created!=c.p) throw std::runtime_error("pthread_create failed: reduce --threads");
    uint64_t largest=0; for(uint64_t b=0;b<c.p;++b) largest=std::max(largest,c.dst[b+1]-c.dst[b]);
    c.result.data=out; c.result.imbalance=imbalance(largest,n,c.p); return c.result;
}
int main(int argc,char** argv) { return run_cpu(argc,argv,"pthreads","local",bucket); }
