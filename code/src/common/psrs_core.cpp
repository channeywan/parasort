#include "psrs_core.hpp"
#include <queue>
#include <tuple>
#include <omp.h>
namespace parasort {
bool sample_less(const Sample& a,const Sample& b) { return std::tie(a.key,a.block,a.pos)<std::tie(b.key,b.block,b.pos); }
std::vector<Sample> regular_samples(const Key* a,uint64_t n,uint64_t block,uint64_t count) {
    count=std::min(count,n); std::vector<Sample> out; out.reserve(count);
    for(uint64_t s=0;s<count;++s) { uint64_t pos=cut(n,s,count); out.push_back({a[pos],block,pos}); }
    return out;
}
std::vector<Sample> select_splitters(std::vector<Sample> s,uint64_t buckets) {
    std::sort(s.begin(),s.end(),sample_less); std::vector<Sample> result;
    for(uint64_t j=1;j<buckets;++j) result.push_back(s.empty() ? Sample{UINT32_MAX,UINT64_MAX,UINT64_MAX}:s[cut(s.size(),j,buckets)]);
    return result;
}
uint64_t split_position(const Key* a,uint64_t n,uint64_t block,const Sample& s,bool tie) {
    if(!tie || block<s.block) return std::upper_bound(a,a+n,Key(s.key))-a;
    if(block>s.block) return std::lower_bound(a,a+n,Key(s.key))-a;
    uint64_t lo=std::lower_bound(a,a+n,Key(s.key))-a,hi=std::upper_bound(a,a+n,Key(s.key))-a;
    // Inclusive upper bound on (key, block, sorted position).
    return std::clamp(s.pos==UINT64_MAX ? n:s.pos+1,lo,hi);
}
void merge_runs(const std::vector<Run>& runs,Key* out) {
    struct Node { Key value; uint64_t run,pos; };
    auto greater=[](const Node& a,const Node& b) { return std::tie(a.value,a.run)>std::tie(b.value,b.run); };
    std::vector<Node> storage; storage.reserve(runs.size());
    std::priority_queue<Node,std::vector<Node>,decltype(greater)> heap(greater,std::move(storage));
    for(uint64_t i=0;i<runs.size();++i) if(runs[i].size) heap.push({runs[i].data[0],i,0});
    uint64_t dst=0;
    while(!heap.empty()) { Node x=heap.top(); heap.pop(); out[dst++]=x.value; if(++x.pos<runs[x.run].size) { x.value=runs[x.run].data[x.pos]; heap.push(x); } }
}
void parallel_merge_runs(const std::vector<Run>& runs,Key* out,const Config& c) {
    if(c.threads==1) { merge_runs(runs,out); return; }
    uint64_t B=c.threads,R=runs.size(); std::vector<Sample> samples;
    for(uint64_t i=0;i<R;++i) { auto s=regular_samples(runs[i].data,runs[i].size,i,B*c.oversample); samples.insert(samples.end(),s.begin(),s.end()); }
    auto splitters=select_splitters(std::move(samples),B);
    std::vector<uint64_t> bounds(R*(B+1)),dst(B+1);
    for(uint64_t i=0;i<R;++i) { bounds[i*(B+1)+B]=runs[i].size; for(uint64_t j=1;j<B;++j) bounds[i*(B+1)+j]=split_position(runs[i].data,runs[i].size,i,splitters[j-1],true); }
    for(uint64_t j=0;j<B;++j) { dst[j+1]=dst[j]; for(uint64_t i=0;i<R;++i) dst[j+1]+=bounds[i*(B+1)+j+1]-bounds[i*(B+1)+j]; }
    #pragma omp parallel for num_threads(c.threads) schedule(static)
    for(uint64_t j=0;j<B;++j) { std::vector<Run> pieces; pieces.reserve(R); for(uint64_t i=0;i<R;++i) { auto b=bounds[i*(B+1)+j]; pieces.push_back({runs[i].data+b,bounds[i*(B+1)+j+1]-b}); } merge_runs(pieces,out+dst[j]); }
}
SortResult psrs_openmp(Key* a,Key* out,uint64_t n,const Config& c) {
    uint64_t p=c.threads,B=p*c.buckets_per_thread; SortResult r; r.data=out;
    std::vector<std::vector<Sample>> per_block(p); std::vector<uint64_t> bounds(p*(B+1)),dst(B+1); Timer t;
    #pragma omp parallel for num_threads(c.threads) schedule(static)
    for(uint64_t i=0;i<p;++i) std::sort(a+cut(n,i,p),a+cut(n,i+1,p));
    r.phase.local_sort=t.lap();
    #pragma omp parallel for num_threads(c.threads) schedule(static)
    for(uint64_t i=0;i<p;++i) { auto begin=cut(n,i,p); per_block[i]=regular_samples(a+begin,cut(n,i+1,p)-begin,i,uint64_t(c.oversample)*B); }
    r.phase.sampling=t.lap(); std::vector<Sample> samples;
    for(auto& v:per_block) samples.insert(samples.end(),v.begin(),v.end());
    auto splitters=select_splitters(std::move(samples),B); r.phase.splitters=t.lap();
    #pragma omp parallel for num_threads(c.threads) schedule(static)
    for(uint64_t i=0;i<p;++i) { uint64_t begin=cut(n,i,p),m=cut(n,i+1,p)-begin; bounds[i*(B+1)+B]=m; for(uint64_t j=1;j<B;++j) bounds[i*(B+1)+j]=split_position(a+begin,m,i,splitters[j-1],c.tiebreak); }
    r.phase.partition=t.lap(); uint64_t largest=0;
    for(uint64_t j=0;j<B;++j) { uint64_t size=0; for(uint64_t i=0;i<p;++i) size+=bounds[i*(B+1)+j+1]-bounds[i*(B+1)+j]; dst[j+1]=dst[j]+size; largest=std::max(largest,size); }
    r.phase.exchange=t.lap();
    omp_sched_t sched=c.schedule=="dynamic" ? omp_sched_dynamic:c.schedule=="guided" ? omp_sched_guided:omp_sched_static;
    omp_set_schedule(sched,1);
    #pragma omp parallel for num_threads(c.threads) schedule(runtime)
    for(uint64_t j=0;j<B;++j) { std::vector<Run> pieces; pieces.reserve(p); for(uint64_t i=0;i<p;++i) { auto b=bounds[i*(B+1)+j]; pieces.push_back({a+cut(n,i,p)+b,bounds[i*(B+1)+j+1]-b}); } merge_runs(pieces,out+dst[j]); }
    r.phase.merge=t.ms(); r.imbalance=imbalance(largest,n,B); r.extra="buckets="+std::to_string(B); return r;
}
}
