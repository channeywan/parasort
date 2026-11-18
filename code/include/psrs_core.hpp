#pragma once
#include "common.hpp"
namespace parasort {
struct Sample { uint64_t key=0, block=0, pos=0; };
struct Run { const Key* data; uint64_t size; };
bool sample_less(const Sample&,const Sample&);
std::vector<Sample> regular_samples(const Key*,uint64_t,uint64_t block,uint64_t count);
std::vector<Sample> select_splitters(std::vector<Sample>,uint64_t buckets);
uint64_t split_position(const Key*,uint64_t,uint64_t block,const Sample&,bool tiebreak);
void merge_runs(const std::vector<Run>&,Key*);
void parallel_merge_runs(const std::vector<Run>&,Key*,const Config&);
SortResult psrs_openmp(Key*,Key*,uint64_t,const Config&);
inline uint64_t bucket_for(Key x,Key lo,uint64_t width,uint64_t p) { return std::min(p-1,(uint64_t(x)-lo)/width); }
}
