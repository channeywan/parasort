#pragma once
#include "common.hpp"
namespace parasort {
uint64_t mix64(uint64_t);
Key generate_key(const Config&,uint64_t global_index);
void generate(Key*,uint64_t count,uint64_t global_start,const Config&,bool serial=false);
void first_touch(Key*,uint64_t,int threads,bool serial=false);
}
