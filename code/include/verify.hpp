#pragma once
#include "common.hpp"
namespace parasort {
struct Fingerprint {
    uint64_t count=0,sum=0,xr=0;
    bool operator==(const Fingerprint& o) const { return count==o.count && sum==o.sum && xr==o.xr; }
};
Fingerprint fingerprint(const Key*,uint64_t,int threads=1);
bool ordered(const Key*,uint64_t,int threads=1);
bool verify_result(const Key*,uint64_t,const Fingerprint&,const std::vector<Key>& exact,int threads);
}
