#include "datagen.hpp"
#include <cmath>
#include <omp.h>
namespace parasort {
uint64_t mix64(uint64_t z) { z+=0x9e3779b97f4a7c15ULL; z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL; return z^(z>>31); }
Key generate_key(const Config& c,uint64_t g) {
    uint64_t h=mix64(c.seed^(g*0x9e3779b97f4a7c15ULL));
    double u=double(h>>11)*0x1.0p-53;
    if(c.dist=="uniform") return Key(h>>32);
    if(c.dist=="normal") {
        double v=double(mix64(h)>>11)*0x1.0p-53;
        double z=std::sqrt(-2.*std::log(std::max(u,0x1.0p-53)))*std::cos(6.283185307179586*v);
        return Key(std::clamp(2147483648.+134217728.*z,0.,4294967295.));
    }
    if(c.dist=="power") return Key(std::pow(u,8)*4294967296.);
    if(c.dist=="hotkeys") return u<.5 ? Key((h%8)*uint64_t(1ULL<<29)):Key(h>>32);
    if(c.dist=="few_unique") return Key((h%16)*uint64_t(1ULL<<28));
    uint64_t index=c.dist=="reverse" ? c.n-1-g:g;
    Key sorted=c.n ? Key((__uint128_t(index)<<32)/c.n):0;
    return c.dist=="nearly_sorted" && h%100==0 ? Key(h>>32):sorted;
}
void generate(Key* a,uint64_t n,uint64_t start,const Config& c,bool serial) {
    #pragma omp parallel num_threads(c.threads) if(!serial)
    {
        int t=omp_get_thread_num(),p=omp_get_num_threads();
        for(uint64_t i=cut(n,t,p);i<cut(n,t+1,p);++i) a[i]=generate_key(c,start+i);
    }
}
void first_touch(Key* a,uint64_t n,int threads,bool serial) {
    #pragma omp parallel num_threads(threads) if(!serial)
    {
        int t=omp_get_thread_num(),p=omp_get_num_threads();
        for(uint64_t i=cut(n,t,p);i<cut(n,t+1,p);++i) a[i]=0;
    }
}
}
