#include "verify.hpp"
#include "datagen.hpp"
namespace parasort {
Fingerprint fingerprint(const Key* a,uint64_t n,int threads) {
    uint64_t sum=0,xr=0;
    #pragma omp parallel for num_threads(threads) reduction(+:sum) reduction(^:xr) schedule(static)
    for(uint64_t i=0;i<n;++i) { uint64_t h=mix64(a[i]); sum+=h; xr^=h; }
    return {n,sum,xr};
}
bool ordered(const Key* a,uint64_t n,int threads) {
    int bad=0;
    #pragma omp parallel for num_threads(threads) reduction(|:bad) schedule(static)
    for(uint64_t i=1;i<n;++i) bad|=a[i-1]>a[i];
    return !bad;
}
bool verify_result(const Key* a,uint64_t n,const Fingerprint& before,const std::vector<Key>& exact,int threads) {
    return ordered(a,n,threads) && fingerprint(a,n,threads)==before && (exact.empty() || std::equal(exact.begin(),exact.end(),a));
}
}
