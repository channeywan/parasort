#pragma once
#include "cli.hpp"
#include "datagen.hpp"
#include "verify.hpp"
#include "psrs_core.hpp"
#include <mpi.h>
#include <climits>
#include <iostream>
#include <stdexcept>
namespace parasort {
inline void mpi_check(int code) { if(code!=MPI_SUCCESS) { char s[MPI_MAX_ERROR_STRING]; int len; MPI_Error_string(code,s,&len); throw std::runtime_error(std::string(s,len)); } }
inline int mpi_int(uint64_t n) { if(n>=uint64_t(INT_MAX)) throw std::length_error("MPI int count/displacement limit exceeded; use fewer elements per rank"); return int(n); }
inline Fingerprint global_fingerprint(const Key* a,uint64_t n,int threads) {
    auto local=fingerprint(a,n,threads); Fingerprint global;
    uint64_t in[2]={local.count,local.sum},out[2]; mpi_check(MPI_Allreduce(in,out,2,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD));
    global.count=out[0]; global.sum=out[1]; mpi_check(MPI_Allreduce(&local.xr,&global.xr,1,MPI_UINT64_T,MPI_BXOR,MPI_COMM_WORLD)); return global;
}
inline bool verify_distributed(const Key* a,uint64_t n,const Config& c,const Fingerprint& before,int rank,int p) {
    auto after=global_fingerprint(a,n,c.threads);
    int ok=ordered(a,n,c.threads) && after==before;
    std::vector<uint64_t> counts(p); mpi_check(MPI_Allgather(&n,1,MPI_UINT64_T,counts.data(),1,MPI_UINT64_T,MPI_COMM_WORLD));
    int left=MPI_PROC_NULL,right=MPI_PROC_NULL;
    if(n) { for(int i=rank-1;i>=0;--i) if(counts[i]) { left=i; break; } for(int i=rank+1;i<p;++i) if(counts[i]) { right=i; break; } }
    Key largest=n ? a[n-1]:0,prev=0;
    mpi_check(MPI_Sendrecv(&largest,1,MPI_UINT32_T,right,91,&prev,1,MPI_UINT32_T,left,91,MPI_COMM_WORLD,MPI_STATUS_IGNORE));
    if(n && left!=MPI_PROC_NULL && prev>a[0]) ok=0;
    if(c.n<=100000) {
        std::vector<int> sizes(p),offsets(p); uint64_t total=0;
        for(int i=0;i<p;++i) { sizes[i]=mpi_int(counts[i]); offsets[i]=mpi_int(total); total+=counts[i]; }
        auto gathered=buffer(rank==0 ? total:0);
        mpi_check(MPI_Gatherv(a,mpi_int(n),MPI_UINT32_T,gathered.get(),sizes.data(),offsets.data(),MPI_UINT32_T,0,MPI_COMM_WORLD));
        if(rank==0) { auto expected=buffer(c.n); generate(expected.get(),c.n,0,c); std::sort(expected.get(),expected.get()+c.n); if(total!=c.n || !std::equal(expected.get(),expected.get()+c.n,gathered.get())) ok=0; }
    }
    int all=0; mpi_check(MPI_Allreduce(&ok,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD)); return all;
}
inline PhaseTimes max_phases(const PhaseTimes& x) {
    double in[9]={x.local_sort,x.sampling,x.splitters,x.partition,x.exchange,x.merge,x.h2d,x.d2h,x.other},out[9]={};
    mpi_check(MPI_Reduce(in,out,9,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD));
    return {out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8]};
}
int mpi_main(int,char**,bool hybrid);
}
