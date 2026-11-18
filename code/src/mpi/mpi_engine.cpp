#include "mpi_support.hpp"
#include <cstddef>
namespace parasort {
int mpi_main(int argc,char** argv,bool hybrid) {
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided); int rank=0,p=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p); MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
    try {
        if(provided<MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED unavailable");
        Config c=parse_cli(argc,argv,"psrs"); if(c.help) { if(!rank) std::cout<<help_text(); MPI_Finalize(); return 0; }
        require_variant(c,{"psrs","allgather","serial_init"}); if(!hybrid) c.threads=1;
        uint64_t begin=cut(c.n,rank,p),n=cut(c.n,rank+1,p)-begin; mpi_int(n);
        auto a=buffer(n),local_out=buffer(n); first_touch(local_out.get(),n,c.threads,c.variant=="serial_init");
        MPI_Datatype sample_type; static_assert(sizeof(Sample)==3*sizeof(uint64_t));
        mpi_check(MPI_Type_contiguous(3,MPI_UINT64_T,&sample_type)); mpi_check(MPI_Type_commit(&sample_type));
        for(int rep=-1;rep<c.repeat;++rep) {
            double scatter_ms=0;
            if(c.scatter) {
                // Scatterv's legacy int displacement restricts this optional demonstration to n < INT_MAX.
                mpi_int(c.n); auto all=buffer(rank==0 ? c.n:0); if(!rank) generate(all.get(),c.n,0,c);
                std::vector<int> counts(p),offsets(p); for(int i=0;i<p;++i) { offsets[i]=mpi_int(cut(c.n,i,p)); counts[i]=mpi_int(cut(c.n,i+1,p)-cut(c.n,i,p)); }
                mpi_check(MPI_Barrier(MPI_COMM_WORLD)); double ts=MPI_Wtime();
                mpi_check(MPI_Scatterv(all.get(),counts.data(),offsets.data(),MPI_UINT32_T,a.get(),mpi_int(n),MPI_UINT32_T,0,MPI_COMM_WORLD));
                double elapsed=(MPI_Wtime()-ts)*1000; mpi_check(MPI_Allreduce(&elapsed,&scatter_ms,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD));
            } else generate(a.get(),n,begin,c,c.variant=="serial_init");
            Fingerprint before; if(c.verify) before=global_fingerprint(a.get(),n,c.threads);
            mpi_check(MPI_Barrier(MPI_COMM_WORLD)); double start=MPI_Wtime(); Timer phase; SortResult result;
            Key* local=a.get();
            if(hybrid) local=psrs_openmp(a.get(),local_out.get(),n,c).data;
            else std::sort(local,local+n);
            result.phase.local_sort=phase.lap();
            auto samples=regular_samples(local,n,rank,uint64_t(c.oversample)*p); int ns=mpi_int(samples.size());
            std::vector<int> counts(p),offsets(p); mpi_check(MPI_Allgather(&ns,1,MPI_INT,counts.data(),1,MPI_INT,MPI_COMM_WORLD));
            uint64_t S=0; for(int i=0;i<p;++i) { offsets[i]=mpi_int(S); S+=counts[i]; } mpi_int(S);
            bool allgather=c.variant=="allgather";
            std::vector<Sample> all((allgather || rank==0) ? S:0);
            if(allgather) mpi_check(MPI_Allgatherv(samples.data(),ns,sample_type,all.data(),counts.data(),offsets.data(),sample_type,MPI_COMM_WORLD));
            else mpi_check(MPI_Gatherv(samples.data(),ns,sample_type,all.data(),counts.data(),offsets.data(),sample_type,0,MPI_COMM_WORLD));
            result.phase.sampling=phase.lap(); std::vector<Sample> splitters(p-1);
            if(allgather || !rank) splitters=select_splitters(std::move(all),p);
            if(!allgather) mpi_check(MPI_Bcast(splitters.data(),p-1,sample_type,0,MPI_COMM_WORLD));
            result.phase.splitters=phase.lap();
            std::vector<int> sendcounts(p),sendoffsets(p),recvcounts(p),recvoffsets(p); uint64_t prev=0;
            for(int j=0;j<p;++j) { uint64_t end=j==p-1 ? n:split_position(local,n,rank,splitters[j],c.tiebreak); sendoffsets[j]=mpi_int(prev); sendcounts[j]=mpi_int(end-prev); prev=end; }
            result.phase.partition=phase.lap();
            mpi_check(MPI_Alltoall(sendcounts.data(),1,MPI_INT,recvcounts.data(),1,MPI_INT,MPI_COMM_WORLD));
            uint64_t received=0; for(int j=0;j<p;++j) { recvoffsets[j]=mpi_int(received); received+=recvcounts[j]; } mpi_int(received);
            auto incoming=buffer(received),out=buffer(received);
            first_touch(incoming.get(),received,c.threads); first_touch(out.get(),received,c.threads);
            mpi_check(MPI_Alltoallv(local,sendcounts.data(),sendoffsets.data(),MPI_UINT32_T,incoming.get(),recvcounts.data(),recvoffsets.data(),MPI_UINT32_T,MPI_COMM_WORLD));
            result.phase.exchange=phase.lap(); std::vector<Run> runs;
            for(int j=0;j<p;++j) runs.push_back({incoming.get()+recvoffsets[j],uint64_t(recvcounts[j])});
            if(hybrid) parallel_merge_runs(runs,out.get(),c); else merge_runs(runs,out.get());
            result.phase.merge=phase.lap(); double elapsed=(MPI_Wtime()-start)*1000,total=0;
            mpi_check(MPI_Reduce(&elapsed,&total,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD));
            uint64_t largest=0; mpi_check(MPI_Reduce(&received,&largest,1,MPI_UINT64_T,MPI_MAX,0,MPI_COMM_WORLD));
            result.phase=max_phases(result.phase); result.imbalance=imbalance(largest,c.n,p); result.data=out.get();
            int verified=c.verify ? int(verify_distributed(out.get(),received,c,before,rank,p)):-1;
            if(verified==0) throw std::runtime_error("distributed verification FAILED");
            result.extra="scatter="+std::to_string(c.scatter)+";scatter_ms="+std::to_string(scatter_ms)+";receive_allocation_timed=1";
            if(rep>=0 && !rank) emit_csv(c,hybrid ? "hybrid":"mpi",result,rep,total,verified,p);
        }
        MPI_Type_free(&sample_type); MPI_Finalize(); return 0;
    } catch(const std::exception& e) { std::cerr<<"rank "<<rank<<": "<<e.what()<<'\n'; MPI_Abort(MPI_COMM_WORLD,1); return 1; }
}
}
