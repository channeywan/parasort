#include "psrs_core.hpp"
using namespace parasort;
int main(int argc,char** argv) { return run_cpu(argc,argv,"omp","psrs",[](Key* a,Key* o,uint64_t n,const Config& c) { require_variant(c,{"psrs","serial_init"}); return psrs_openmp(a,o,n,c); }); }
