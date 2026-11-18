#include "cli.hpp"
#include <cmath>
#include <limits>
#include <omp.h>
#include <stdexcept>
namespace parasort {
uint64_t parse_size(const std::string& s) {
    auto power=s.find('^');
    if (power!=std::string::npos) {
        if(s.substr(0,power)!="2") throw std::invalid_argument("only 2^k notation is supported");
        size_t used=0; auto exponent=std::stoul(s.substr(power+1),&used);
        if(used!=s.size()-power-1 || exponent>=64) throw std::invalid_argument("invalid exponent: "+s);
        return uint64_t(1)<<exponent;
    }
    if(s.empty() || s.front()=='-') throw std::invalid_argument("invalid nonnegative integer: "+s);
    size_t used=0; long double x=std::stold(s,&used);
    if(used!=s.size() || !std::isfinite(x) || x<0 || x>std::numeric_limits<uint64_t>::max() || std::floor(x)!=x)
        throw std::invalid_argument("invalid nonnegative integer: "+s);
    return static_cast<uint64_t>(x);
}
std::string help_text() { return R"(ParaSort --n <N|2^k|1e9> --dist <uniform|normal|power|hotkeys|sorted|reverse|nearly_sorted|few_unique>
  --seed S --threads T --variant NAME --repeat R --oversample K
  --buckets-per-thread K --schedule static|dynamic|guided --no-tiebreak
  --csv FILE --verify|--no-verify --scatter --batch B --naive-atomics --help
Defaults: n=1e6 seed=42 repeat=5, one additional unrecorded warmup; verification enabled.
)"; }
Config parse_cli(int argc,char** argv,const std::string& default_variant) {
    Config c; c.variant=default_variant; c.threads=omp_get_max_threads();
    auto positive=[](const std::string& s) { auto v=parse_size(s); if(!v || v>4096) throw std::invalid_argument("parameter must be in [1,4096]"); return int(v); };
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        auto value=[&]() { if(i+1==argc) throw std::invalid_argument("missing value for "+a); return std::string(argv[++i]); };
        if(a=="--n") c.n=parse_size(value());
        else if(a=="--seed") c.seed=parse_size(value());
        else if(a=="--threads") c.threads=positive(value());
        else if(a=="--repeat") c.repeat=positive(value());
        else if(a=="--oversample") c.oversample=positive(value());
        else if(a=="--buckets-per-thread") c.buckets_per_thread=positive(value());
        else if(a=="--batch") c.batch=parse_size(value());
        else if(a=="--dist") c.dist=value();
        else if(a=="--variant") c.variant=value();
        else if(a=="--schedule") c.schedule=value();
        else if(a=="--csv") c.csv=value();
        else if(a=="--verify") c.verify=true;
        else if(a=="--no-verify") c.verify=false;
        else if(a=="--no-tiebreak") c.tiebreak=false;
        else if(a=="--scatter") c.scatter=true;
        else if(a=="--naive-atomics") c.naive_atomics=true;
        else if(a=="--help" || a=="-h") c.help=true;
        else throw std::invalid_argument("unknown option: "+a);
    }
    const std::vector<std::string> d={"uniform","normal","power","hotkeys","sorted","reverse","nearly_sorted","few_unique"};
    if(std::find(d.begin(),d.end(),c.dist)==d.end()) throw std::invalid_argument("unknown distribution: "+c.dist);
    if(c.schedule!="static" && c.schedule!="dynamic" && c.schedule!="guided") throw std::invalid_argument("unknown schedule");
    if(c.n>std::numeric_limits<size_t>::max()/sizeof(Key)) throw std::length_error("n exceeds address space");
    if(uint64_t(c.threads)*c.buckets_per_thread>65536) throw std::length_error("too many output buckets");
    omp_set_dynamic(0); omp_set_num_threads(c.threads); return c;
}
void require_variant(const Config& c,std::initializer_list<const char*> variants) {
    for(auto v:variants) if(c.variant==v) return;
    throw std::invalid_argument("unsupported variant: "+c.variant);
}
}
