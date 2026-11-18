#include "datagen.hpp"
#include "verify.hpp"
#include "psrs_core.hpp"
#include "cli.hpp"
#include <cassert>
#include <iostream>
using namespace parasort;
int main() {
    assert(parse_size("2^30")==1073741824); assert(parse_size("1e9")==1000000000);
    bool threw=false; try { parse_size("-1"); } catch(...) { threw=true; } assert(threw);
    for(auto dist:{"uniform","normal","power","hotkeys","sorted","reverse","nearly_sorted","few_unique"}) {
        Config c; c.n=1003; c.dist=dist; c.threads=3;
        std::vector<Key> a(c.n),b(c.n),again(c.n); generate(a.data(),c.n,0,c); generate(again.data(),c.n,0,c);
        for(int r=0;r<7;++r) { auto lo=cut(c.n,r,7),hi=cut(c.n,r+1,7); generate(b.data()+lo,hi-lo,lo,c); }
        assert(a==b && a==again); auto before=fingerprint(a.data(),a.size(),3);
        std::sort(a.begin(),a.end()); assert(ordered(a.data(),a.size(),3)); assert(before==fingerprint(a.data(),a.size(),3));
        a[0]^=1; assert(!(before==fingerprint(a.data(),a.size(),3)));
    }
    Key equal[10]={7,7,7,7,7,7,7,7,7,7}; Sample s{7,1,4};
    assert(split_position(equal,10,0,s,true)==10); assert(split_position(equal,10,1,s,true)==5); assert(split_position(equal,10,2,s,true)==0);
    std::cout<<"PASS deterministic generation, partitioned generation, checksum fault detection, tie-break boundaries, CLI\n";
}
