#include "hzl/model.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
int main(int argc,char** argv) {
 try {
    bool gpu=argc==2 && std::string(argv[1])=="--cuda";
    if(argc>1 && !gpu) throw std::invalid_argument("usage: hzl-benchmark [--cuda]");
    hzl::Grid g; hzl::Physics p; auto c=hzl::gaussian(g); std::vector<double> temp(g.size(),1000);
    std::cout<<"backend,width,height,horizon_s,steps,samples,mean_ms,median_ms,p95_ms,max_ms,deadline_misses,real_time_factor,relative_mass_error,device_mean_ms\n";
    for(double horizon:{5.,30.,120.,600.}) {
      auto plan=hzl::plan(g,p,temp,horizon); std::vector<double> times; double err=0,device_total=0;
      for(int run=-1;run<10;++run) {
        double device=0; auto start=std::chrono::steady_clock::now(); std::vector<double> out;
        if(gpu) {
#ifdef HZL_HAS_CUDA
          out=hzl::advance_cuda(g,p,c,temp,horizon,&device);
#else
          throw std::runtime_error("CUDA backend not built");
#endif
        } else out=hzl::advance_cpu(g,p,c,temp,horizon);
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(run>=0) { times.push_back(ms); device_total+=device; }
        err=std::max(err,std::abs(hzl::mass(g,out)/hzl::mass(g,c)-1));
      }
      std::sort(times.begin(),times.end()); double mean=std::accumulate(times.begin(),times.end(),0.)/times.size();
      std::cout<<(gpu?"cuda":"cpu")<<','<<g.width<<','<<g.height<<','<<horizon<<','<<plan.steps<<",10,"<<mean<<','<<(times[4]+times[5])/2<<','<<times[9]<<','<<times.back()<<','<<std::count_if(times.begin(),times.end(),[](double t){return t>1000;})<<','<<horizon*1000/mean<<','<<err<<','<<(gpu?std::to_string(device_total/10):"NA")<<'\n';
    }
 } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
