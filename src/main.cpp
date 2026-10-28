#include "hzl/model.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
int main(int argc,char** argv) {
    try {
        bool gpu=false; double duration=5; std::string prefix="snapshot";
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--cuda") gpu=true;
            else if(arg=="--seconds" && i+1<argc) { std::string v=argv[++i]; std::size_t n; duration=std::stod(v,&n); if(n!=v.size()) throw std::invalid_argument("invalid duration"); }
            else if(arg=="--output" && i+1<argc) prefix=argv[++i];
            else if(arg=="--help") { std::cout<<"hzl-diffusion [--cuda] [--seconds 5] [--output snapshot]\n"; return 0; }
            else throw std::invalid_argument("unknown or incomplete argument: "+arg);
        }
        hzl::Grid g; hzl::Physics p; auto conc=hzl::gaussian(g); std::vector<double> temp(g.size(),900);
        // A fixed spatial control changes mobility, never adds an attraction term.
        for(int y=0;y<g.height;++y) for(int x=g.width/2;x<g.width;++x) temp[std::size_t(y)*g.width+x]=1000;
        auto integration=hzl::plan(g,p,temp,duration); double initial=hzl::mass(g,conc);
        auto start=std::chrono::steady_clock::now();
        if(gpu) {
#ifdef HZL_HAS_CUDA
            conc=hzl::advance_cuda(g,p,conc,temp,duration);
#else
            throw std::runtime_error("CUDA backend not built; configure with --preset cuda");
#endif
        } else conc=hzl::advance_cpu(g,p,conc,temp,duration);
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::ofstream csv(prefix+".csv"),pgm(prefix+".pgm",std::ios::binary);
        if(!csv || !pgm) throw std::runtime_error("cannot open output files");
        csv<<std::setprecision(17)<<"# timestamp_s="<<duration<<", dx_m="<<g.dx()<<", dy_m="<<g.dy()<<"\nx,y,concentration,temp_K\n";
        pgm<<"P5\n# timestamp_s="<<duration<<"; display range [0,1], not measurement encoding\n"<<g.width<<' '<<g.height<<"\n255\n";
        for(int y=0;y<g.height;++y) for(int x=0;x<g.width;++x) {
            auto i=std::size_t(y)*g.width+x; csv<<x<<','<<y<<','<<conc[i]<<','<<temp[i]<<'\n';
            pgm.put(static_cast<char>(static_cast<unsigned char>(255*std::clamp(conc[i],0.0,1.0))));
        }
        csv.flush(); pgm.flush(); if(!csv || !pgm) throw std::runtime_error("output write failed");
        std::cout<<std::setprecision(10)<<"backend="<<(gpu?"cuda":"cpu")<<" timestamp_s="<<duration<<" steps="<<integration.steps<<" dt_s="<<integration.dt<<" wall_ms="<<ms<<" mass_relative_error="<<(hzl::mass(g,conc)-initial)/initial<<" min="<<*std::min_element(conc.begin(),conc.end())<<'\n';
    } catch(const std::exception& e) { std::cerr<<"error: "<<e.what()<<'\n'; return 1; }
}
