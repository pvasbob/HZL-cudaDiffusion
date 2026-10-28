#include "hzl/observation.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
int main(int argc,char** argv) {
 try {
    hzl::Backend backend=hzl::Backend::cpu; hzl::Filter filter;
    for(int i=1;i<argc;++i) {
        std::string arg=argv[i];
        if(arg=="--cuda") backend=hzl::Backend::cuda;
        else if(arg=="--filter") filter={2,1};
        else throw std::invalid_argument("usage: hzl-observe [--cuda] [--filter]");
    }
    hzl::Grid grid; hzl::Physics physics;
    std::vector<double> temp(grid.size(),900);
    hzl::Process process(grid,physics,hzl::gaussian(grid),temp,backend);
    double initial_mass=hzl::mass(grid,process.concentration_for_validation());
    std::cout<<std::setprecision(12)<<"timestamp_s,forecast_endpoint_s,interval_s,horizon_s,reconstruction_rmse,filter_change_rmse,mass_relative_error,observation_forecast_ms\n";
    for(int k=0;k<5;++k) {
        auto start=std::chrono::steady_clock::now();
        auto image=process.observe(7); auto snapshot=hzl::reconstruct(grid,image,filter,backend);
        auto exact=hzl::reconstruct(grid,image,{},backend);
        auto prediction=hzl::forecast(snapshot,physics,temp,5,backend);
        double error=0,change=0;
        auto truth=process.concentration_for_validation();
        for(std::size_t i=0;i<truth.size();++i) {
            error+=std::pow(snapshot.concentration[i]-truth[i],2);
            change+=std::pow(snapshot.concentration[i]-exact.concentration[i],2);
        }
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::cout<<snapshot.timestamp<<','<<prediction.endpoint_time<<",1,5,"<<std::sqrt(error/truth.size())<<','<<std::sqrt(change/truth.size())<<','<<hzl::mass(grid,truth)/initial_mass-1<<','<<ms<<'\n';
        std::ofstream file("observation-"+std::to_string(k)+".csv");
        if(!file) throw std::runtime_error("cannot open observation output");
        file<<std::setprecision(17)<<"# observed_s="<<snapshot.timestamp<<", forecast_s="<<prediction.endpoint_time<<", fixed_future_temp_K="<<temp[0]<<"\nx,y,measurement,reconstructed,underlying,forecast,temp_K\n";
        auto stride=image.stride_bytes/sizeof(double);
        for(int y=0;y<grid.height;++y) for(int x=0;x<grid.width;++x) {
            auto i=std::size_t(y)*grid.width+x;
            file<<x<<','<<y<<','<<image.pixels[y*stride+x]<<','<<snapshot.concentration[i]<<','<<truth[i]<<','<<prediction.concentration[i]<<','<<temp[i]<<'\n';
        }
        file.flush(); if(!file) throw std::runtime_error("observation write failed");
        std::ofstream display("observation-"+std::to_string(k)+".pgm",std::ios::binary);
        if(!display) throw std::runtime_error("cannot open display output");
        display<<"P5\n# timestamp_s="<<snapshot.timestamp<<"; display range [0,1]\n"<<grid.width<<' '<<grid.height<<"\n255\n";
        for(double value:truth) display.put(static_cast<char>(static_cast<unsigned char>(255*std::clamp(value,0.0,1.0))));
        display.flush(); if(!display) throw std::runtime_error("display write failed");
        // Only one physical second is applied, independently of the 5s forecast.
        if(k<4) { std::fill(temp.begin(),temp.end(),k%2==0 ? 1000 : 900); process.apply_temperature(temp); process.advance(1); }
    }
 } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
