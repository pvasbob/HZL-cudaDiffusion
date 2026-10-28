#include "hzl/control.hpp"
#ifdef HZL_HAS_CUDA
#include "hzl/cuda_forecast.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
void row(const std::string& operation,const std::string& variant,const hzl::Grid& g,double parameter,double setup,std::vector<double> times,double device,double transfer,double error,double mass_error) {
    std::sort(times.begin(),times.end()); double mean=std::accumulate(times.begin(),times.end(),0.)/times.size(); auto n=times.size();
    std::cout<<operation<<','<<variant<<','<<g.width<<','<<g.height<<','<<parameter<<','<<setup<<','<<n<<','<<mean<<','<<(times[(n-1)/2]+times[n/2])/2<<','<<times[std::size_t(std::ceil(0.95*n))-1]<<','<<times.back()<<','<<device<<','<<transfer<<','<<error<<','<<mass_error<<'\n';
}
}
int main(int argc,char** argv) {
 try {
    bool cuda=false,quick=false;
    for(int i=1;i<argc;++i) { std::string a=argv[i]; if(a=="--cuda") cuda=true; else if(a=="--quick") quick=true; else throw std::invalid_argument("usage: hzl-compare [--cuda] [--quick]"); }
#ifndef HZL_HAS_CUDA
    if(cuda) throw std::runtime_error("CUDA backend not built");
#else
    if(cuda && !hzl::cuda_available()) throw std::runtime_error("CUDA unavailable; no CPU fallback performed");
#endif
    std::cout<<"operation,variant,width,height,horizon_s_or_radius,setup_ms,samples,mean_ms,median_ms,p95_ms,max_ms,device_mean_ms,final_download_ms,mse_vs_cpu,mass_relative_error\n";
    std::vector<std::pair<int,int>> dimensions=quick ? std::vector<std::pair<int,int>>{{33,25},{65,49}}:std::vector<std::pair<int,int>>{{32,32},{129,97},{256,256},{513,257},{1024,1024}};
    int samples=quick?5:20;
    for(auto [width,height]:dimensions) {
        hzl::Grid g{width,height,0.001,0.001}; hzl::Physics p; auto initial=hzl::pattern(g,hzl::Pattern::deposits);
        std::vector<double> temp(g.size()); for(std::size_t i=0;i<temp.size();++i) temp[i]=800+200*double(i%width)/(width-1);
        for(double horizon:quick?std::vector<double>{5}:std::vector<double>{5,30}) {
            auto reference=hzl::advance_cpu(g,p,initial,temp,horizon); auto target=hzl::gaussian(g);
            std::vector<double> times; double objective=0;
            for(int run=-1;run<samples;++run) {
                auto start=Clock::now(); auto out=hzl::advance_cpu(g,p,initial,temp,horizon); objective=hzl::mse(out,target);
                double latency=ms(start); if(run>=0) times.push_back(latency);
            }
            row("forecast_mse","cpu_double",g,horizon,0,times,0,0,0,hzl::mass(g,reference)/hzl::mass(g,initial)-1);
#ifdef HZL_HAS_CUDA
            if(cuda) for(bool tiled:{false,true}) for(bool graph:{false,true}) {
                auto start=Clock::now(); hzl::CudaForecaster solver({g,0,initial},target,tiled?hzl::CudaKernel::tiled:hzl::CudaKernel::global,graph);
                double setup=ms(start),device=0; times.clear();
                for(int run=-1;run<samples;++run) {
                    start=Clock::now(); double actual=solver.evaluate(p,temp,horizon); double latency=ms(start);
                    if(std::abs(actual-objective)>1e-12) throw std::runtime_error("CUDA objective accuracy failed");
                    if(run>=0) { times.push_back(latency); device+=solver.device_ms(); }
                }
                start=Clock::now(); auto out=solver.download(); double download=ms(start),error=hzl::mse(out,reference);
                if(error>1e-24) throw std::runtime_error("CUDA forecast accuracy failed");
                row("forecast_mse",std::string(tiled?"cuda_tiled":"cuda_global")+(graph?"_graph":"_launch"),g,horizon,setup,times,device/samples,download,error,hzl::mass(g,out)/hzl::mass(g,initial)-1);
            }
#endif
        }
        auto image=hzl::render(g,initial,0,13);
        for(int radius:quick?std::vector<int>{0,2}:std::vector<int>{0,1,2,4,8}) {
            hzl::Filter filter{radius,1.3}; auto direct=filter; direct.reference=true; auto reference=hzl::reconstruct(g,image,direct);
            for(int variant=0;variant<(cuda?3:2);++variant) {
                std::vector<double> times,output; double error=0;
                for(int run=-1;run<samples;++run) {
                    auto start=Clock::now();
                    if(variant==0) output=hzl::reconstruct(g,image,direct).concentration;
                    else if(variant==1) {
                        auto decoded=hzl::reconstruct(g,image); cv::Mat data(g.height,g.width,CV_64F,decoded.concentration.data()),out;
                        if(radius) cv::GaussianBlur(data,out,{2*radius+1,2*radius+1},filter.sigma,filter.sigma,cv::BORDER_REPLICATE); else out=data;
                        output.assign(out.ptr<double>(),out.ptr<double>()+g.size());
                    } else output=hzl::reconstruct(g,image,filter,hzl::Backend::cuda).concentration;
                    double latency=ms(start); if(run>=0) times.push_back(latency);
                    error=std::max(error,hzl::mse(output,reference.concentration));
                }
                if(error>1e-24) throw std::runtime_error("filter accuracy failed");
                row("decode_filter",variant==0?"cpu_direct":variant==1?"opencv_separable":"cuda_tiled",g,radius,0,times,0,0,error,hzl::mass(g,output)/hzl::mass(g,initial)-1);
            }
        }
    }
 } catch(const std::exception& e) { std::cerr<<"error: "<<e.what()<<'\n'; return 1; }
}
