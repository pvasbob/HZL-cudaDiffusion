#include "hzl/observation.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#ifdef HZL_OPENCV_REFERENCE
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif
void expect(bool b,const char* s) { if(!b) throw std::runtime_error(s); }
double difference(std::span<const double> a,std::span<const double> b) {
    expect(a.size()==b.size(),"size mismatch"); double d=0;
    for(std::size_t i=0;i<a.size();++i) d=std::max(d,std::abs(a[i]-b[i])); return d;
}
template<class F> void rejects(F f) { bool caught=false; try { f(); } catch(const std::invalid_argument&) { caught=true; } expect(caught,"invalid observation configuration accepted"); }
int main(int argc,char**) {
 try {
    auto backend=argc>1 ? hzl::Backend::cuda : hzl::Backend::cpu;
#ifdef HZL_HAS_CUDA
    if(backend==hzl::Backend::cuda && !hzl::cuda_available()) { std::cout<<"SKIP: CUDA device unavailable\n"; return 77; }
#endif
    for(auto dims:{std::pair{2,3},std::pair{37,23},std::pair{64,33}}) {
        hzl::Grid g{dims.first,dims.second,0.001,0.001}; hzl::Physics p;
        auto c=hzl::gaussian(g); c[0]=1; c.back()=0.7;
        auto image=hzl::render(g,c,3.5,11); auto pristine=image.pixels;
        auto decoded=hzl::reconstruct(g,image,{},backend);
        expect(difference(c,decoded.concentration)<3e-16,"roundtrip reconstruction failed");
        expect(decoded.timestamp==3.5,"observation timestamp lost");
        for(int radius:{1,2,8}) {
            hzl::Filter f{radius,1.2}; auto filtered=hzl::reconstruct(g,image,f,backend);
            auto direct=f; direct.reference=true; auto cpu=hzl::reconstruct(g,image,direct);
            expect(difference(cpu.concentration,filtered.concentration)<2e-14,"CUDA tile/halo disagrees with CPU");
            expect(difference(c,filtered.concentration)>1e-3,"filter did not affect observations");
#ifdef HZL_OPENCV_REFERENCE
            cv::Mat input(g.height,g.width,CV_64F,decoded.concentration.data()); cv::Mat reference;
            cv::GaussianBlur(input,reference,cv::Size(2*radius+1,2*radius+1),f.sigma,f.sigma,cv::BORDER_REPLICATE);
            expect(difference(filtered.concentration,std::span<const double>(reference.ptr<double>(),g.size()))<2e-14,"OpenCV Gaussian disagreement");
#endif
        }
        // Padding deliberately contains NaN: only active pixels may be read.
        for(int y=0;y<g.height;++y) for(int x=0;x<g.width;++x)
            expect(image.pixels[y*(g.width+11)+x]==pristine[y*(g.width+11)+x],"image mutated");
        auto invalid=image; invalid.stride_bytes=1; rejects([&]{ hzl::reconstruct(g,invalid); });
        invalid=image; invalid.pixels.pop_back(); rejects([&]{ hzl::reconstruct(g,invalid); });
        invalid=image; invalid.pixels[0]=std::numeric_limits<double>::quiet_NaN(); rejects([&]{ hzl::reconstruct(g,invalid); });
        rejects([&]{ hzl::reconstruct(g,image,{-1,1}); });
        rejects([&]{ hzl::reconstruct(g,image,{1,0}); });
        invalid=image; invalid.timestamp=-1; rejects([&]{ hzl::reconstruct(g,invalid); });
        hzl::Grid wrong=g; wrong.width+=1; rejects([&]{ hzl::reconstruct(wrong,image); });
        std::vector<double> constant(g.size(),0.4);
        auto flat=hzl::reconstruct(g,hzl::render(g,constant,0,5),{8,1.2},backend);
        expect(difference(flat.concentration,constant)<2e-14,"filter changed constant field");
        std::vector<double> temp(g.size(),900),hot(g.size(),1100);
        hzl::Process process(g,p,c,temp,backend); auto snapshot=hzl::reconstruct(g,process.observe(3),{},backend);
        auto saved=snapshot.concentration;
        auto a=hzl::forecast(snapshot,p,temp,5,backend),b=hzl::forecast(snapshot,p,hot,5,backend);
        expect(process.timestamp()==0 && difference(process.concentration_for_validation(),c)==0,"forecast changed process");
        expect(snapshot.concentration==saved,"forecast changed snapshot");
        expect(difference(a.concentration,b.concentration)>1e-4,"candidate temperature has no effect");
        expect(a.start_time==0 && a.endpoint_time==5,"forecast timestamps incorrect");
        auto expected=hzl::advance_cpu(g,p,c,hot,1); process.apply_temperature(hot); process.advance(1);
        expect(process.timestamp()==1,"process time did not advance one interval");
        expect(difference(expected,process.concentration_for_validation())<2e-13,"process adopted wrong endpoint/control");
        auto next=hzl::reconstruct(g,process.observe(),{},backend); expect(next.timestamp==1,"next observation timestamp wrong");
        expect(difference(next.concentration,snapshot.concentration)>1e-4,"snapshots do not evolve");
        process.advance(1); expect(process.observe().timestamp==2,"chronology failed");
        rejects([&]{ process.advance(-1); }); expect(process.timestamp()==2,"failed advancement changed time");
        auto badtemp=hot; badtemp[0]=0; rejects([&]{ process.apply_temperature(badtemp); });
    }
    std::cout<<"PASS: strided measurement encoding, filter bypass, Gaussian radii/boundaries, chronological process, reconstructed forecasts, candidate isolation, applied controls, invalid input";
#ifdef HZL_OPENCV_REFERENCE
    std::cout<<", OpenCV reference";
#endif
    std::cout<<'\n';
 } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
