#include "hzl/model.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
void expect(bool b,const char* message) { if(!b) throw std::runtime_error(message); }
template<class F> void rejects(F f) { bool caught=false; try { f(); } catch(const std::invalid_argument&) { caught=true; } expect(caught,"invalid configuration accepted"); }
double cosine_error(int n,double duration) {
    hzl::Grid g{n,n,1,1}; hzl::Physics p{0.01,0,300,1200};
    std::vector<double> c(g.size()),t(g.size(),900);
    for(int y=0;y<n;++y) for(int x=0;x<n;++x) c[y*n+x]=1+0.5*std::cos(std::numbers::pi*(x+0.5)/n);
    auto a=hzl::advance_cpu(g,p,c,t,duration,0.0005); double error=0;
    for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
        double exact=1+0.5*std::exp(-0.01*std::numbers::pi*std::numbers::pi*duration)*std::cos(std::numbers::pi*(x+0.5)/n);
        error+=(a[y*n+x]-exact)*(a[y*n+x]-exact);
    } return std::sqrt(error/g.size());
}
double time_error(double dt) {
    constexpr int n=24; hzl::Grid g{n,n,1,1}; hzl::Physics p{0.01,0,300,1200};
    std::vector<double> c(g.size()),temp(g.size(),900);
    for(int y=0;y<n;++y) for(int x=0;x<n;++x) c[y*n+x]=1+0.5*std::cos(std::numbers::pi*(x+0.5)/n);
    auto out=hzl::advance_cpu(g,p,c,temp,1,dt);
    double lambda=-4*p.d0*std::pow(std::sin(std::numbers::pi/(2*n)),2)/(g.dx()*g.dx()),error=0;
    for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
        double expected=1+0.5*std::exp(lambda)*std::cos(std::numbers::pi*(x+0.5)/n);
        error+=std::pow(out[y*n+x]-expected,2);
    } return std::sqrt(error/g.size());
}
int main(int argc,char**) {
 try {
    bool gpu=argc>1;
#ifdef HZL_HAS_CUDA
    if(gpu && !hzl::cuda_available()) { std::cout<<"SKIP: CUDA device/driver unavailable\n"; return 77; }
#else
    expect(!gpu,"CUDA backend not built");
#endif
    hzl::Grid g{37,23,0.001,0.0014}; hzl::Physics p;
    auto conc=hzl::gaussian(g); auto original=conc; std::vector<double> temp(g.size());
    for(std::size_t i=0;i<temp.size();++i) temp[i]=600+500*double(i%37)/36;
    auto advance=[&](std::span<const double> c,double duration) {
#ifdef HZL_HAS_CUDA
        if(gpu) return hzl::advance_cuda(g,p,c,temp,duration);
#endif
        return hzl::advance_cpu(g,p,c,temp,duration);
    };
    auto out=advance(conc,30);
    expect(conc==original,"forecast mutated input");
    expect(std::abs(hzl::mass(g,out)/hzl::mass(g,conc)-1)<2e-13,"mass conservation failed");
    expect(*std::min_element(out.begin(),out.end())>=0,"negative concentration");
    expect(*std::max_element(out.begin(),out.end())<=*std::max_element(conc.begin(),conc.end())+1e-14,"maximum principle failed");
    std::vector<double> uniform(g.size(),0.35); expect(advance(uniform,30)==uniform,"uniform state changed");
    expect(advance(conc,0)==conc,"zero duration changed state");
    auto twice=advance(advance(conc,15),15);
    auto direct=advance(conc,30); // Independent calls must give deterministic isolated results.
    expect(direct==out,"candidate isolation failed");
    expect(twice!=conc,"successive advancement did not evolve");
#ifdef HZL_HAS_CUDA
    if(gpu) {
        auto cpu=hzl::advance_cpu(g,p,conc,temp,30); double maxerr=0;
        for(std::size_t i=0;i<cpu.size();++i) maxerr=std::max(maxerr,std::abs(cpu[i]-out[i]));
        expect(maxerr<2e-13,"CPU/GPU disagreement");
    }
#endif
    expect(p.diffusivity(1000)>p.diffusivity(800),"incorrect temperature dependence");
    rejects([&]{ hzl::Grid bad{1,4,1,1}; bad.validate(); });
    rejects([&]{ auto t=temp; t[0]=0; hzl::plan(g,p,t,1); });
    rejects([&]{ hzl::plan(g,p,temp,-1); });
    rejects([&]{ auto c=conc; c[0]=-1; hzl::advance_cpu(g,p,c,temp,1); });
    rejects([&]{ hzl::advance_cpu(g,p,std::span<const double>{},temp,1); });
    // Variable-D operator: a discontinuous concentration in a 2x2 grid,
    // one tiny step; independent two-cell flux and boundary balance.
    hzl::Grid tiny{2,2,2,2}; hzl::Physics simple{1,0.001,300,1200};
    std::vector<double> c{1,0,1,0},t{300,1200,300,1200}; double dt=0.01;
    auto v=hzl::advance_cpu(tiny,simple,c,t,dt);
    double flux=0.5*(simple.diffusivity(300)+simple.diffusivity(1200))*dt;
    expect(std::abs(v[0]-(1-flux))<1e-15 && std::abs(v[1]-flux)<1e-15,"variable face flux/boundaries incorrect");
    double e16=cosine_error(16,1), e32=cosine_error(32,1),e64=cosine_error(64,1);
    expect(e32<e16*0.35 && e64<e32*0.35,"space/time refinement convergence failed");
    rejects([&]{ hzl::mass(g,std::vector<double>(g.size(),1e308)); });
    double t1=time_error(0.03),t2=time_error(0.015),t3=time_error(0.0075);
    expect(t2<t1*0.55 && t3<t2*0.55,"independent first-order time convergence failed");
    rejects([&]{ hzl::plan(g,p,temp,1,0); });
    std::cout<<"time errors "<<t1<<' '<<t2<<' '<<t3<<'\n';
    std::cout<<"PASS: conservation, positivity, maximum principle, uniform state, boundaries, variable diffusivity, isolation, invalid inputs, analytical convergence; errors "<<e16<<' '<<e32<<' '<<e64<<'\n';
 } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
