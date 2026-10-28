#include "hzl/model.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace hzl {
namespace { void require(bool b, const char* s) { if (!b) throw std::invalid_argument(s); } }
void Grid::validate() const {
    require(width >= 2 && height >= 2 && width <= 32768 && height <= 32768, "grid dimensions must be in [2,32768]");
    require(std::isfinite(lx) && std::isfinite(ly) && lx > 0 && ly > 0, "domain lengths must be positive finite metres");
    require(std::isfinite(1/(dx()*dx())) && std::isfinite(1/(dy()*dy())), "grid spacing is too small");
}
std::size_t Grid::size() const { validate(); return std::size_t(width)*height; }
void Physics::validate() const {
    require(std::isfinite(d0) && d0 > 0 && std::isfinite(ea) && ea >= 0, "invalid Arrhenius parameters");
    require(std::isfinite(min_temp) && std::isfinite(max_temp) && min_temp > 0 && max_temp >= min_temp, "invalid Kelvin bounds");
}
double Physics::diffusivity(double t) const {
    validate(); require(std::isfinite(t) && t >= min_temp && t <= max_temp, "temperature outside Kelvin bounds");
    double d = d0*std::exp(-ea/(8.617333262145e-5*t));
    require(std::isfinite(d) && d > 0, "diffusivity underflow or overflow"); return d;
}
void validate_concentration(const Grid& g, std::span<const double> c) {
    require(c.size()==g.size(), "concentration dimensions mismatch");
    for(double v:c) require(std::isfinite(v) && v>=0, "concentration must be finite and nonnegative");
}
Plan plan(const Grid& g,const Physics& p,std::span<const double> temp,double seconds,double max_dt) {
    require(temp.size()==g.size(), "temperature dimensions mismatch"); p.validate();
    require(std::isfinite(seconds) && seconds>=0, "duration must be finite nonnegative seconds");
    Plan r; r.diffusivity.reserve(temp.size());
    for(double t:temp) r.diffusivity.push_back(p.diffusivity(t));
    double d=*std::max_element(r.diffusivity.begin(),r.diffusivity.end());
    auto integration=schedule(g,d,seconds,max_dt);
    r.steps=integration.steps; r.dt=integration.dt; return r;
}
Schedule schedule(const Grid& g,double d,double seconds,double max_dt) {
    g.validate();
    require(std::isfinite(d) && d>0,"maximum diffusivity must be positive finite");
    require(std::isfinite(seconds) && seconds>=0,"duration must be finite nonnegative seconds");
    // All face conductances are <= Dmax; safety factor 0.9.
    double limit=0.9/(2*d*(1/(g.dx()*g.dx())+1/(g.dy()*g.dy())));
    require(std::isfinite(limit) && limit>0, "invalid stability bound");
    require(max_dt>0 && !std::isnan(max_dt), "maximum time step must be positive");
    limit=std::min(limit,max_dt);
    double n=std::ceil(seconds/limit);
    require(std::isfinite(n) && n<=10000000, "integration exceeds 10 million step budget");
    auto steps=static_cast<std::size_t>(n); return {steps,steps ? seconds/steps : 0};
}
std::vector<double> gaussian(const Grid& g) {
    std::vector<double> c(g.size());
    for(int y=0;y<g.height;++y) for(int x=0;x<g.width;++x) {
        double a=((x+0.5)/g.width-0.5)/0.08, b=((y+0.5)/g.height-0.5)/0.08;
        c[std::size_t(y)*g.width+x]=std::exp(-0.5*(a*a+b*b));
    } return c;
}
std::vector<double> advance_cpu(const Grid& g,const Physics& p,std::span<const double> conc,std::span<const double> temp,double seconds,double max_dt) {
    validate_concentration(g,conc); auto r=plan(g,p,temp,seconds,max_dt);
    std::vector<double> a(conc.begin(),conc.end()),b(a.size());
    double ix=1/(g.dx()*g.dx()),iy=1/(g.dy()*g.dy());
    for(std::size_t s=0;s<r.steps;++s) {
        for(int y=0;y<g.height;++y) for(int x=0;x<g.width;++x) {
            auto i=std::size_t(y)*g.width+x; double flux=0;
            auto face=[&](std::size_t j,double inv) { return (0.5*r.diffusivity[i]+0.5*r.diffusivity[j])*(a[j]-a[i])*inv; };
            if(x>0) flux+=face(i-1,ix);
            if(x+1<g.width) flux+=face(i+1,ix);
            if(y>0) flux+=face(i-g.width,iy);
            if(y+1<g.height) flux+=face(i+g.width,iy);
            b[i]=a[i]+r.dt*flux;
        } a.swap(b);
    } validate_concentration(g,a); return a;
}
double mass(const Grid& g,std::span<const double> c) {
    validate_concentration(g,c); double value=std::accumulate(c.begin(),c.end(),0.0)*g.dx()*g.dy();
    require(std::isfinite(value),"mass overflow"); return value;
}
}
