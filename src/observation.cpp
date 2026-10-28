#include "hzl/observation.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef HZL_OPENCV_IMAGE
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif
namespace hzl {
namespace {
void require(bool b,const char* s) { if(!b) throw std::invalid_argument(s); }
std::vector<double> solve(const Grid& g,const Physics& p,std::span<const double> c,std::span<const double> t,double seconds,Backend backend) {
    if(backend==Backend::cpu) return advance_cpu(g,p,c,t,seconds);
#ifdef HZL_HAS_CUDA
    return advance_cuda(g,p,c,t,seconds);
#else
    throw std::runtime_error("CUDA backend not built");
#endif
}
}
void Image::validate() const {
    require(width>=2 && height>=2 && width<=32768 && height<=32768,"invalid image dimensions");
    require(stride_bytes%sizeof(double)==0 && stride_bytes/sizeof(double)>=std::size_t(width),"invalid image byte stride");
    auto stride=stride_bytes/sizeof(double);
    require(stride<=std::numeric_limits<std::size_t>::max()/std::size_t(height),"image size overflow");
    require(pixels.size()>=stride*height,"truncated image storage");
    require(std::isfinite(timestamp) && timestamp>=0,"invalid image timestamp");
    for(int y=0;y<height;++y) for(int x=0;x<width;++x) {
        double value=pixels[y*stride+x];
        require(std::isfinite(value) && value>=0.1,"invalid measurement intensity");
    }
}
void Filter::validate() const {
    require(radius>=0 && radius<=8 && std::isfinite(sigma) && sigma>0,"filter requires radius 0..8 and positive finite sigma");
}
std::vector<double> gaussian_weights(Filter f) {
    f.validate(); std::vector<double> w(2*f.radius+1); double sum=0;
    for(int i=-f.radius;i<=f.radius;++i) { double z=i/f.sigma; w[i+f.radius]=std::exp(-0.5*z*z); sum+=w[i+f.radius]; }
    for(double& v:w) v/=sum;
    return w;
}
Image render(const Grid& g,std::span<const double> conc,double timestamp,std::size_t padding) {
    validate_concentration(g,conc);
    require(padding<=4096,"image padding exceeds supported limit");
    require(std::isfinite(timestamp) && timestamp>=0,"invalid timestamp");
    Image image{g.width,g.height,(std::size_t(g.width)+padding)*sizeof(double),timestamp,{}};
    image.pixels.assign((std::size_t(g.width)+padding)*g.height,std::numeric_limits<double>::quiet_NaN());
    for(int y=0;y<g.height;++y) for(int x=0;x<g.width;++x) {
        double value=0.1+0.8*conc[std::size_t(y)*g.width+x];
        require(std::isfinite(value),"measurement overflow"); image.pixels[std::size_t(y)*(g.width+padding)+x]=value;
    } return image;
}
Snapshot reconstruct(const Grid& g,const Image& image,Filter f,Backend backend) {
    image.validate(); f.validate(); g.validate();
    require(g.width==image.width && g.height==image.height,"image/domain mismatch");
    std::vector<double> out(g.size());
    if(backend==Backend::cuda) {
#ifdef HZL_HAS_CUDA
        out=reconstruct_cuda(image,f);
#else
        throw std::runtime_error("CUDA backend not built");
#endif
    } else {
        auto stride=image.stride_bytes/sizeof(double); auto w=gaussian_weights(f);
        // Filter after decoding, avoiding cancellation and preserving zero background.
        std::vector<double> decoded(g.size());
        for(int y=0;y<g.height;++y) for(int x=0;x<g.width;++x) decoded[std::size_t(y)*g.width+x]=(image.pixels[y*stride+x]-0.1)/0.8;
        if(f.radius==0) out=std::move(decoded);
#ifdef HZL_OPENCV_IMAGE
        else if(!f.reference) {
            cv::Mat input(g.height,g.width,CV_64F,decoded.data()),filtered;
            cv::GaussianBlur(input,filtered,{2*f.radius+1,2*f.radius+1},f.sigma,f.sigma,cv::BORDER_REPLICATE);
            for(int y=0;y<g.height;++y) std::copy_n(filtered.ptr<double>(y),g.width,out.data()+std::size_t(y)*g.width);
        }
#endif
        else for(int y=0;y<g.height;++y) for(int x=0;x<g.width;++x) {
            double sum=0;
            for(int j=-f.radius;j<=f.radius;++j) for(int i=-f.radius;i<=f.radius;++i) {
                int xx=std::clamp(x+i,0,g.width-1), yy=std::clamp(y+j,0,g.height-1);
                sum+=w[i+f.radius]*w[j+f.radius]*decoded[std::size_t(yy)*g.width+xx];
            } out[std::size_t(y)*g.width+x]=sum;
        }
    }
    validate_concentration(g,out); return {g,image.timestamp,std::move(out)};
}
Forecast forecast(const Snapshot& s,const Physics& p,std::span<const double> temp,double horizon,Backend backend) {
    require(std::isfinite(s.timestamp) && s.timestamp>=0,"invalid snapshot timestamp");
    require(std::isfinite(horizon) && horizon>=0 && std::isfinite(s.timestamp+horizon),"invalid forecast horizon");
    auto out=solve(s.grid,p,s.concentration,temp,horizon,backend);
    return {s.timestamp,s.timestamp+horizon,std::move(out)};
}
Process::Process(Grid g,Physics p,std::vector<double> conc,std::vector<double> temp,Backend b)
    :grid_(g),physics_(p),backend_(b),conc_(std::move(conc)),temp_(std::move(temp)) {
    validate_concentration(grid_,conc_); plan(grid_,physics_,temp_,0);
}
void Process::apply_temperature(std::span<const double> temp) { plan(grid_,physics_,temp,0); temp_.assign(temp.begin(),temp.end()); }
void Process::advance(double seconds) {
    require(std::isfinite(seconds) && seconds>0 && std::isfinite(timestamp_+seconds) && timestamp_+seconds>timestamp_,"process interval must advance finite time");
    auto next=solve(grid_,physics_,conc_,temp_,seconds,backend_); conc_.swap(next); timestamp_+=seconds;
}
Image Process::observe(std::size_t padding) const { return render(grid_,conc_,timestamp_,padding); }
}
