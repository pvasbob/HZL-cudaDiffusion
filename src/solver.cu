#include "cuda_stencil.cuh"
#include "hzl/model.hpp"
#include <sstream>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>
namespace hzl {
using namespace cuda_detail;
bool cuda_available() { int n=0; return cudaGetDeviceCount(&n)==cudaSuccess && n>0; }
std::string cuda_device_info() {
    int count=0; auto status=cudaGetDeviceCount(&count);
    if(status!=cudaSuccess) return std::string("unavailable: ")+cudaGetErrorString(status);
    if(count==0) return "unavailable: no CUDA devices";
    int device=0,driver=0,runtime=0; check(cudaGetDevice(&device));
    cudaDeviceProp properties{}; check(cudaGetDeviceProperties(&properties,device));
    check(cudaDriverGetVersion(&driver)); check(cudaRuntimeGetVersion(&runtime));
    std::ostringstream info; info<<properties.name<<"; compute="<<properties.major<<'.'<<properties.minor
        <<"; memory_bytes="<<properties.totalGlobalMem<<"; driver="<<driver<<"; runtime="<<runtime;
    return info.str();
}
std::vector<double> advance_cuda(const Grid& g,const Physics& p,std::span<const double> conc,std::span<const double> temp,double seconds,double* device_ms,double max_dt) {
    validate_concentration(g,conc); auto r=plan(g,p,temp,seconds,max_dt);
    if(!cuda_available()) throw std::runtime_error("CUDA device unavailable; no CPU fallback performed");
    Buffer a(g.size()),b(g.size()),d(g.size()); Stream stream; Event start,stop;
    auto bytes=g.size()*sizeof(double);
    check(cudaMemcpyAsync(a.p,conc.data(),bytes,cudaMemcpyHostToDevice,stream.s));
    check(cudaMemcpyAsync(d.p,r.diffusivity.data(),bytes,cudaMemcpyHostToDevice,stream.s));
    check(cudaEventRecord(start.e,stream.s));
    dim3 block(32,8),grid((g.width+31)/32,(g.height+7)/8);
    for(std::size_t s=0;s<r.steps;++s) {
        diffusion_global<<<grid,block,0,stream.s>>>(a.p,b.p,d.p,g.width,g.height,1/(g.dx()*g.dx()),1/(g.dy()*g.dy()),r.dt);
        check(cudaGetLastError()); std::swap(a.p,b.p);
    }
    check(cudaEventRecord(stop.e,stream.s));
    std::vector<double> out(g.size());
    check(cudaMemcpyAsync(out.data(),a.p,bytes,cudaMemcpyDeviceToHost,stream.s));
    check(cudaStreamSynchronize(stream.s));
    float ms=0; check(cudaEventElapsedTime(&ms,start.e,stop.e)); if(device_ms) *device_ms=ms;
    validate_concentration(g,out); return out;
}
}
