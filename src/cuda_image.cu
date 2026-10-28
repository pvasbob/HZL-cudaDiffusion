#include "hzl/observation.hpp"
#include "cuda_resources.cuh"

namespace hzl {
using namespace cuda_detail;
namespace {
__global__ void decode_filter(const double* image,double* out,int w,int h,const double* weights,int radius) {
    // Cooperative tile including a replicated-boundary halo. Every thread
    // participates in the barrier, including threads outside the output domain.
    extern __shared__ double tile[];
    int tw=blockDim.x+2*radius, th=blockDim.y+2*radius;
    int lane=threadIdx.y*blockDim.x+threadIdx.x;
    for(int i=lane;i<tw*th;i+=blockDim.x*blockDim.y) {
        int x=int(blockIdx.x*blockDim.x)+i%tw-radius;
        int y=int(blockIdx.y*blockDim.y)+i/tw-radius;
        x=max(0,min(w-1,x)); y=max(0,min(h-1,y));
        tile[i]=(image[y*w+x]-0.1)/0.8;
    }
    __syncthreads();
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w || y>=h) return;
    double sum=0;
    for(int j=0;j<2*radius+1;++j) for(int i=0;i<2*radius+1;++i)
        sum+=weights[i]*weights[j]*tile[(threadIdx.y+j)*tw+threadIdx.x+i];
    out[y*w+x]=sum;
}
}
std::vector<double> reconstruct_cuda(const Image& image,Filter f) {
    image.validate(); f.validate();
    if(!cuda_available()) throw std::runtime_error("CUDA device unavailable; no CPU fallback performed");
    auto weights=gaussian_weights(f); std::size_t n=std::size_t(image.width)*image.height;
    Buffer input(n),output(n),kernel(weights.size()); Stream stream;
    check(cudaMemcpy2DAsync(input.p,image.width*sizeof(double),image.pixels.data(),image.stride_bytes,
          image.width*sizeof(double),image.height,cudaMemcpyHostToDevice,stream.s));
    check(cudaMemcpyAsync(kernel.p,weights.data(),weights.size()*sizeof(double),cudaMemcpyHostToDevice,stream.s));
    dim3 block(32,8),grid((image.width+31)/32,(image.height+7)/8);
    std::size_t shared=(32+2*f.radius)*(8+2*f.radius)*sizeof(double);
    decode_filter<<<grid,block,shared,stream.s>>>(input.p,output.p,image.width,image.height,kernel.p,f.radius);
    check(cudaGetLastError()); std::vector<double> out(n);
    check(cudaMemcpyAsync(out.data(),output.p,n*sizeof(double),cudaMemcpyDeviceToHost,stream.s));
    check(cudaStreamSynchronize(stream.s)); return out;
}
}
