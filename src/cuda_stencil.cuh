#pragma once
#include "cuda_resources.cuh"
namespace hzl::cuda_detail {
static __global__ void diffusion_global(const double* a,double* b,const double* d,int w,int h,double ix,double iy,double dt) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w || y>=h) return;
    int i=y*w+x; double f=0;
    if(x>0) f+=(0.5*d[i]+0.5*d[i-1])*(a[i-1]-a[i])*ix;
    if(x+1<w) f+=(0.5*d[i]+0.5*d[i+1])*(a[i+1]-a[i])*ix;
    if(y>0) f+=(0.5*d[i]+0.5*d[i-w])*(a[i-w]-a[i])*iy;
    if(y+1<h) f+=(0.5*d[i]+0.5*d[i+w])*(a[i+w]-a[i])*iy;
    b[i]=a[i]+dt*f;
}
}
