#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>
namespace hzl::cuda_detail {
inline void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(e)); }
struct Buffer {
    double* p=nullptr;
    explicit Buffer(std::size_t n,std::size_t* allocations=nullptr) {
        if(n) { check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(double))); if(allocations) ++*allocations; }
    }
    ~Buffer() { if(p) cudaFree(p); }
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
};
struct PinnedBuffer {
    double* p=nullptr;
    explicit PinnedBuffer(std::size_t n) { if(n) check(cudaMallocHost(reinterpret_cast<void**>(&p),n*sizeof(double))); }
    ~PinnedBuffer() { if(p) cudaFreeHost(p); }
    PinnedBuffer(const PinnedBuffer&)=delete;
    PinnedBuffer& operator=(const PinnedBuffer&)=delete;
};
struct Stream {
    cudaStream_t s{};
    Stream() { check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(s); }
    Stream(const Stream&)=delete; Stream& operator=(const Stream&)=delete;
};
struct Event {
    cudaEvent_t e{};
    Event() { check(cudaEventCreate(&e)); }
    ~Event() { cudaEventDestroy(e); }
    Event(const Event&)=delete; Event& operator=(const Event&)=delete;
};
}
