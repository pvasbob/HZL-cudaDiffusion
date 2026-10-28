#include "hzl/cuda_forecast.hpp"
#include "cuda_stencil.cuh"
#include <cub/device/device_reduce.cuh>
#include <cmath>
#include <algorithm>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <utility>

namespace hzl {
using namespace cuda_detail;
namespace {
__global__ void tiled_step(const double* a,double* b,const double* d,int w,int h,double ix,double iy,double dt) {
    // 32x8 interior, one-cell halo; two shared planes, no unnecessary padding.
    __shared__ double c[10][34];
    __shared__ double diffusion[10][34];
    int lane=threadIdx.y*32+threadIdx.x;
    for(int i=lane;i<340;i+=256) {
        int x=int(blockIdx.x*32)+i%34-1,y=int(blockIdx.y*8)+i/34-1;
        x=max(0,min(w-1,x)); y=max(0,min(h-1,y));
        c[i/34][i%34]=a[y*w+x]; diffusion[i/34][i%34]=d[y*w+x];
    }
    __syncthreads();
    int x=blockIdx.x*32+threadIdx.x,y=blockIdx.y*8+threadIdx.y;
    if(x>=w || y>=h) return;
    int tx=threadIdx.x+1,ty=threadIdx.y+1; double v=c[ty][tx],dd=diffusion[ty][tx],f=0;
    if(x>0) f+=(0.5*dd+0.5*diffusion[ty][tx-1])*(c[ty][tx-1]-v)*ix;
    if(x+1<w) f+=(0.5*dd+0.5*diffusion[ty][tx+1])*(c[ty][tx+1]-v)*ix;
    if(y>0) f+=(0.5*dd+0.5*diffusion[ty-1][tx])*(c[ty-1][tx]-v)*iy;
    if(y+1<h) f+=(0.5*dd+0.5*diffusion[ty+1][tx])*(c[ty+1][tx]-v)*iy;
    b[y*w+x]=v+dt*f;
}
__global__ void squared_errors(const double* c,const double* target,double* error,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) { double d=c[i]-target[i]; error[i]=d*d; }
}
struct SquaredDifference {
    const double* state; const double* target;
    __host__ __device__ double operator()(int i) const { double delta=state[i]-target[i]; return delta*delta; }
};
using ErrorIterator=cub::TransformInputIterator<double,SquaredDifference,cub::CountingInputIterator<int>>;
__global__ void prepare_diffusivity(const double* temp,double* d,int width,int height,int nx,int ny,double d0,double ea) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=width*height) return;
    int control=nx ? ((i/width)*ny/height)*nx+(i%width)*nx/width : i;
    d[i]=d0*exp(-ea/(8.617333262145e-5*temp[control]));
}
struct Graph {
    cudaGraph_t graph{}; cudaGraphExec_t executable{};
    ~Graph() { clear(); }
    void clear() { if(executable) cudaGraphExecDestroy(executable); if(graph) cudaGraphDestroy(graph); executable=nullptr; graph=nullptr; }
};
}
struct CudaForecaster::Impl {
    Grid grid;
    mutable CudaForecastStats counters;
    Buffer initial,a,b,d,target,error,sum,control,best;
    PinnedBuffer scalar,coefficients;
    void* scratch=nullptr;
    std::size_t scratch_bytes=0;
    Stream stream; Event start,stop; Graph graph;
    CudaKernel kernel;
    bool graphs,optimized,ready=false,best_ready=false;
    std::size_t cached_steps=0;
    double cached_dt=-1;
    double* result=nullptr;
    double milliseconds=0;
    std::vector<double> cached_target;

    Impl(const Grid& g,CudaKernel k,bool graph_mode,bool optimized_mode)
        :grid(g),initial(g.size(),&counters.device_allocations),a(g.size(),&counters.device_allocations),
         b(g.size(),&counters.device_allocations),d(g.size(),&counters.device_allocations),
         target(g.size(),&counters.device_allocations),error(optimized_mode?0:g.size(),&counters.device_allocations),
         sum(1,&counters.device_allocations),control(optimized_mode?g.size():0,&counters.device_allocations),
         best(optimized_mode?g.size():0,&counters.device_allocations),
         scalar(optimized_mode?1:0),coefficients(optimized_mode?64:0),
         kernel(k),graphs(graph_mode),optimized(optimized_mode) {
        try {
            reduce(nullptr,scratch_bytes,a.p);
            check(cudaMalloc(&scratch,scratch_bytes)); ++counters.device_allocations;
        } catch(...) { if(scratch) cudaFree(scratch); throw; }
    }
    ~Impl() { cudaStreamSynchronize(stream.s); if(scratch) cudaFree(scratch); }
    void reduce(void* storage,std::size_t& bytes,const double* state) {
        if(optimized) {
            ErrorIterator input(cub::CountingInputIterator<int>(0),SquaredDifference{state,target.p});
            check(cub::DeviceReduce::Sum(storage,bytes,input,sum.p,int(grid.size()),stream.s));
        } else check(cub::DeviceReduce::Sum(storage,bytes,error.p,sum.p,int(grid.size()),stream.s));
    }
    void launch(std::size_t steps,double dt) {
        auto in=a.p,out=b.p; dim3 block(32,8),blocks((grid.width+31)/32,(grid.height+7)/8);
        for(std::size_t i=0;i<steps;++i) {
            if(kernel==CudaKernel::global) diffusion_global<<<blocks,block,0,stream.s>>>(in,out,d.p,grid.width,grid.height,1/(grid.dx()*grid.dx()),1/(grid.dy()*grid.dy()),dt);
            else tiled_step<<<blocks,block,0,stream.s>>>(in,out,d.p,grid.width,grid.height,1/(grid.dx()*grid.dx()),1/(grid.dy()*grid.dy()),dt);
            check(cudaGetLastError()); std::swap(in,out);
        }
        result=in;
    }
    double integrate(Schedule integration) {
        if(graphs && integration.steps>100000) throw std::invalid_argument("CUDA graph exceeds 100000-node integration budget");
        check(cudaMemcpyAsync(a.p,initial.p,grid.size()*sizeof(double),cudaMemcpyDeviceToDevice,stream.s));
        check(cudaEventRecord(start.e,stream.s));
        if(graphs && integration.steps>0) {
            if(!graph.executable || integration.steps!=cached_steps || integration.dt!=cached_dt) {
                graph.clear();
                check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));
                try { launch(integration.steps,integration.dt); }
                catch(...) { cudaGraph_t abandoned{}; cudaStreamEndCapture(stream.s,&abandoned); if(abandoned) cudaGraphDestroy(abandoned); throw; }
                check(cudaStreamEndCapture(stream.s,&graph.graph));
                check(cudaGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0));
                cached_steps=integration.steps; cached_dt=integration.dt; ++counters.graph_rebuilds;
            }
            check(cudaGraphLaunch(graph.executable,stream.s));
            result=integration.steps%2 ? b.p:a.p;
        } else launch(integration.steps,integration.dt);
        if(!optimized) {
            squared_errors<<<(grid.size()+255)/256,256,0,stream.s>>>(result,target.p,error.p,int(grid.size()));
            check(cudaGetLastError());
        }
        reduce(scratch,scratch_bytes,result);
        check(cudaEventRecord(stop.e,stream.s));
        double stack_sum=0; double* host_sum=optimized ? scalar.p : &stack_sum;
        check(cudaMemcpyAsync(host_sum,sum.p,sizeof(double),cudaMemcpyDeviceToHost,stream.s));
        check(cudaStreamSynchronize(stream.s));
        float ms=0; check(cudaEventElapsedTime(&ms,start.e,stop.e)); milliseconds=ms;
        ++counters.evaluations;
        double loss=*host_sum/grid.size();
        if(!std::isfinite(loss) || loss<0) throw std::runtime_error("nonfinite GPU objective");
        return loss;
    }
    Schedule prepare(const Physics& p,std::span<const double> temp,int nx,int ny,double horizon) {
        p.validate();
        if(temp.empty()) throw std::invalid_argument("empty control");
        double minimum=p.max_temp,maximum=p.min_temp;
        for(double t:temp) {
            if(!std::isfinite(t) || t<p.min_temp || t>p.max_temp) throw std::invalid_argument("temperature outside Kelvin bounds");
            minimum=std::min(minimum,t); maximum=std::max(maximum,t);
        }
        p.diffusivity(minimum); // Reject underflow exactly as the CPU reference does.
        auto integration=schedule(grid,p.diffusivity(maximum),horizon);
        const double* source=temp.data();
        if(nx) { std::copy(temp.begin(),temp.end(),coefficients.p); source=coefficients.p; }
        check(cudaMemcpyAsync(control.p,source,temp.size_bytes(),cudaMemcpyHostToDevice,stream.s));
        counters.control_upload_bytes+=temp.size_bytes();
        prepare_diffusivity<<<(grid.size()+255)/256,256,0,stream.s>>>(control.p,d.p,grid.width,grid.height,nx,ny,p.d0,p.ea);
        check(cudaGetLastError()); return integration;
    }
    std::vector<double> download(const double* field) const {
        std::vector<double> out(grid.size());
        check(cudaMemcpyAsync(out.data(),field,out.size()*sizeof(double),cudaMemcpyDeviceToHost,stream.s));
        check(cudaStreamSynchronize(stream.s)); counters.forecast_download_bytes+=out.size()*sizeof(double);
        validate_concentration(grid,out); return out;
    }
};
CudaForecaster::CudaForecaster(const Snapshot& s,std::span<const double> target,CudaKernel k,bool graphs,bool optimized) {
    validate_concentration(s.grid,s.concentration); validate_concentration(s.grid,target);
    if(!cuda_available()) throw std::runtime_error("CUDA device unavailable; no CPU fallback performed");
    impl_=std::make_unique<Impl>(s.grid,k,graphs,optimized); update(s,target);
}
CudaForecaster::~CudaForecaster()=default;
void CudaForecaster::update(const Snapshot& s,std::span<const double> target) {
    auto& w=*impl_;
    if(s.grid.width!=w.grid.width || s.grid.height!=w.grid.height || s.grid.lx!=w.grid.lx || s.grid.ly!=w.grid.ly)
        throw std::invalid_argument("workspace domain mismatch");
    validate_concentration(s.grid,s.concentration); validate_concentration(s.grid,target);
    if(!std::isfinite(s.timestamp) || s.timestamp<0) throw std::invalid_argument("invalid snapshot timestamp");
    w.ready=false; w.best_ready=false; w.result=nullptr;
    try {
        auto bytes=w.grid.size()*sizeof(double);
        check(cudaMemcpyAsync(w.initial.p,s.concentration.data(),bytes,cudaMemcpyHostToDevice,w.stream.s));
        w.counters.snapshot_upload_bytes+=bytes;
        if(w.cached_target.size()!=target.size() || !std::equal(target.begin(),target.end(),w.cached_target.begin())) {
            check(cudaMemcpyAsync(w.target.p,target.data(),bytes,cudaMemcpyHostToDevice,w.stream.s));
            w.counters.target_upload_bytes+=bytes;
            w.cached_target.assign(target.begin(),target.end());
        }
        check(cudaStreamSynchronize(w.stream.s)); w.ready=true;
    } catch(...) { cudaStreamSynchronize(w.stream.s); w.cached_target.clear(); throw; }
}
double CudaForecaster::evaluate(const Physics& p,std::span<const double> temp,double horizon) {
    auto& w=*impl_;
    try {
        if(!w.ready) throw std::logic_error("workspace needs a valid snapshot");
        if(temp.size()!=w.grid.size()) throw std::invalid_argument("temperature dimensions mismatch");
        if(w.optimized) return w.integrate(w.prepare(p,temp,0,0,horizon));
        auto integration=plan(w.grid,p,temp,horizon);
        check(cudaMemcpyAsync(w.d.p,integration.diffusivity.data(),temp.size_bytes(),cudaMemcpyHostToDevice,w.stream.s));
        w.counters.control_upload_bytes+=temp.size_bytes();
        return w.integrate({integration.steps,integration.dt});
    } catch(...) { cudaStreamSynchronize(w.stream.s); w.result=nullptr; throw; }
}
double CudaForecaster::evaluate_tiles(const Physics& p,std::span<const double> coefficients,int nx,int ny,double horizon) {
    auto& w=*impl_;
    if(!w.optimized) throw std::logic_error("coefficient controls require optimized workspace");
    if(nx<1 || ny<1 || nx>w.grid.width || ny>w.grid.height || nx*1LL*ny>64 || nx*1LL*ny!=static_cast<long long>(coefficients.size()))
        throw std::invalid_argument("invalid control tiles");
    try {
        if(!w.ready) throw std::logic_error("workspace needs a valid snapshot");
        return w.integrate(w.prepare(p,coefficients,nx,ny,horizon));
    } catch(...) { cudaStreamSynchronize(w.stream.s); w.result=nullptr; throw; }
}
void CudaForecaster::retain() {
    auto& w=*impl_;
    if(!w.optimized || !w.result) throw std::logic_error("no optimized result to retain");
    check(cudaMemcpyAsync(w.best.p,w.result,w.grid.size()*sizeof(double),cudaMemcpyDeviceToDevice,w.stream.s));
    w.best_ready=true; ++w.counters.retained_forecasts;
}
std::vector<double> CudaForecaster::download() const {
    const auto& w=*impl_;
    if(!w.result) throw std::logic_error("no forecast has been evaluated");
    return w.download(w.result);
}
std::vector<double> CudaForecaster::download_retained() const {
    const auto& w=*impl_;
    if(!w.best_ready) throw std::logic_error("no retained forecast");
    return w.download(w.best.p);
}
bool CudaForecaster::optimized() const { return impl_->optimized; }
CudaForecastStats CudaForecaster::stats() const { return impl_->counters; }
double CudaForecaster::device_ms() const { return impl_->milliseconds; }
}
