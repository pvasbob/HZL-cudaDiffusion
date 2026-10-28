#include "hzl/cuda_forecast.hpp"
#include "hzl/control.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
 try {
    if(!hzl::cuda_available()) { std::cout<<"SKIP: CUDA device unavailable\n"; return 77; }
    for(auto dims:{std::pair{2,3},std::pair{37,23},std::pair{64,33}}) {
        hzl::Grid g{dims.first,dims.second,0.001,0.001}; hzl::Physics p;
        auto c=hzl::pattern(g,hzl::Pattern::deposits); c[0]=1; auto target=hzl::gaussian(g);
        std::vector<double> temp(g.size());
        for(std::size_t i=0;i<temp.size();++i) temp[i]=700+400*double(i%g.width)/(g.width-1);
        for(auto kernel:{hzl::CudaKernel::global,hzl::CudaKernel::tiled}) for(bool graphs:{false,true}) for(bool optimized:{false,true}) {
            hzl::CudaForecaster workspace({g,2,c},target,kernel,graphs,optimized);
            for(double duration:{0.,1.,1.,5.,2.}) {
                auto expected=hzl::advance_cpu(g,p,c,temp,duration);
                double loss=workspace.evaluate(p,temp,duration); auto out=workspace.download();
                if(std::abs(loss-hzl::mse(expected,target))>1e-13) throw std::runtime_error("CUB MSE mismatch");
                for(std::size_t i=0;i<out.size();++i) if(std::abs(out[i]-expected[i])>2e-13) throw std::runtime_error("stencil/graph mismatch");
            }
            if(optimized) {
                std::vector<double> coefficients{750,850,1000,1100};
                auto expanded=hzl::expand_tiles(g,2,2,coefficients);
                auto reference=hzl::advance_cpu(g,p,c,expanded,3);
                auto allocations=workspace.stats().device_allocations;
                auto bytes_before=workspace.stats().control_upload_bytes;
                double loss=workspace.evaluate_tiles(p,coefficients,2,2,3);
                if(std::abs(loss-hzl::mse(reference,target))>1e-13) throw std::runtime_error("coefficient objective mismatch");
                if(workspace.stats().control_upload_bytes-bytes_before!=coefficients.size()*sizeof(double)) throw std::runtime_error("full map uploaded for tile candidate");
                workspace.retain();
                workspace.evaluate(p,temp,5);
                auto kept=workspace.download_retained();
                for(std::size_t i=0;i<kept.size();++i) if(std::abs(kept[i]-reference[i])>2e-13) throw std::runtime_error("retained candidate overwritten");
                auto target_bytes=workspace.stats().target_upload_bytes;
                workspace.update({g,3,reference},target);
                if(workspace.stats().target_upload_bytes!=target_bytes) throw std::runtime_error("unchanged target reuploaded");
                bool rejected=false;
                try { workspace.download_retained(); } catch(const std::logic_error&) { rejected=true; }
                if(!rejected) throw std::runtime_error("stale retained forecast survived snapshot update");
                auto evolved=hzl::advance_cpu(g,p,reference,temp,1);
                workspace.evaluate(p,temp,1); auto next=workspace.download();
                for(std::size_t i=0;i<next.size();++i) if(std::abs(next[i]-evolved[i])>2e-13) throw std::runtime_error("cross-decision state mismatch");
                auto changed_target=target; changed_target[0]+=0.25;
                workspace.update({g,4,evolved},changed_target);
                double changed=workspace.evaluate(p,temp,0);
                if(std::abs(changed-hzl::mse(evolved,changed_target))>1e-13) throw std::runtime_error("changed target cache was stale");
                if(workspace.stats().target_upload_bytes-target_bytes!=g.size()*sizeof(double)) throw std::runtime_error("changed target upload missing");
                if(workspace.stats().device_allocations!=allocations) throw std::runtime_error("per-candidate/decision device allocation");
                bool invalid=false;
                try { workspace.evaluate_tiles(p,coefficients,3,2,1); } catch(const std::invalid_argument&) { invalid=true; }
                if(!invalid) throw std::runtime_error("invalid tile shape accepted");
                if(g.width>=8 && g.height>=8) {
                    std::vector<double> many(64);
                    for(std::size_t i=0;i<many.size();++i) many[i]=700+6*i;
                    auto dense=hzl::expand_tiles(g,8,8,many);
                    auto dense_reference=hzl::advance_cpu(g,p,evolved,dense,1);
                    workspace.evaluate_tiles(p,many,8,8,1); auto dense_result=workspace.download();
                    for(std::size_t i=0;i<dense_result.size();++i) if(std::abs(dense_result[i]-dense_reference[i])>2e-13) throw std::runtime_error("64-coefficient staging or mapping mismatch");
                }
                auto wrong=g; wrong.lx*=2; invalid=false;
                try { workspace.update({wrong,4,evolved},changed_target); } catch(const std::invalid_argument&) { invalid=true; }
                if(!invalid) throw std::runtime_error("workspace grid change accepted");
                auto bad_control=coefficients; bad_control[0]=0; invalid=false;
                try { workspace.evaluate_tiles(p,bad_control,2,2,1); } catch(const std::invalid_argument&) { invalid=true; }
                if(!invalid) throw std::runtime_error("invalid optimized control accepted");
                invalid=false;
                try { workspace.evaluate_tiles(p,coefficients,2,2,-1); } catch(const std::invalid_argument&) { invalid=true; }
                if(!invalid) throw std::runtime_error("negative optimized horizon accepted");
            }
        }
    }
    std::cout<<"PASS: global/tiled stencils, graphs/ordinary launches, CUB MSE, resident candidate reset, zero steps, graph reuse/rebuild, optimized/fused reduction, coefficient uploads, retained isolation, cross-decision workspace reuse, target cache\n";
 } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
