#include "hzl/experiment.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <initializer_list>
void expect(bool b,const char* s) { if(!b) throw std::runtime_error(s); }
template<class F> void rejects(F f) { bool caught=false; try { f(); } catch(const std::invalid_argument&) { caught=true; } expect(caught,"invalid control configuration accepted"); }
int main(int argc,char**) {
 try {
    auto parse=[](std::initializer_list<std::string> words) {
        std::vector<std::string> storage(words); std::vector<char*> arguments;
        for(auto& word:storage) arguments.push_back(word.data());
        return hzl::parse_settings(static_cast<int>(arguments.size()),arguments.data());
    };
    expect(!parse({"test","--backend","cpu"}).control.cuda_resident,"CPU default enabled GPU residency");
    rejects([&]{ parse({"test","--backend","cpu","--cuda-resident","1"}); });
#ifdef HZL_HAS_CUDA
    expect(parse({"test","--backend","cuda"}).control.cuda_resident,"CUDA CLI did not default to residency");
    expect(!parse({"test","--cuda-resident","0","--backend","cuda"}).control.cuda_resident,"legacy override before backend ignored");
    expect(!parse({"test","--backend","cuda","--cuda-resident","0"}).control.cuda_resident,"legacy override after backend ignored");
    expect(parse({"test","--backend","cuda","--cuda-resident","0","--cuda-resident","1"}).control.cuda_resident,"last residency override ignored");
#endif
    hzl::Settings settings; settings.grid={33,25,0.001,0.001}; settings.endpoint=5; settings.interval=0.7; settings.control.evaluations=45;
#ifndef HZL_OPENCV_OPTIM
    settings.control.simplex=false;
#endif
#ifdef HZL_HAS_CUDA
    if(argc>1) { if(!hzl::cuda_available()) { std::cout<<"SKIP: CUDA device unavailable\n"; return 77; } settings.backend=hzl::Backend::cuda; }
#else
    (void)argc;
#endif
    if(argc>2) settings.control.cuda_resident=true;
    hzl::Experiment experiment(settings); auto original=std::vector<double>(experiment.truth().begin(),experiment.truth().end());
    auto snapshot=experiment.snapshot;
    auto decision=hzl::optimize(snapshot,settings.physics,experiment.target,experiment.temp,settings.control,settings.backend);
    expect(experiment.time()==0 && std::equal(original.begin(),original.end(),experiment.truth().begin()),"optimization changed process");
    expect(decision.evaluations<=settings.control.evaluations && decision.evaluations>0,"forecast budget exceeded");
    expect(decision.objective<=decision.initial_objective,"optimizer discarded better initial candidate");
    expect(decision.prediction.start_time==0 && decision.prediction.endpoint_time==5,"endpoint forecast semantics failed");
    for(double t:decision.temp) expect(t>=settings.control.lower && t<=settings.control.upper,"temperature constraints violated");
    expect(decision.objective==decision.target_mse,"disabled penalties changed objective");
    auto penalized=settings.control; penalized.simplex=false; penalized.evaluations=12;
    penalized.heating_penalty=100;
    auto cold=hzl::optimize(snapshot,settings.physics,experiment.target,experiment.temp,penalized,settings.backend);
    expect(std::all_of(cold.temp.begin(),cold.temp.end(),[&](double value){return value==penalized.lower;}),"heating penalty did not prefer cold control");
    penalized.heating_penalty=0; penalized.change_penalty=100;
    auto unchanged=hzl::optimize(snapshot,settings.physics,experiment.target,experiment.temp,penalized,settings.backend);
    expect(unchanged.temp==experiment.temp,"change penalty did not preserve applied control");
    auto tiny=settings.control; tiny.evaluations=1;
    auto one=hzl::optimize(snapshot,settings.physics,experiment.target,experiment.temp,tiny,settings.backend);
    expect(one.evaluations==1 && one.stopping_reason=="evaluation_budget","small budget handling failed");
    while(!experiment.finished()) {
        auto time=experiment.time(); auto cycle=experiment.step(); expect(cycle.timestamp==time && cycle.next_timestamp>time,"nonchronological cycle");
        expect(cycle.reconstruction_rmse<1e-15,"control reconstruction bypassed or inaccurate");
        expect(std::abs(cycle.mass_error)<1e-12,"closed-loop mass conservation failed");
        expect(cycle.evaluations<=settings.control.evaluations,"cycle budget exceeded");
        if(settings.control.cuda_resident) {
            expect(cycle.device_allocations==0,"resident cycle allocated forecasting device buffers");
            expect(cycle.forecast_download_bytes==settings.grid.size()*sizeof(double),"resident cycle downloaded multiple candidate fields");
            expect(cycle.target_upload_bytes==0,"resident cycle reuploaded unchanged target");
            expect(cycle.control_upload_bytes<settings.grid.size()*sizeof(double)*cycle.evaluations,"coefficient transfer path unused");
        }
    }
    expect(experiment.time()==settings.endpoint,"partial last interval overshot endpoint");
    auto last=experiment.cycles.back(); expect(last.target_mse<last.baseline_mse*0.02,"control failed to improve fixed baseline");
    rejects([&]{ experiment.step(); });
    experiment.reset(); expect(experiment.time()==0 && experiment.cycles.empty(),"reset failed");
    expect(std::equal(original.begin(),original.end(),experiment.truth().begin()),"reset did not restore initial state");
    auto unreachable=experiment.target; for(auto& v:unreachable.concentration) v+=0.5;
    auto bad=hzl::optimize(experiment.snapshot,settings.physics,unreachable,experiment.temp,tiny,settings.backend);
    expect(bad.target_mse>=0.25-1e-12,"unreachable mass mismatch hidden");
    auto stress=settings; stress.endpoint=1e-10; stress.interval=1e-11; stress.control.evaluations=1;
    hzl::Experiment deadlines(stress); expect(deadlines.step().deadline_miss,"subnanosecond deadline miss was hidden");
    auto invalid=settings; invalid.control.tiles_x=100; rejects([&]{ invalid.validate(); });
    invalid=settings; invalid.interval=0; rejects([&]{ invalid.validate(); });
    invalid=settings; invalid.backend=hzl::Backend::cpu; invalid.control.cuda_resident=true; rejects([&]{ invalid.validate(); });
    invalid=settings; invalid.initial_temp=100; rejects([&]{ invalid.validate(); });
    auto tile=hzl::expand_tiles({5,3,1,1},2,1,std::vector<double>{700,900}); expect(tile[0]==700 && tile[4]==900,"rectangular tile expansion failed");
    std::cout<<"PASS: endpoint semantics, isolation, bounds, budgets, chronological closed-loop improvement, reset, unreachable target, invalid configuration; final MSE="<<last.target_mse<<" baseline="<<last.baseline_mse<<'\n';
 } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
