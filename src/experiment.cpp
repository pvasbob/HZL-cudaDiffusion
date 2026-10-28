#include "hzl/experiment.hpp"
#ifdef HZL_HAS_CUDA
#include "hzl/cuda_forecast.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
namespace hzl {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
void require(bool b,const char* s) { if(!b) throw std::invalid_argument(s); }
double number(const std::string& s) { std::size_t n; double v=std::stod(s,&n); require(n==s.size() && std::isfinite(v),"expected finite number"); return v; }
int integer(const std::string& s) { double v=number(s); require(v==std::floor(v) && v>=0 && v<=1000000000,"expected nonnegative integer"); return static_cast<int>(v); }
void set(Settings& s,const std::string& key,const std::string& value,bool& resident_explicit) {
    if(key=="width") s.grid.width=integer(value);
    else if(key=="height") s.grid.height=integer(value);
    else if(key=="lx") s.grid.lx=number(value);
    else if(key=="ly") s.grid.ly=number(value);
    else if(key=="temperature") s.temperature=value;
    else if(key=="d0") s.physics.d0=number(value);
    else if(key=="ea") s.physics.ea=number(value);
    else if(key=="endpoint") s.endpoint=number(value);
    else if(key=="interval") s.interval=number(value);
    else if(key=="horizon") s.horizon=number(value);
    else if(key=="temp") s.initial_temp=number(value);
    else if(key=="lower") s.control.lower=number(value);
    else if(key=="upper") s.control.upper=number(value);
    else if(key=="budget") s.control.evaluations=integer(value);
    else if(key=="tiles-x") s.control.tiles_x=integer(value);
    else if(key=="tiles-y") s.control.tiles_y=integer(value);
    else if(key=="change-penalty") s.control.change_penalty=number(value);
    else if(key=="heating-penalty") s.control.heating_penalty=number(value);
    else if(key=="spatial-penalty") s.control.spatial_penalty=number(value);
    else if(key=="tolerance") s.control.tolerance=number(value);
    else if(key=="seed") s.seed=static_cast<unsigned>(integer(value));
    else if(key=="filter-radius") s.filter.radius=integer(value);
    else if(key=="sigma") s.filter.sigma=number(value);
    else if(key=="output") { require(!value.empty(),"empty output directory"); s.output=value; }
    else if(key=="cuda-resident") { require(value=="0" || value=="1","cuda-resident must be 0 or 1"); s.control.cuda_resident=value=="1"; resident_explicit=true; }
    else if(key=="cuda-kernel") { require(value=="global" || value=="tiled","cuda-kernel must be global or tiled"); s.control.cuda_tiled=value=="tiled"; }
    else if(key=="cuda-graphs") { require(value=="0" || value=="1","cuda-graphs must be 0 or 1"); s.control.cuda_graphs=value=="1"; }
    else if(key=="method") { require(value=="simplex" || value=="finite","method must be simplex or finite"); s.control.simplex=value=="simplex"; }
    else if(key=="backend") { require(value=="cpu" || value=="cuda","backend must be cpu or cuda"); s.backend=value=="cuda"?Backend::cuda:Backend::cpu; }
    else if(key=="pattern") {
        if(value=="gaussian") s.preset=Pattern::gaussian; else if(value=="deposits") s.preset=Pattern::deposits;
        else if(value=="line") s.preset=Pattern::line; else if(value=="circle") s.preset=Pattern::circle; else if(value=="rosette") s.preset=Pattern::rosette; else if(value=="network") s.preset=Pattern::network; else throw std::invalid_argument("unknown pattern");
    } else throw std::invalid_argument("unknown setting: "+key);
}
std::string trim(std::string s) { auto b=s.find_first_not_of(" \t\r\n"); if(b==std::string::npos) return {}; auto e=s.find_last_not_of(" \t\r\n"); return s.substr(b,e-b+1); }
}
void Settings::validate() const {
    grid.validate(); physics.validate(); control.validate(grid,physics); filter.validate();
    require(!control.cuda_resident || backend==Backend::cuda,"cuda-resident requires the CUDA backend");
    require(temperature=="uniform" || temperature=="gradient" || temperature=="hotspot" || temperature=="spots","unknown temperature preset");
    require(std::isfinite(endpoint) && endpoint>0 && endpoint<=86400,"endpoint must be in (0,86400] seconds");
    require(std::isfinite(interval) && interval>0 && endpoint/interval<=100000,"invalid control interval/cycle budget");
    require(std::isfinite(horizon) && horizon>0,"preview horizon must be positive finite seconds");
    require(std::isfinite(initial_temp) && initial_temp>=control.lower && initial_temp<=control.upper,"initial temperature outside control bounds");
#ifndef HZL_HAS_CUDA
    require(backend==Backend::cpu,"CUDA backend not built");
#endif
}
Settings parse_settings(int argc,char** argv) {
    Settings s; bool resident_explicit=false;
#ifndef HZL_OPENCV_OPTIM
    s.control.simplex=false;
#endif
    for(int i=1;i<argc;++i) {
        std::string arg=argv[i];
        if(arg=="--paced") { s.paced=true; continue; }
        require(arg.starts_with("--") && i+1<argc,"expected --setting value (see README)");
        auto value=std::string(argv[++i]);
        if(arg=="--config") {
            std::ifstream file(value); require(bool(file),"cannot open configuration"); std::string line;
            while(std::getline(file,line)) {
                line=trim(line.substr(0,line.find('#'))); if(line.empty()) continue;
                auto eq=line.find('='); require(eq!=std::string::npos,"configuration expects key=value"); set(s,trim(line.substr(0,eq)),trim(line.substr(eq+1)),resident_explicit);
            }
        } else set(s,arg.substr(2),value,resident_explicit);
    }
    if(!resident_explicit) s.control.cuda_resident=s.backend==Backend::cuda;
    s.validate(); return s;
}
Experiment::Experiment(Settings settings):settings_(std::move(settings)),initial_(pattern(settings_.grid,settings_.preset)) { settings_.validate(); reset(); }
Experiment::~Experiment()=default;
void Experiment::reset() {
    auto initialization_start=Clock::now();
    const auto& grid=settings_.grid;
    std::vector<double> new_temp(grid.size(),settings_.initial_temp);
    for(int y=0;y<grid.height;++y) for(int x=0;x<grid.width;++x) {
        double xx=(x+0.5)/grid.width, yy=(y+0.5)/grid.height;
        if(settings_.temperature=="gradient") new_temp[std::size_t(y)*grid.width+x]=settings_.control.lower+(settings_.control.upper-settings_.control.lower)*xx;
        else if((settings_.temperature=="hotspot" && std::hypot(xx-0.5,yy-0.5)<0.2) ||
                (settings_.temperature=="spots" && (std::hypot(xx-0.3,yy-0.35)<0.15 || std::hypot(xx-0.7,yy-0.65)<0.15)))
            new_temp[std::size_t(y)*grid.width+x]=std::min(settings_.control.upper,settings_.initial_temp+200);
    }
    // Construct a replacement completely before committing: failed reset leaves
    // the current process, target, snapshots and timeline available for inspection.
    auto new_target=reachable_target(grid,settings_.physics,initial_,settings_.endpoint,settings_.seed,settings_.backend,settings_.control.lower,settings_.control.upper);
    auto new_process=std::make_unique<Process>(grid,settings_.physics,initial_,new_temp,settings_.backend);
    auto new_baseline=std::make_unique<Process>(grid,settings_.physics,initial_,std::vector<double>(grid.size(),settings_.initial_temp),settings_.backend);
    auto new_image=new_process->observe(7);
    auto new_snapshot=reconstruct(grid,new_image,settings_.filter,settings_.backend);
    double new_mass=mass(grid,initial_);
#ifdef HZL_HAS_CUDA
    std::unique_ptr<CudaForecaster> new_workspace;
    if(settings_.control.cuda_resident) new_workspace=std::make_unique<CudaForecaster>(new_snapshot,new_target.concentration,settings_.control.cuda_tiled?CudaKernel::tiled:CudaKernel::global,settings_.control.cuda_graphs,true);
    cuda_workspace_=std::move(new_workspace);
#endif
    target=std::move(new_target); process_=std::move(new_process); baseline_=std::move(new_baseline);
    temp=std::move(new_temp); image=std::move(new_image); snapshot=std::move(new_snapshot);
    initial_mass_=new_mass; cycles.clear(); decision={}; prediction={}; initialization_ms_=elapsed(initialization_start);
}
void Experiment::replace_initial(std::vector<double> initial) {
    validate_concentration(settings_.grid,initial);
    auto previous=std::move(initial_); initial_=std::move(initial);
    try { reset(); } catch(...) { initial_=std::move(previous); throw; }
}
void Experiment::replace_target(std::vector<double> conc) { validate_concentration(settings_.grid,conc); target.concentration=std::move(conc); }
void Experiment::refresh() {
    auto next_image=process_->observe(7);
    auto next_snapshot=reconstruct(settings_.grid,next_image,settings_.filter,settings_.backend);
    image=std::move(next_image); snapshot=std::move(next_snapshot);
}
void Experiment::preview() { refresh(); prediction=forecast(snapshot,settings_.physics,temp,settings_.horizon,settings_.backend); }
bool Experiment::finished() const { return time()>=settings_.endpoint; }
Cycle Experiment::step(bool automatic) {
    require(!finished(),"experiment endpoint reached; reset or start a new experiment");
    auto start=Clock::now(); Cycle c; c.timestamp=time();
    auto stage=Clock::now(); image=process_->observe(7); c.image_ms=elapsed(stage);
    stage=Clock::now(); snapshot=reconstruct(settings_.grid,image,settings_.filter,settings_.backend); c.reconstruction_ms=elapsed(stage);
    if(automatic) {
        decision=optimize(snapshot,settings_.physics,target,temp,settings_.control,settings_.backend
#ifdef HZL_HAS_CUDA
                          ,cuda_workspace_.get()
#endif
                          );
        temp=decision.temp; prediction=decision.prediction; c.evaluations=decision.evaluations;
        c.control_upload_bytes=decision.control_upload_bytes; c.forecast_download_bytes=decision.forecast_download_bytes; c.device_allocations=decision.device_allocations;
        c.snapshot_upload_bytes=decision.snapshot_upload_bytes; c.target_upload_bytes=decision.target_upload_bytes; c.graph_rebuilds=decision.graph_rebuilds;
        c.forecast_ms=decision.forecast_ms; c.optimization_ms=decision.optimization_ms; c.status=decision.stopping_reason;
    } else {
        stage=Clock::now(); prediction=forecast(snapshot,settings_.physics,temp,settings_.endpoint-time(),settings_.backend); c.forecast_ms=elapsed(stage); c.evaluations=1; c.status="manual";
    }
    c.predicted_endpoint_mse=mse(prediction.concentration,target.concentration);
    c.reconstruction_rmse=std::sqrt(mse(snapshot.concentration,process_->concentration_for_validation()));
    stage=Clock::now(); process_->apply_temperature(temp); c.command_ms=elapsed(stage);
    double interval=std::min(settings_.interval,settings_.endpoint-time());
    auto integration=plan(settings_.grid,settings_.physics,temp,interval); c.solver_steps=integration.steps; c.solver_dt=integration.dt;
    stage=Clock::now(); process_->advance(interval); c.advance_ms=elapsed(stage); c.next_timestamp=time();
    c.cycle_ms=elapsed(start); c.deadline_miss=c.cycle_ms>1000*interval;
    // Baseline and validation/display work is deliberately outside control latency.
    baseline_->advance(interval);
    c.target_mse=mse(process_->concentration_for_validation(),target.concentration);
    c.baseline_mse=mse(baseline_->concentration_for_validation(),target.concentration);
    double final_mass=mass(settings_.grid,process_->concentration_for_validation());
    c.mass_error=initial_mass_>0 ? (final_mass-initial_mass_)/initial_mass_:final_mass;
    if(finished()) c.status+=";endpoint_reached";
    cycles.push_back(c); refresh();
    if(settings_.paced) {
        auto deadline=start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(interval));
        std::this_thread::sleep_until(deadline);
    }
    return c;
}
void Experiment::save() const {
    std::filesystem::create_directories(settings_.output);
    auto path=std::filesystem::path(settings_.output);
    std::ofstream log(path/"cycles.csv"),fields(path/"fields.csv"),meta(path/"metadata.txt");
    if(!log || !fields || !meta) throw std::runtime_error("cannot open experiment outputs");
    log<<std::setprecision(17)<<"observation_s,next_observation_s,reconstruction_rmse,target_mse,baseline_mse,mass_relative_error,image_ms,reconstruction_ms,forecast_total_ms,optimization_total_ms,command_ms,advance_ms,cycle_ms,evaluations,deadline_miss,status,predicted_endpoint_mse,solver_steps,solver_dt_s,control_upload_bytes,forecast_download_bytes,forecast_device_allocations,snapshot_upload_bytes,target_upload_bytes,graph_rebuilds\n";
    for(const auto& c:cycles) log<<c.timestamp<<','<<c.next_timestamp<<','<<c.reconstruction_rmse<<','<<c.target_mse<<','<<c.baseline_mse<<','<<c.mass_error<<','<<c.image_ms<<','<<c.reconstruction_ms<<','<<c.forecast_ms<<','<<c.optimization_ms<<','<<c.command_ms<<','<<c.advance_ms<<','<<c.cycle_ms<<','<<c.evaluations<<','<<c.deadline_miss<<','<<c.status<<','<<c.predicted_endpoint_mse<<','<<c.solver_steps<<','<<c.solver_dt<<','<<c.control_upload_bytes<<','<<c.forecast_download_bytes<<','<<c.device_allocations<<','<<c.snapshot_upload_bytes<<','<<c.target_upload_bytes<<','<<c.graph_rebuilds<<'\n';
    fields<<std::setprecision(17)<<"# timestamp_s="<<time()<<", endpoint_s="<<target.endpoint<<", forecast_start_s="<<prediction.start_time<<", forecast_endpoint_s="<<prediction.endpoint_time<<"\nx,y,underlying,reconstructed,temp_K,target,forecast,baseline\n";
    auto current=process_->concentration_for_validation(),base=baseline_->concentration_for_validation();
    for(int y=0;y<settings_.grid.height;++y) for(int x=0;x<settings_.grid.width;++x) {
        auto i=std::size_t(y)*settings_.grid.width+x;
        fields<<x<<','<<y<<','<<current[i]<<','<<snapshot.concentration[i]<<','<<temp[i]<<','<<target.concentration[i]<<',';
        if(prediction.concentration.size()==current.size()) fields<<prediction.concentration[i];
        fields<<','<<base[i]<<'\n';
    }
    meta<<std::setprecision(17)<<"backend="<<(settings_.backend==Backend::cpu?"cpu":"cuda")<<"\ncompiler="<<__VERSION__<<"\ncplusplus="<<__cplusplus<<"\nwidth="<<settings_.grid.width<<"\nheight="<<settings_.grid.height<<"\nlx_m="<<settings_.grid.lx<<"\nly_m="<<settings_.grid.ly<<"\nd0_m2_s="<<settings_.physics.d0<<"\nea_eV="<<settings_.physics.ea<<"\nseed="<<settings_.seed<<"\nendpoint_s="<<settings_.endpoint<<"\ninterval_s="<<settings_.interval<<"\npreview_horizon_s="<<settings_.horizon<<"\noptimizer="<<(settings_.control.simplex?"opencv_nelder_mead":"finite")<<"\nbudget="<<settings_.control.evaluations<<"\ntiles_x="<<settings_.control.tiles_x<<"\ntiles_y="<<settings_.control.tiles_y<<"\nlower_K="<<settings_.control.lower<<"\nupper_K="<<settings_.control.upper<<"\nfilter_radius="<<settings_.filter.radius<<"\nfilter_sigma_px="<<settings_.filter.sigma<<"\ninitial_temp_K="<<settings_.initial_temp<<"\npaced="<<settings_.paced<<"\nforecast_assumption=fixed candidate temperature to endpoint\n";
    meta<<"temperature_preset="<<settings_.temperature<<"\npattern="<<static_cast<int>(settings_.preset)
        <<"\nchange_penalty="<<settings_.control.change_penalty<<"\nheating_penalty="<<settings_.control.heating_penalty
        <<"\nspatial_penalty="<<settings_.control.spatial_penalty<<"\noptimizer_tolerance="<<settings_.control.tolerance
        <<"\ncuda_kernel="<<(settings_.control.cuda_tiled?"tiled":"global")<<"\ncuda_graphs="<<settings_.control.cuda_graphs<<'\n';
    meta<<"cuda_resident="<<settings_.control.cuda_resident<<'\n';
    meta<<"initialization_ms="<<initialization_ms_<<'\n';
#ifdef HZL_HAS_CUDA
    meta<<"cuda_device="<<cuda_device_info()<<'\n';
#endif
    std::ifstream cpu_info("/proc/cpuinfo"); std::string cpu_line;
    while(std::getline(cpu_info,cpu_line)) if(cpu_line.starts_with("model name")) { meta<<"cpu="<<cpu_line<<'\n'; break; }
    double desired=mass(settings_.grid,target.concentration);
    meta<<"target_mass_difference="<<desired-initial_mass_<<"\nmass_compatibility_is_not_sufficient_for_reachability=1\n";
    if(!cycles.empty()) {
        std::vector<double> latencies; double sum=0,max=0; int misses=0;
        for(const auto& c:cycles) { latencies.push_back(c.cycle_ms); sum+=c.cycle_ms; max=std::max(max,c.cycle_ms); misses+=c.deadline_miss; }
        std::sort(latencies.begin(),latencies.end()); auto n=latencies.size();
        meta<<"cycle_mean_ms="<<sum/n<<"\ncycle_median_ms="<<(latencies[(n-1)/2]+latencies[n/2])/2<<"\ncycle_p95_ms="<<latencies[std::size_t(std::ceil(0.95*n))-1]<<"\ncycle_max_ms="<<max<<"\ndeadline_misses="<<misses<<"\ncycle_real_time_factor="<<time()*1000/sum<<"\n";
    }
    log.flush(); fields.flush(); meta.flush(); if(!log || !fields || !meta) throw std::runtime_error("experiment output write failed");
}
}
