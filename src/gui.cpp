#include "hzl/experiment.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
namespace {
constexpr int panel_w=340,panel_h=205,top=110;
constexpr const char* window="HZL diffusion / synthetic observation and endpoint control";
std::string format(double v,int digits=3) { std::ostringstream s; s<<std::setprecision(digits)<<v; return s.str(); }
void text(cv::Mat& canvas,const std::string& s,int x,int y,double scale=0.46,cv::Scalar color={210,218,228}) {
    cv::putText(canvas,s,{x,y},cv::FONT_HERSHEY_SIMPLEX,scale,color,1,cv::LINE_AA);
}
struct Window {
    Window() { cv::namedWindow(window,cv::WINDOW_AUTOSIZE); }
    ~Window() { try { cv::destroyWindow(window); } catch(...) {} }
    bool is_open() const {
        // OpenCV 4.5 GTK lacks WND_PROP_VISIBLE. Its AUTOSIZE query supports
        // existing normal/autosize windows (0/1) and returns -1 when closed.
        try {
            double autosize=cv::getWindowProperty(window,cv::WND_PROP_AUTOSIZE);
            return !std::isfinite(autosize) || autosize>=0;
        } catch(const cv::Exception& e) {
            if(e.code==cv::Error::StsNullPtr) return false;
            throw;
        }
    }
    Window(const Window&)=delete;
    Window& operator=(const Window&)=delete;
};
struct UI {
    hzl::Settings settings;
    std::unique_ptr<hzl::Experiment> experiment;
    bool playing=false,automatic=true;
    double brush=850;
    double render_ms=0;
    std::string status="Paused. Click temperature or target panels to paint.";
    explicit UI(hzl::Settings s):settings(std::move(s)),experiment(std::make_unique<hzl::Experiment>(settings)) { brush=std::clamp(brush,settings.control.lower,settings.control.upper); }
    void rebuild() { experiment=std::make_unique<hzl::Experiment>(settings); playing=false; status="Reset: reachable target regenerated."; }
    cv::Mat draw() {
        auto start=std::chrono::steady_clock::now(); auto& e=*experiment; const auto& g=settings.grid;
        cv::Mat canvas(900,1020,CV_8UC3,cv::Scalar(19,22,28));
        text(canvas,"HZL / DIFFUSION PREDICTION & TEMPERATURE CONTROL",16,27,0.65,{115,220,250});
        text(canvas,"t="+format(e.time())+"s  endpoint="+format(e.target.endpoint)+"s  interval="+format(settings.interval)+"s  preview="+format(settings.horizon)+"s",16,52);
        text(canvas,"Backend: "+std::string(settings.backend==hzl::Backend::cpu?"CPU":"CUDA")+" | "+(automatic?"AUTO":"MANUAL")+" | "+(playing?"RUNNING":"PAUSED")+" | grid "+std::to_string(g.width)+"x"+std::to_string(g.height)+" | Kelvin brush="+format(brush,5),16,74);
        text(canvas,"Space play/pause | N step | A auto/manual | F preview | R reset | G new target | S save | Q quit",16,96);
        std::vector<double> encoded(g.size()),err(g.size()),recerr(g.size()); auto truth=e.truth();
        auto stride=e.image.stride_bytes/sizeof(double);
        for(int y=0;y<g.height;++y) for(int x=0;x<g.width;++x) {
            auto i=std::size_t(y)*g.width+x; encoded[i]=(e.image.pixels[y*stride+x]-0.1)/0.8;
            err[i]=std::abs(truth[i]-e.target.concentration[i]); recerr[i]=std::abs(truth[i]-e.snapshot.concentration[i]);
        }
        auto panel=[&](int index,const std::string& title,std::span<const double> values,double lo,double hi) {
            int px=index%3*panel_w,py=top+index/3*panel_h;
            text(canvas,title,px+10,py+18,0.46,{130,220,250});
            cv::Mat data(g.height,g.width,CV_64F,const_cast<double*>(values.data())),gray,color,resized;
            data.convertTo(gray,CV_8U,255/(hi-lo),-lo*255/(hi-lo)); cv::applyColorMap(gray,color,cv::COLORMAP_VIRIDIS);
            cv::resize(color,resized,{panel_w-20,panel_h-54},0,0,cv::INTER_NEAREST);
            resized.copyTo(canvas(cv::Rect(px+10,py+27,resized.cols,resized.rows)));
            text(canvas,"scale ["+format(lo)+", "+format(hi)+"]",px+10,py+panel_h-7,0.4);
        };
        panel(0,"Synthetic image (decoded for display)",encoded,0,1);
        panel(1,"Reconstructed concentration",e.snapshot.concentration,0,1);
        panel(2,"Underlying concentration / paint at t=0",truth,0,1);
        panel(3,"Temperature K / paint with mouse",e.temp,settings.control.lower,settings.control.upper);
        panel(4,"Target concentration / paint with mouse",e.target.concentration,0,1);
        if(e.prediction.concentration.size()==g.size()) panel(5,"Forecast "+format(e.prediction.start_time)+" -> "+format(e.prediction.endpoint_time)+" s",e.prediction.concentration,0,1);
        else panel(5,"Forecast (press F or step)",e.snapshot.concentration,0,1);
        panel(6,"Absolute current - target error",err,0,0.1);
        panel(7,"Absolute reconstruction error",recerr,0,0.01);
        panel(8,"Fixed-temperature comparison",e.baseline(),0,1);
        int y=top+3*panel_h+16;
        text(canvas,"1 Gaussian | 2 deposits | 3 line | 4 disk | 5 rosette | 6 network | B filter | T temp | [ / ] brush | E +5s",16,y);
        text(canvas,"Current MSE="+format(hzl::mse(truth,e.target.concentration))+"  baseline MSE="+format(hzl::mse(e.baseline(),e.target.concentration))+"  reconstruction RMSE="+format(std::sqrt(hzl::mse(truth,e.snapshot.concentration))),16,y+23);
        double delta=hzl::mass(g,e.target.concentration)-hzl::mass(g,truth);
        text(canvas,"Target mass difference="+format(delta)+" (mass compatibility alone does not imply reachability)",16,y+46);
        if(!e.cycles.empty()) {
            auto c=e.cycles.back();
            text(canvas,"Cycle="+format(c.cycle_ms)+"ms  optimization="+format(c.optimization_ms)+"ms  forecasts="+format(c.forecast_ms)+"ms / "+std::to_string(c.evaluations)+" evals  mass rel.err="+format(c.mass_error),16,y+69);
            text(canvas,"Cycle RTF="+format((c.next_timestamp-c.timestamp)*1000/c.cycle_ms)+"x  deadline miss="+std::to_string(c.deadline_miss)+"  status="+c.status,16,y+92);
        }
        text(canvas,status,16,y+115,0.43,{160,220,180});
        text(canvas,"Rendering (previous frame): "+format(render_ms)+"ms | display scales fixed; values may exceed scales | stepped simulation",16,y+138,0.4);
        render_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); return canvas;
    }
    void step() {
        if(experiment->finished()) { playing=false; status="Endpoint reached: experiment ended. Material would continue diffusing."; return; }
        experiment->step(automatic);
        if(experiment->finished()) { playing=false; status="Endpoint reached. Press S to save or R to reset."; }
    }
    void action(int key) {
        if(key==' ') playing=!playing;
        else if(key=='n') step();
        else if(key=='a') automatic=!automatic;
        else if(key=='f') { experiment->preview(); status="Manual forecast computed with current temperature held fixed."; }
        else if(key=='r') rebuild();
        else if(key=='g') { ++settings.seed; rebuild(); }
        else if(key=='e') { settings.endpoint+=5; rebuild(); }
        else if(key>='1' && key<='6') { settings.preset=static_cast<hzl::Pattern>(key-'1'); rebuild(); }
        else if(key=='t') {
            const std::vector<std::string> presets{"uniform","gradient","hotspot","spots"};
            auto i=std::find(presets.begin(),presets.end(),settings.temperature)-presets.begin();
            settings.temperature=presets[(i+1)%presets.size()]; rebuild();
        }
        else if(key=='b') { settings.filter.radius=settings.filter.radius==0?2:0; rebuild(); }
        else if(key=='[') brush=std::max(settings.control.lower,brush-25);
        else if(key==']') brush=std::min(settings.control.upper,brush+25);
        else if(key=='s') { experiment->save(); if(!cv::imwrite(settings.output+"/dashboard.png",draw())) throw std::runtime_error("dashboard write failed"); status="Saved experiment and dashboard to "+settings.output; }
    }
    static void mouse(int event,int x,int y,int flags,void* data) {
        if(event!=cv::EVENT_LBUTTONDOWN && !(event==cv::EVENT_MOUSEMOVE && (flags&cv::EVENT_FLAG_LBUTTON))) return;
        auto& ui=*static_cast<UI*>(data);
        try {
            int col=x/panel_w,row=(y-top)/panel_h;
            if(y<top || row<0 || row>=3 || col<0 || col>=3) return;
            int index=row*3+col; if(index!=2 && index!=3 && index!=4) return;
            int xx=x-col*panel_w-10,yy=y-(top+row*panel_h)-27;
            if(xx<0 || yy<0 || xx>=panel_w-20 || yy>=panel_h-54) return;
            if(ui.playing) { ui.status="Pause before editing a field."; return; }
            const auto& g=ui.settings.grid; int gx=xx*g.width/(panel_w-20),gy=yy*g.height/(panel_h-54);
            auto paint=[&](std::vector<double>& v,double value) {
                for(int j=-2;j<=2;++j) for(int i=-2;i<=2;++i) if(i*i+j*j<=4 && gx+i>=0 && gy+j>=0 && gx+i<g.width && gy+j<g.height) v[std::size_t(gy+j)*g.width+gx+i]=value;
            };
            if(index==3) { paint(ui.experiment->temp,ui.brush); ui.status="Temperature edited; next step applies it (AUTO may optimize it)."; }
            else if(index==4) { auto target=ui.experiment->target.concentration; paint(target,1); ui.experiment->replace_target(std::move(target)); ui.status="Designed target edited; residual and mass mismatch are reported honestly."; }
            else if(ui.experiment->time()==0) { auto initial=std::vector<double>(ui.experiment->truth().begin(),ui.experiment->truth().end()); paint(initial,1); ui.experiment->replace_initial(std::move(initial)); ui.status="Initial field edited; reachable target regenerated."; }
            else ui.status="Initial concentration can be edited only at t=0; reset first.";
        } catch(const std::exception& e) { ui.playing=false; ui.status=e.what(); }
    }
};
void interaction_smoke(UI& ui) {
    auto check=[](bool value,const char* message) { if(!value) throw std::runtime_error(message); };
    check(ui.settings.endpoint>=2*ui.settings.interval,"interaction smoke requires at least two control intervals");
    auto paint=[&](int index) {
        UI::mouse(cv::EVENT_LBUTTONDOWN,index%3*panel_w+10+(panel_w-20)/2,
                  top+index/3*panel_h+27+(panel_h-54)/2,cv::EVENT_FLAG_LBUTTON,&ui);
    };
    ui.action('n'); check(ui.experiment->time()==ui.settings.interval,"automatic step timestamp failed");
    auto observed_time=ui.experiment->time();
    ui.action('f'); check(ui.experiment->time()==observed_time,"preview advanced process time");
    check(ui.experiment->prediction.endpoint_time==observed_time+ui.settings.horizon,"preview horizon failed");
    auto old_temp=ui.experiment->temp;
    ui.action(' '); paint(3); check(ui.experiment->temp==old_temp,"editing allowed while running");
    ui.action(' '); ui.action('a'); check(!ui.automatic,"manual mode toggle failed");
    ui.brush=ui.settings.control.upper; paint(3);
    check(ui.experiment->temp!=old_temp,"temperature painting did not change map");
    auto previous=std::vector<double>(ui.experiment->truth().begin(),ui.experiment->truth().end());
    auto expected=hzl::advance_cpu(ui.settings.grid,ui.settings.physics,previous,ui.experiment->temp,ui.settings.interval);
    ui.action('n');
    check(hzl::mse(expected,ui.experiment->truth())<1e-24,"manual painted control not applied");
    previous.assign(ui.experiment->truth().begin(),ui.experiment->truth().end()); paint(2);
    check(hzl::mse(previous,ui.experiment->truth())==0,"initial-state painting allowed after time zero");
    ui.action('r'); check(ui.experiment->time()==0 && ui.experiment->cycles.empty(),"reset did not restore timeline");
    previous.assign(ui.experiment->truth().begin(),ui.experiment->truth().end()); paint(2);
    check(hzl::mse(previous,ui.experiment->truth())>0,"initial-state painting failed");
    auto old_target=ui.experiment->target.concentration; paint(4);
    check(hzl::mse(old_target,ui.experiment->target.concentration)>0,"target painting failed");
    ui.action('6'); check(ui.settings.preset==hzl::Pattern::network,"connected target preset failed");
    ui.action('5'); check(ui.settings.preset==hzl::Pattern::rosette,"structured target preset failed");
    ui.action('2'); check(ui.settings.preset==hzl::Pattern::deposits && ui.experiment->time()==0,"preset failed");
    ui.action('t');
    auto limits=std::minmax_element(ui.experiment->temp.begin(),ui.experiment->temp.end());
    check(*limits.first<*limits.second,"temperature preset failed");
    ui.action('b'); check(ui.settings.filter.radius==2,"filter toggle failed");
    check(hzl::mse(ui.experiment->snapshot.concentration,ui.experiment->truth())>0,"filter bias missing");
    auto seed=ui.settings.seed; ui.action('g'); check(ui.settings.seed==seed+1,"target regeneration failed");
    auto endpoint=ui.settings.endpoint; ui.action('e'); check(ui.experiment->target.endpoint==endpoint+5,"endpoint edit failed");
    ui.action('s');
    auto output=std::filesystem::path(ui.settings.output);
    for(const auto& name:{"cycles.csv","fields.csv","metadata.txt","dashboard.png"})
        check(std::filesystem::file_size(output/name)>0,"save output missing");
    std::cout<<"PASS: shared GUI key/mouse actions, chronological steps, preview isolation, editing guards, manual control, presets, filtering, reset, endpoint and save. Desktop event delivery is not exercised."<<std::endl;
}
}
int main(int argc,char** argv) {
 try {
    bool render_only=false,smoke=false,interactions=false; std::vector<char*> args{argv[0]};
    for(int i=1;i<argc;++i) {
        std::string a=argv[i]; if(a=="--render-only") render_only=true; else if(a=="--smoke") smoke=true; else if(a=="--interaction-smoke") interactions=true; else args.push_back(argv[i]);
    }
    std::cout<<"[HZL] Initializing simulation and forecast..."<<std::endl;
    UI ui(hzl::parse_settings(static_cast<int>(args.size()),args.data())); ui.experiment->preview();
    if(interactions) { interaction_smoke(ui); return 0; }
    if(render_only) {
        ui.step(); std::filesystem::create_directories(ui.settings.output);
        std::vector<double> times; cv::Mat dashboard;
        for(int frame=-3;frame<20;++frame) { dashboard=ui.draw(); if(frame>=0) times.push_back(ui.render_ms); }
        if(!cv::imwrite(ui.settings.output+"/dashboard.png",dashboard)) throw std::runtime_error("dashboard write failed");
        std::sort(times.begin(),times.end());
        std::ofstream report(ui.settings.output+"/render.csv");
        report<<"samples,mean_ms,median_ms,p95_ms,max_ms\n20,"
              <<std::accumulate(times.begin(),times.end(),0.)/20<<','<<(times[9]+times[10])/2<<','<<times[18]<<','<<times[19]<<'\n';
        report.flush(); if(!report) throw std::runtime_error("render report write failed");
        ui.experiment->save(); return 0;
    }
    std::cout<<"[HZL] Opening desktop window..."<<std::endl;
    Window desktop_window; cv::setMouseCallback(window,UI::mouse,&ui);
    int frames=0;
    while(true) {
        try { if(ui.playing) ui.step(); } catch(const std::exception& e) { ui.playing=false; ui.status=e.what(); }
        cv::imshow(window,ui.draw()); int key=cv::waitKey(20)&0xff;
        if(frames==0) std::cout<<"[HZL] Dashboard ready (paused). Space: run/pause; N: step; Q/Esc: quit."<<std::endl;
        if(key=='q' || key==27) { std::cout<<"[HZL] Quit requested."<<std::endl; break; }
        if(!desktop_window.is_open()) { std::cout<<"[HZL] Window closed."<<std::endl; break; }
        try {
            ui.action(key);
        } catch(const std::exception& e) { ui.playing=false; ui.status=e.what(); }
        ++frames;
        if(smoke && frames==2) { ui.step(); ui.experiment->save(); std::cout<<"[HZL] GUI smoke test completed."<<std::endl; break; }
    }

 } catch(const std::exception& e) { std::cerr<<"error: "<<e.what()<<'\n'; return 1; }
}
