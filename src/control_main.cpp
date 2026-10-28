#include "hzl/experiment.hpp"
#include <iostream>
#include <string>
int main(int argc,char** argv) {
 try {
    if(argc==2 && std::string(argv[1])=="--help") {
        std::cout<<"hzl-control [--config file] [--backend cpu|cuda] [--cuda-resident 0|1 (CUDA default: 1)] [--method simplex|finite] [--width 65 --height 49] [--endpoint 10 --interval 1 --horizon 5] [--budget 60] [--pattern gaussian|deposits|line|circle|rosette|network] [--output run] [--paced]\nSee README for all configuration fields.\n"; return 0;
    }
    hzl::Experiment experiment(hzl::parse_settings(argc,argv));
    std::cout<<"observation_s,next_s,target_mse,baseline_mse,evaluations,cycle_ms,deadline_miss,status\n";
    while(!experiment.finished()) {
        auto c=experiment.step();
        std::cout<<c.timestamp<<','<<c.next_timestamp<<','<<c.target_mse<<','<<c.baseline_mse<<','<<c.evaluations<<','<<c.cycle_ms<<','<<c.deadline_miss<<','<<c.status<<'\n';
    }
    experiment.save();
    std::cout<<"Experiment ended at its endpoint. Diffusion would continue under any subsequently applied temperature.\n";
 } catch(const std::exception& e) { std::cerr<<"error: "<<e.what()<<'\n'; return 1; }
}
