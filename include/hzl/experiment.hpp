#pragma once
#include "hzl/control.hpp"
#include <memory>
namespace hzl {
struct Settings {
    Grid grid{65,49,0.001,0.001}; Physics physics; ControlConfig control;
    Backend backend=Backend::cpu;
    Pattern preset=Pattern::gaussian;
    Filter filter;
    double endpoint=10, interval=1, horizon=5, initial_temp=800;
    unsigned seed=42;
    std::string output="run", temperature="uniform";
    bool paced=false;
    void validate() const;
};
Settings parse_settings(int argc,char** argv);
struct Cycle {
    double timestamp{}, next_timestamp{}, reconstruction_rmse{}, target_mse{}, baseline_mse{}, mass_error{},
           image_ms{}, reconstruction_ms{}, forecast_ms{}, optimization_ms{}, command_ms{}, advance_ms{}, cycle_ms{};
    double predicted_endpoint_mse{}, solver_dt{};
    std::size_t solver_steps{}, control_upload_bytes{}, forecast_download_bytes{}, device_allocations{}, snapshot_upload_bytes{}, target_upload_bytes{}, graph_rebuilds{};
    int evaluations{}; bool deadline_miss{}; std::string status;
};
class Experiment {
    Settings settings_;
    std::vector<double> initial_;
    std::unique_ptr<Process> process_,baseline_;
#ifdef HZL_HAS_CUDA
    std::unique_ptr<CudaForecaster> cuda_workspace_;
#endif
    double initial_mass_{}, initialization_ms_{};
public:
    Target target;
    std::vector<double> temp;
    Image image;
    Snapshot snapshot;
    Forecast prediction;
    Decision decision;
    std::vector<Cycle> cycles;
    explicit Experiment(Settings);
    ~Experiment();
    const Settings& settings() const { return settings_; }
    void reset();
    void replace_initial(std::vector<double>);
    void replace_target(std::vector<double>);
    void refresh();
    void preview();
    Cycle step(bool automatic=true);
    bool finished() const;
    double time() const { return process_->timestamp(); }
    std::span<const double> truth() const { return process_->concentration_for_validation(); }
    std::span<const double> baseline() const { return baseline_->concentration_for_validation(); }
    void save() const;
};
}
