#pragma once
#include "hzl/observation.hpp"
#include <string>
namespace hzl {
class CudaForecaster;
enum class Pattern { gaussian, deposits, line, circle, rosette, network };
std::vector<double> pattern(const Grid&, Pattern);
struct ControlConfig {
    int tiles_x=2, tiles_y=2;
    int evaluations=60;
    double lower=700, upper=1100;
    double change_penalty=0, heating_penalty=0, spatial_penalty=0;
    double tolerance=1e-10;
    bool simplex=true;
    bool cuda_tiled=false, cuda_graphs=false, cuda_resident=false;
    void validate(const Grid&, const Physics&) const;
};
struct Target { Grid grid; double endpoint; std::vector<double> concentration; };
struct Decision {
    std::vector<double> temp;
    Forecast prediction;
    double objective{}, target_mse{}, initial_objective{}, forecast_ms{}, optimization_ms{};
    std::size_t control_upload_bytes{}, forecast_download_bytes{}, device_allocations{}, snapshot_upload_bytes{}, target_upload_bytes{}, graph_rebuilds{};
    int evaluations{};
    std::string stopping_reason;
};
double mse(std::span<const double>,std::span<const double>);
std::vector<double> expand_tiles(const Grid&, int nx, int ny, std::span<const double> coefficients);
Target reachable_target(const Grid&, const Physics&, std::span<const double> initial, double endpoint, unsigned seed, Backend, double lower=700, double upper=1100);
Decision optimize(const Snapshot&, const Physics&, const Target&, std::span<const double> current_temp, const ControlConfig&, Backend, CudaForecaster* workspace=nullptr);
}
