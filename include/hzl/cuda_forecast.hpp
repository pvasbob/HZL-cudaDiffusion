#pragma once
#include "hzl/observation.hpp"
#include <memory>
namespace hzl {
// A resident snapshot/target with reusable scratch and CUB storage.
// The optional optimized path supports reuse across control decisions.
// Not thread safe: each concurrently evaluated candidate needs its own workspace.
enum class CudaKernel { global, tiled };
struct CudaForecastStats {
    std::size_t evaluations{}, device_allocations{}, snapshot_upload_bytes{}, target_upload_bytes{},
                control_upload_bytes{}, forecast_download_bytes{}, retained_forecasts{}, graph_rebuilds{};
};
class CudaForecaster {
    struct Impl; std::unique_ptr<Impl> impl_;
public:
    CudaForecaster(const Snapshot&,std::span<const double> target,CudaKernel=CudaKernel::global,bool graphs=false,bool optimized=false);
    ~CudaForecaster();
    CudaForecaster(const CudaForecaster&)=delete;
    CudaForecaster& operator=(const CudaForecaster&)=delete;
    void update(const Snapshot&,std::span<const double> target);
    bool optimized() const;
    CudaForecastStats stats() const;
    void retain();
    std::vector<double> download_retained() const;
    double evaluate_tiles(const Physics&,std::span<const double> coefficients,int tiles_x,int tiles_y,double horizon);
    double evaluate(const Physics&,std::span<const double> temp,double horizon);
    std::vector<double> download() const;
    double device_ms() const;
};
}
