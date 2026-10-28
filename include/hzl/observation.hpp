#pragma once
#include "hzl/model.hpp"
namespace hzl {
enum class Backend { cpu, cuda };
struct Image {
    int width{}, height{};
    std::size_t stride_bytes{}; // one linear grayscale channel, IEEE float64
    double timestamp{};
    std::vector<double> pixels;
    void validate() const;
};
struct Filter {
    int radius=0; // zero bypasses filtering
    double sigma=1; // pixels
    bool reference=false; // select direct CPU convolution for comparisons
    void validate() const;
};
struct Snapshot { Grid grid; double timestamp; std::vector<double> concentration; };
struct Forecast { double start_time, endpoint_time; std::vector<double> concentration; };
// Measurement encoding I = 0.1 + 0.8*c, no clipping or display colormap.
Image render(const Grid&, std::span<const double> conc, double timestamp, std::size_t padding=0);
Snapshot reconstruct(const Grid&, const Image&, Filter={}, Backend=Backend::cpu);
Forecast forecast(const Snapshot&, const Physics&, std::span<const double> temp, double horizon, Backend=Backend::cpu);
class Process {
    Grid grid_; Physics physics_; Backend backend_;
    std::vector<double> conc_, temp_;
    double timestamp_=0;
public:
    Process(Grid, Physics, std::vector<double> conc, std::vector<double> temp, Backend=Backend::cpu);
    void apply_temperature(std::span<const double> temp);
    void advance(double seconds);
    Image observe(std::size_t padding=0) const;
    double timestamp() const { return timestamp_; }
    std::span<const double> concentration_for_validation() const { return conc_; }
};
std::vector<double> gaussian_weights(Filter);
#ifdef HZL_HAS_CUDA
std::vector<double> reconstruct_cuda(const Image&, Filter);
#endif
}
