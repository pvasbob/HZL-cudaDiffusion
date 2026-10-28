#pragma once
#include <cstddef>
#include <span>
#include <limits>
#include <vector>
#include <string>
namespace hzl {
struct Grid {
    int width = 129, height = 97;
    double lx = 0.001, ly = 0.001; // metres; cell-centred finite volumes
    void validate() const;
    std::size_t size() const;
    double dx() const { return lx / width; }
    double dy() const { return ly / height; }
};
struct Physics {
    double d0 = 1e-6; // m^2/s (illustrative)
    double ea = 0.8; // eV per particle
    double min_temp = 300, max_temp = 1200; // Kelvin
    double diffusivity(double kelvin) const;
    void validate() const;
};
struct Schedule { std::size_t steps{}; double dt{}; };
Schedule schedule(const Grid&, double max_diffusivity, double seconds, double max_dt=std::numeric_limits<double>::infinity());
struct Plan {
    std::vector<double> diffusivity;
    std::size_t steps;
    double dt;
};
Plan plan(const Grid&, const Physics&, std::span<const double> temp, double seconds, double max_dt=std::numeric_limits<double>::infinity());
void validate_concentration(const Grid&, std::span<const double>);
std::vector<double> gaussian(const Grid&);
std::vector<double> advance_cpu(const Grid&, const Physics&, std::span<const double> conc,
                                std::span<const double> temp, double seconds, double max_dt=std::numeric_limits<double>::infinity());
double mass(const Grid&, std::span<const double>);
#ifdef HZL_HAS_CUDA
bool cuda_available();
std::string cuda_device_info();
std::vector<double> advance_cuda(const Grid&, const Physics&, std::span<const double> conc,
                                 std::span<const double> temp, double seconds,
                                 double* device_ms = nullptr, double max_dt=std::numeric_limits<double>::infinity());
#endif
}
