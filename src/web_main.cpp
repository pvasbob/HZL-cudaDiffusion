#include "hzl/experiment.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
void field(const char* name, std::span<const double> values) {
    std::cout << ",\"" << name << "\":[";
    const char* separator = "";
    for (double value : values) {
        std::cout << separator << value;
        separator = ",";
    }
    std::cout << ']';
}

void state(const hzl::Experiment& run) {
    const auto& s = run.settings();
    std::cout << "{\"width\":" << s.grid.width << ",\"height\":" << s.grid.height
              << ",\"time\":" << run.time() << ",\"endpoint\":" << s.endpoint
              << ",\"interval\":" << s.interval << ",\"horizon\":" << s.horizon
              << ",\"lower\":" << s.control.lower << ",\"upper\":" << s.control.upper
              << ",\"tilesX\":" << s.control.tiles_x << ",\"tilesY\":" << s.control.tiles_y
              << ",\"finished\":" << (run.finished() ? "true" : "false")
              << ",\"targetError\":" << hzl::mse(run.truth(), run.target.concentration)
              << ",\"baselineError\":" << hzl::mse(run.baseline(), run.target.concentration)
              << ",\"forecastStart\":" << run.prediction.start_time
              << ",\"forecastEnd\":" << run.prediction.endpoint_time;
    field("concentration", run.truth());
    field("reconstructed", run.snapshot.concentration);
    field("temperature", run.temp);
    field("target", run.target.concentration);
    field("forecast", run.prediction.concentration);
    field("baseline", run.baseline());
    std::cout << "}\n" << std::flush;
}
}

int main(int argc, char** argv) {
    try {
        hzl::Experiment run(hzl::parse_settings(argc, argv));
        std::cout << std::setprecision(17);
        state(run);
        for (std::string line; std::getline(std::cin, line);) {
            try {
                std::istringstream input(line);
                std::string command;
                input >> command;
                if (command == "auto") run.step();
                else if (command == "manual") run.step(false);
                else if (command == "preview") run.preview();
                else if (command == "reset") run.reset();
                else if (command == "save") run.save();
                else if (command == "temperature" || command == "target") {
                    int x, y;
                    double value;
                    const auto& s = run.settings();
                    const auto& g = s.grid;
                    const bool temperature = command == "temperature";
                    const double lower = temperature ? s.control.lower : 0;
                    const double upper = temperature ? s.control.upper : 1;
                    if (!(input >> x >> y >> value) || x < 0 || x >= g.width ||
                        y < 0 || y >= g.height || !(value >= lower && value <= upper))
                        throw std::invalid_argument("Brush position or value is out of range");
                    auto& values = temperature ? run.temp : run.target.concentration;
                    for (int dy = -2; dy <= 2; ++dy)
                        for (int dx = -2; dx <= 2; ++dx)
                            if (dx*dx + dy*dy <= 4 && x+dx >= 0 && x+dx < g.width &&
                                y+dy >= 0 && y+dy < g.height)
                                values[std::size_t(y+dy)*g.width + x+dx] = value;
                    run.prediction = {};
                } else if (command != "state") throw std::invalid_argument("Unknown command");
                state(run);
            } catch (const std::exception& error) {
                std::cout << '!' << error.what() << '\n' << std::flush;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
