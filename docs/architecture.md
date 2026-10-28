# Architecture

## Implemented components

| Component | Files | Ownership and purpose |
|---|---|---|
| Model/configuration | model.hpp/model.cpp | Validated grid and Arrhenius units; finite-volume CPU reference; generated fields |
| CUDA solver | solver.cu, cuda_stencil.cuh | Conservative global stencil; checked explicit integration |
| CUDA resources | cuda_resources.cuh | Noncopyable RAII buffers, streams and events; no global mutable state |
| Images/process | observation.hpp/observation.cpp | Process owns underlying concentration, applied map and chronological time; strided measurement, reconstruction, snapshot, forecast |
| CUDA images | cuda_image.cu | Fused measurement decoding and shared-memory Gaussian filtering |
| Control | control.hpp/control.cpp | Reachable target generation; bounded four-tile parameterization; objective; hard budget; finite/OpenCV optimizer strategies |
| Resident CUDA forecasting | cuda_forecast.hpp/cuda_forecast.cu | Decision-scoped initial state and target; reusable scratch; CUB reduction; global/tiled and graph alternatives |
| Experiment | experiment.hpp/experiment.cpp | Full chronological loop, independent fixed baseline, config parsing, timing and exports |
| Desktop | gui.cpp | OpenCV nine-panel rendering, mouse painting, manual/automatic controls, presets and diagnostics |
| CLI/benchmarks | control_main.cpp, benchmark.cpp, compare.cpp | Headless experiments, solver latency, equivalent implementation checks |
| Automation | tools/ | Repeated representative benchmark matrix; GPU sanitizer and Nsight commands |

`Process` never accepts forecast endpoints as its state. Its state changes only through `advance`. `forecast` accepts an image-reconstructed `Snapshot`. `optimize` receives only that snapshot, target, model and current temperature, never the process concentration or target-generating schedule. Every candidate reads the same immutable snapshot/time. `Process::concentration_for_validation` is used for mass/reconstruction/error metrics and display only. Public `Experiment::temp` is the edited/selected pointwise map; `apply_temperature` copies it to the ongoing process at the update boundary. No artificial force or heater dynamics exists.

Measurement images explicitly expose channels (one), precision (float64), byte stride, dimensions and timestamp. A real-image adapter can populate this calibrated input structure and use the same reconstruction/forecast interfaces; actual camera acquisition, calibration and optical color inversion are outside this implementation. Field vectors are contiguous row-major double precision, distinct from padded image buffers. Gaussian filtering is optional and reports error rather than being treated as an exact observation.

The default cycle is render → reconstruct → optimize endpoint forecasts → apply map → advance one interval. Simulation time pauses while computing. The last interval is truncated to hit the endpoint; errors never silently advance time. Baseline and controlled processes have separate concentration storage and receive the same interval lengths. Baseline stays at the configured initial uniform temperature. A design-target edit retains its endpoint and is not forced to be reachable.

## Dependency and language decisions

C++20 is the portable core and CUDA dialect. Host C++23 is selectable and build-tested; GCC 11 has incomplete C++23 library support, so unsupported conveniences are not prerequisites. CUDA 12.1 is compatible with the installed GCC 11 host. The older CMake flag-discovery workaround explicitly requests C++20, not a downgrade.

OpenCV 4.5+ supplies standard Gaussian processing, GUI rendering/image encoding and Nelder–Mead optimization. All dependencies are system-resolved, without network fetches at configure time. OpenCV CPU Gaussian was faster than the direct reference in local small-grid comparisons, so the application uses the library path. CUDA image modules are absent; the custom CUDA implementation has a CPU/OpenCV reference. NPP was considered but not integrated without GPU access and verified module/link/performance evidence.

NLopt BOBYQA was evaluated as a suitable bounded derivative-free alternative, but its development package is unavailable. The installed OpenCV solver handles this small-dimensional problem using a smooth bound mapping and a hard callback budget. No general optimizer was reimplemented. The finite strategy is a transparent baseline, not a new general optimization library. Global optimality and arbitrary pattern reachability are not promised. Larger parameterizations need sensitivity/adjoint or more suitable optimizers before scaling.

CUB supplies GPU objective reduction; no custom reduction, BLAS, FFT, or sparse dependency is added without need. OpenCV HighGUI replaced the initially proposed GLFW/ImGui stack to deliver interaction using installed dependencies. This requires CPU presentation buffers and does not implement CUDA–OpenGL interoperability. Headless operation is independent of graphics initialization.

Licenses: OpenCV 4.5.4 uses Apache-2.0; CUB is distributed with CUDA under its included BSD-family license; CUDA toolkit/runtime distribution is governed by NVIDIA terms. GCC/CMake/Ninja/Python are development/system tools. No third-party source is vendored. Review distribution terms for packaged binaries and choose a project license before external release. The placeholder .git directory in this workspace is not a usable Git repository; no commit or remote CI execution is claimed.

## Milestone status

1. Numerical foundation: implemented and CPU-tested, including independent spatial/time refinement and viewable exports. CUDA compilation passes; GPU agreement passed in the RTX 3080 host run.
2. Chronological observations: implemented; stride, filter, timestamp and state-isolation checks pass on CPU/OpenCV. CUDA tiled/halo tests pass on the host; they skip when GPU access is unavailable.
3. Interactive prediction: implemented native GUI and offscreen dashboard tests; display interaction requires a GTK desktop. Model/grid/solver controls are CLI/config driven; painted maps, presets, preview, control and endpoint controls are interactive.
4. Automatic control: implemented and tested, with reachable targets, budgets, bounds, objective penalties, repeated feedback and fixed baselines. Control schedule optimization beyond a fixed future map is deferred.
5. Performance engineering: representative CPU matrix measured; CUDA global/tiled/graphs and CUB workspace implemented and compile-checked; GPU evidence and hardware-specific selection remain pending. See performance.md.

This is a functional engineering prototype, with explicit acceptance gaps rather than an unqualified production-ready claim. Failure recovery is exception-based with top-level CLI diagnostics and GUI pause/status handling. Host sanitizers and numerical tests complement, not replace, GPU sanitizer/profiler runs.


Updated host evidence: [sanitizer logs, unprofiled comparisons and Systems summaries](../examples/gpu-host-validation/validation.md). Variant memcheck/synccheck and observation racecheck report zero errors/hazards; Nsight Systems captured the GPU pipeline. Nsight Compute alone failed with ERR_NVGPUCTRPERM. Its printed timings must not be used as ordinary benchmark measurements. Broader workloads and live GUI interaction remain unverified.


Experimental forecasting residency is described in residency.md. With `cuda-resident=1`, Experiment owns a reusable CudaForecaster, which validates snapshot/target updates, caches the target, prepares diffusivity from GPU-expanded controls, fuses CUB error reduction and retains the best state until selection completes. The host has validated and measured this mode, and CUDA CLI/GUI defaults to it unless explicitly overridden. `cuda-resident=0` preserves the legacy comparison path. See residency-results.md for 25.5–39.3% measured mean-cycle reductions and scope limits; the process/image pipeline is still not fully resident.
