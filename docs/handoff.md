# Project handoff — 2026-10-27

Development is wrapped up for the current prototype. The current delivery is a runnable, tested scientific prototype; it is not a calibrated manufacturing model or a completed production release.

## Run

```sh
cd ~/HZL-projects/HZL-cudaDiffusion
cmake --build --preset cuda
build/cuda/hzl-gui --backend cuda
```

For the connected target:

```sh
build/cuda/hzl-gui --backend cuda --pattern network \
  --width 129 --height 97 --endpoint 60 --interval 3
```

Space runs/pauses, N steps, F previews, A toggles automatic/manual, R resets, S saves, Q/Esc quits. Presets 1–6 include Gaussian, deposits, line, disk, rosette and connected Y network. CPU/headless commands and all configuration options are in README.md.

## Delivered and measured

Conservative variable-diffusivity solver, chronological synthetic observations and reconstruction, isolated forecasts, bounded endpoint control, interactive dashboard/editing, reachable targets, CPU/CUDA tests, reproducible benchmarks and profiling workflows are implemented. Resident forecasting is the CUDA CLI/GUI default, with explicit rollback via `--cuda-resident 0`.

Host RTX 3080 evidence covers functional CUDA suites, exercised memory/synchronization/race checks, repeated control matrices, implementation comparisons, resident closed-loop/GUI actions, and a Systems trace. The paired residency comparison has 180 cycles, no deadline misses and matching endpoint errors. Mean cycle latencies fell from 6.94 to 4.39ms (65×49), 37.47 to 27.89ms (129×97), and 55.21 to 33.50ms (257×129). These are workload-specific steady measurements; initialization, rendering and output have separate timing boundaries.

The latest rosette/network additions were CPU-tested, visually rendered and CUDA-compiled locally. The network target preview was checked for one connected component at a documented display threshold. A new host GPU run of these specific presets has not been reported. Earlier solver/residency validation should not be described as a fresh run of the latest presets.

## Remaining limits

Process/image/display transfers and allocations remain; forecasting residency is not full application GPU residency. Native OS event delivery/window-manager acceptance and desktop presentation latency need broader checking. Nsight Compute hardware counters are permission-blocked. Release packaging, fully pinned dependency distribution, sustained reliability work and hardware-specific experiments remain separate work. Thermal hardware, detailed chemistry and real-glass optical calibration were outside the initial scope.

## Evidence

- residency-results.md: adopted residency measurements and limits.
- implementation-selection.md: global/ordinary-launch selection.
- gpu-results.md: earlier CPU/GPU matrix.
- acceptance.md and gui-acceptance.md: validation status and remaining checks.
- examples/: preserved benchmark data, metadata, sanitizer logs and dashboard previews.
