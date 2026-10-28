# Local acceptance and remaining limits

## Already executed in this workspace

- C++20 Release CPU application, numerical/image/control/config tests and offscreen dashboard rendering pass.
- C++23 host Release compiles and passes the same tests.
- Minimal finite-controller build compiles and passes tests.
- CUDA 12.1 C++20 compilation targets sm_86 and links all applications/variant tests. The full CTest suite ran on the RTX 3080 host: 10/10 passed, including all four CUDA suites. A default GPU control run completed with zero deadline misses; preserved results are in examples/gpu-host-smoke.
- Thirty representative CPU control runs are recorded (three repeats, ten cases), plus solver/filter comparisons and offscreen rendering samples. All benchmark source commands are preserved.
- CPU ASan/UBSan Debug tests pass with leak detection disabled. LeakSanitizer's ptrace restriction prevents leak validation here; it is not a discovered application leak or a passed leak check.
- Live desktop window startup was confirmed after fixing an unsupported GTK visibility query that caused immediate exit. Offscreen rendering/PNG tests pass. Full mouse/keyboard/control/save acceptance is still pending.

## RTX 3080 host

1. Ensure `nvidia-smi` succeeds and identifies the intended RTX 3080. This environment has the toolkit but no accessible driver; no package installation inside the project can substitute for exposing the host driver/device.
2. Build with preset `cuda-desktop` (or `cuda` for headless). Run `tools/validate_cuda.sh build/cuda-desktop gpu-reports`. CUDA requests must actually run, not fall back or all skip.
3. Inspect CPU/GPU agreement, global/tiled/graph/CUB tests and Compute Sanitizer reports. Address failures before making GPU accuracy or speed claims. The graph variants and cooperative halo loads particularly need memory/synchronization checking.
4. Compare representative solver/filter variants and full controller runs at matched grids, model settings and budgets. Keep or reject variants using complete-cycle and numerical-error data. Capture driver/toolkit/GPU versions and profiler availability.
5. Run the desktop application with a GTK display. Check pause/step/reset, presets, manual forecast, painting initial/temperature/target fields, automatic control, target regeneration and save. Confirm actual window latency and its effect on update deadlines.

The host run establishes passing GPU functional tests and a successful default control run. Variant memcheck/synccheck and observation racecheck pass, and a Systems GPU trace was captured. A 30-run GPU matrix now confirms zero misses across 300 cycles (docs/gpu-results.md). Nsight Compute counters, broader GPU variant selection and detailed GUI interaction remain unverified; production acceptance depends on those results.

## Deliberately limited scope

- One species, 2D effective thin layer, no reactions, no real optical calibration, no heater physics or real equipment.
- Fixed future temperature map per candidate; no multi-stage future control schedule or adjoint.
- Default four control tiles, up to 64 supported; increasing variables does not make derivative-free optimization scale efficiently.
- Target mismatch remains an error, including unreachable painted targets. The app reports mass differences but does not implement a general reachability proof.
- Stepped deterministic simulation, optional deadline/pacing demonstration. A slow computation is reported, not hidden as realtime physical capability. There is no continuously evolving virtual plant during a solve.
- GUI/controller execute serially; CUDA–OpenGL interop and full GPU residency are not implemented. Per-cycle allocations and CPU observation/display transfers remain measurable optimization targets.
- Dependency versions are constrained/tested system versions rather than a fully pinned distribution. CI configuration is provided but remote runs and release packaging have not happened here.
- The numerical and software tests are meaningful baselines, not exhaustive production safety certification, device-failure recovery, or manufacturing validation.


Updated host evidence: [sanitizer logs, unprofiled comparisons and Systems summaries](../examples/gpu-host-validation/validation.md). Variant memcheck/synccheck and observation racecheck report zero errors/hazards; Nsight Systems captured the GPU pipeline. Nsight Compute alone failed with ERR_NVGPUCTRPERM. Its printed timings must not be used as ordinary benchmark measurements. The repeated GPU matrix is now recorded in gpu-results.md; window startup is confirmed, while detailed GUI interaction remains unverified.


## Full-controller implementation selection

Run `python3 tools/benchmark_variants.py --binary build/cuda/hzl-control --repeats 3 --output gpu-reports/variants` from the host terminal. The script compares three grids and all four global/tiled/graph combinations, rotates order between repeats, validates endpoint MSE, and includes the optimization-time graph rebuild cost. It needs no hardware counter permissions. Raw output and an accuracy-checked report are saved; changing the default still requires reviewing this evidence.

Desktop startup is confirmed. Check interaction separately using the sequence in gui-acceptance.md; window opening alone does not establish all control/editing behaviors.


Current selection: the host completed all 36 variant experiments, with matched endpoint errors and zero misses across 360 cycles. The existing global/ordinary-launch default is supported on all three measured grids; see implementation-selection.md. A display-independent GUI action test now passes on CPU C++20/C++23 and under ASan/UBSan, exercising keyboard/mouse handlers directly. It does not replace OS-level input/window-manager acceptance. Window startup is confirmed; full interactive acceptance remains pending.


GPU-backed GUI action smoke now passes on the host. Its saved artifacts are preserved in examples/gpu-gui-actions, including CUDA device metadata and all 3185 field rows. The test covers shared application key/mouse actions and their numerical/state effects without OS events. Remaining desktop acceptance is actual event delivery, window-manager close behavior and presentation latency. Nsight Compute counters remain permission-blocked. No additional numerical/controller source change was needed after the successful implementation sweep.


Forecasting residency acceptance completed on the host: expanded numerical/workspace tests, resident closed-loop and GUI actions pass; variant/control GPU memchecks report zero errors/leaks and synccheck reports zero errors. The paired benchmark has 180 cycles, no misses, matching endpoint accuracy and 25.5–39.3% lower mean cycle latency on the tested grids. This supports adopting residency as the CUDA CLI/GUI default, with explicit legacy rollback. See residency-results.md. Complete process/image/graphics residency is still not implemented.
