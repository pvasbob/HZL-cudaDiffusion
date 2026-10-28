# Performance evidence and decisions

## Measured on this host

Hardware/software: Intel i7-8700K, GCC 11.4, Linux x86_64, CMake 3.22.1 Release, OpenCV 4.5.4. NVCC 12.1 compilation is available but NVIDIA device access is not. Data live in examples/control-benchmarks (30 runs over ten representative workloads), examples/comparison-cpu.csv, examples/cpu-benchmark.csv and examples/render-cpu.csv. Each control run includes ten physical intervals; its first cycle is retained but excluded from steady-state summaries. Three repeats are available per case. This is preliminary evidence with small samples, not a production latency certification.

The 257×129 CPU workload with a 5s endpoint, 0.5s intervals and 60-evaluation ceiling averaged about 522–528ms, missing five of nine steady deadlines in each repeat (15 total). It is not realtime under that configuration. GPU execution or a different accuracy/budget/algorithm tradeoff must be measured before claiming otherwise.

The default 65×49 reachable target, endpoint 10s, interval 1s and budget 60, produced final MSE 6.7e-10 with OpenCV simplex and 1.57e-8 with finite candidates; fixed 800 K produced 3.14e-4. With the same 60-evaluation ceiling both improve the baseline (finite search exhausts its 38 candidates, while simplex uses 60); this one case favors simplex's residual. Radius-2 filtering on clean synthetic images worsened simplex's residual to 1.49e-5. Do not enable smoothing merely for a CUDA demonstration.

The direct two-dimensional CPU Gaussian and OpenCV's separable Gaussian agree numerically. The library path is faster on the measured small grids/radius 2, including decode/conversion/allocation costs, and is selected for normal CPU processing. The custom direct path remains for validation and controlled comparisons. Shared CUDA tiling is not assumed faster.

Headless rendering (1020×900 dashboard, three warm-ups, twenty measured frames) averaged approximately 6.2ms, versus roughly 11ms average steady control latency in the default case. Synchronous rendering can therefore materially affect update throughput. GTK window presentation/event handling is unavailable here and is not included in that number. The displayed control deadline is for the headless control portion; the GUI remains a stepped development tool, not a deadline-certified realtime pipeline.

## Timing boundaries

- `hzl-benchmark`: solver wall latency includes validation, planning, allocation and transfers; file output excluded. CUDA events bracket the integration launch interval, including launch gaps, not the sum of isolated kernel durations. Horizons 5/30/120/600s use identical physical/numerical settings except duration.
- `hzl-compare`: CPU forecast plus MSE versus resident CUDA forecast plus CUB MSE. Workspace construction is separate setup time; one warm-up precedes repeated evaluations. Final full-field download is reported separately. This matches a resident scalar-objective workflow; the controller additionally downloads every improving candidate and includes that in full wall time. Decode/filter measurements include image decoding and allocations/transfers where applicable. Only meaningful equivalent operations are compared.
- `hzl-control`: entire decision from measurement render through the selected temperature and physical interval advancement. Optimization time contains forecasting time. Independent baseline, output, extra validation/display reconstruction and GUI are excluded. Initialization is recorded separately. Every cycle retains actual latency, numerical step count/dt, reconstruction/mass error, predicted endpoint MSE, budget, status and deadline miss.
- Matrix script: raw per-run timings and metadata, steady mean/median/nearest-rank p95/max, deadline misses, RTF and final errors, plus subprocess wall time including setup, baseline and I/O. No slowing virtual time is used to pass a deadline. `--paced` can sleep but never changes measured latency or suppresses a miss.

Small samples make p95 approach maximum. All wall timings are subject to scheduling and thermal variability. Repeat and expand samples for tail-latency acceptance.

## CUDA investigation map

| Concern | Implemented or decision |
|---|---|
| Coalescing/layout | Row-major one-species double fields; 32 adjacent x lanes per warp. Single-channel image with explicit byte stride; 2D transfer handles padding. AoS/SoA/AoSoA are irrelevant for one scalar channel/species and no conversion is added. |
| Global versus shared stencil | Global correctness baseline and 32×8 tiled stencil with a one-cell concentration/D halo. Cooperative halo load; all threads reach barriers before output bounds checks. Both selectable. Host functional/sanitizer checks pass; initial small-workload timings are recorded, with wider profiling pending. |
| Gaussian tiling/halos | Shared-memory decode+filter tile, radius 0–8; replicated boundary halos; odd/tiny/nonaligned dimensions in GPU tests. Normal CPU uses separable OpenCV. A separable CUDA/NPP alternative requires measured radius-dependent benefit. |
| Bank conflicts/padding | Double-precision shared rows have 34 entries for the PDE halo. No speculative extra padding is selected. Nsight bank-conflict data must justify padding; filter shared layout follows the requested radius. |
| Block sizes/warp mapping | 32×8 maps x to coalesced warps and bounds shared memory (max image filter tile 48×24 doubles, 9KiB). Other block shapes are deferred until hardware occupancy/register data are available, rather than selecting an invented winner. |
| Registers/occupancy/spilling | No register caps or launch-bound guesses. Use Nsight Compute and compiler resource reports on RTX 3080 before changing. |
| Reuse and traffic | Snapshot and target resident for a decision; ping-pong buffers, diffusivity, error and CUB scratch reused across all evaluations. D uploaded per candidate, input reset device-to-device. Only scalar MSE and improving forecast fields return to host. |
| Reductions/fusion | CUB sum of squared errors rather than handwritten reduction; image decode+filter fused. Further stencil+objective fusion would require careful final-step/reduction design and evidence. |
| Graphs/launch overhead | Ordinary launches and cached graphs selectable. Graphs rebuild when integration step count or dt changes, so optimization can defeat reuse. Cache holds one graph only. Cap capture at 100,000 steps. Compare total setup and cycle cost, not just repeated graph-launch time. |
| Streams/synchronization | Nondefault stream per workspace, explicit completion checks, no global synchronization in each integration step. Workspace is not shared concurrently. Serial candidate evaluation is correct; candidate streams/batching need capacity/bandwidth measurements before introduction. |
| Transfers/pinned memory | Candidate snapshot/target residency removes repeated transfers. Temperature diffusivity still generated on CPU; observation and process currently transfer fields. Pageable transfers remain safe with checked completion; pinned pools and fully resident image/process states are deferred pending Nsight transfer evidence. |
| Allocation/lifetimes | RAII buffers/events/streams/graph and reusable CUB storage per decision. No device allocations per candidate except graph-cache rebuild resources. CPU planning/map/output vectors and per-cycle image/process allocation remain optimization opportunities. Counts and graph storage are bounded. |
| Precision | Double baseline for correctness. RTX 3080's limited FP64 throughput makes float/mixed precision a potentially important experiment. No float production path without positivity/convergence/control-error measurements. |
| Solver algorithm | Explicit conservative finite volume with stable substeps. Refinement can make step count dominant. Implicit/ADI/multigrid methods require variable-D conservative design and accuracy benchmarks; no inappropriate FFT/BLAS dependency is forced in. |
| Optimizer scaling | Four tile variables, hard forecast cap, finite versus library simplex comparisons. Full per-pixel finite differences are excluded. Adjoint/sensitivity methods and schedules are future work for large control spaces. |
| Visualization | OpenCV CPU presentation, headless render measurement and separate control timing. CUDA–OpenGL interop and asynchronous publishing are deferred until device/desktop profiling; no low-latency interop claim. |

The global CUDA stencil remains the default: it passes host tests and has repeated full-cycle evidence. Small unprofiled global/tiled/graph comparisons exist, but do not establish a broader implementation winner. Tiled/graph variants must pass all numerical tests and demonstrate end-to-end benefit before changing that default. Default CPU filtering does follow measured library evidence.

## Reproduction

Run `tools/benchmark_matrix.py --repeats 3 --output benchmarks`. Use `--quick` for small cases and `--cuda --binary build/cuda/hzl-control` on the GPU machine. Field exports are large, so the matrix removes per-run fields after summarizing unless `--keep-fields` is requested; cycle CSVs, configuration metadata, exact commands and stdout/stderr remain.

`hzl-compare` without `--quick` covers 32², 129×97, 256², 513×257 and 1024², radii 0/1/2/4/8 and 5/30s forecasts. Large explicit CPU reference forecasts can be expensive; this is intentional and must not be mistaken for a stalled GPU benchmark. The representative controller matrix avoids an unnecessary full Cartesian product.

Run `tools/validate_cuda.sh build/cuda gpu-reports` for device availability, CTest, Compute Sanitizer memory/synchronization/race checks, a quick comparison, Nsight Systems pipeline trace and Nsight Compute kernels. Nsight may require driver profiling permissions; record unavailable counters instead of inventing results. Hardware-specific measurements, float precision alternatives, other block shapes and interop remain pending as explicitly stated above.


## First host GPU functional run

All ten CTest cases passed on the RTX 3080, followed by a successful default CUDA controller run. The preserved cycles/metadata in examples/gpu-host-smoke show mean 8.90ms, maximum 11.72ms and zero one-second deadline misses across ten cycles; final MSE matches the CPU case (~6.7e-10). Initialization took ~275ms separately. This is a small functional run; Host sanitizer and Systems evidence is now recorded below; Nsight Compute counters and matched repeated wider comparisons are still required.


Updated host evidence: [sanitizer logs, unprofiled comparisons and Systems summaries](../examples/gpu-host-validation/validation.md). Variant memcheck/synccheck and observation racecheck report zero errors/hazards; Nsight Systems captured the GPU pipeline. Nsight Compute alone failed with ERR_NVGPUCTRPERM. Its printed timings must not be used as ordinary benchmark measurements. The repeated GPU matrix is now recorded in gpu-results.md; live GUI interaction remains unverified.


## Repeated host GPU matrix

See [GPU results](gpu-results.md) for ten cases × three repeats and raw-data links. No deadline misses occurred in 300 GPU cycles, including first decisions after initialization. CPU/GPU endpoint errors agree. The GPU removes the recorded CPU deadline misses for the 257×129 configuration; it does not win every workload. Performance ratios use earlier CPU measurements, so they are indicative rather than a controlled simultaneous comparison. Full-cycle global/tiled/graph selection and counter-guided optimization remain separate work.


## Full-controller variant experiment

`tools/benchmark_variants.py` now provides a bounded experiment: 65×49 (10s), 129×97 (30s), 257×129 (5s), 60-evaluation simplex ceiling, ten intervals each, four global/tiled/graph combinations, default three repeats. The variant order rotates per repeat. All normal stage timings and per-run environment/commands are retained, including setup, and graph construction/reconstruction occurs naturally inside the measured control cycle. The final report requires endpoint MSE agreement within relative 1e-6 or absolute 1e-14 across implementations. Such tolerance is a comparison guard, not a general physical-accuracy guarantee. Primitive CPU/GPU tests remain the finer-grained numerical reference.

The sweep has now completed on the GPU host: 36 experiments and 360 cycles, zero misses, matching endpoint errors. Global memory with ordinary launches has the lowest mean full-cycle latency on all three grids, supporting the existing default. See implementation-selection.md and examples/gpu-variant-benchmarks for the report and raw evidence.
