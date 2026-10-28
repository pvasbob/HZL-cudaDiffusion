# Host sanitizer, timing and Systems evidence — 2026-10-26

tools/validate_cuda.sh was executed on the RTX 3080 host. Saved logs and cuda_api_sum/cuda_gpu_kern_sum summaries exported from pipeline.nsys-rep using nsys stats are preserved here.

- Full CTest: 10/10 passed, none skipped (host terminal result; 1.25s).
- Variant memcheck: zero errors. Variant synccheck: zero errors.
- Observation racecheck: zero hazards, errors and warnings.
- Unprofiled comparison.csv: small workloads only, five samples per variant, numerical errors ~1e-35 versus CPU. These timings exclude final forecast download (reported separately).
- Systems capture succeeded; CPU sampling/context-switch tracing unsupported, so CPU stacks/context-switch evidence is absent.
- Kernel summary: 1967 global diffusion kernels, median 2.72us; 300 squared-error and 600 CUB reduction kernels. Total reported GPU kernel time ~6.98ms over the five-decision profile run.
- API summary: first cudaMalloc maximum ~129ms dominates startup-inclusive allocation totals. This is not evidence that each allocation costs that much. Copy and launch API durations include tool overhead and are not directly additive with GPU durations.
- Nsight Compute: ERR_NVGPUCTRPERM, no kernels profiled. Its instrumented application timings are NOT benchmark results. Kernel hardware counters remain unavailable; no driver change was made.

Sanitizer scope is the named executables, not every application path or arbitrary configuration. Broader GPU control workloads, repeatable speedup measurements, live GUI interaction and Nsight Compute counters remain to be assessed.
