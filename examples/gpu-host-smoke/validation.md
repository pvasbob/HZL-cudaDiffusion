# RTX 3080 host validation, 2026-10-26

Executed in the RTX 3080 host terminal. The resulting run-gpu/cycles.csv and metadata.txt are preserved here.

Commands: `cmake --preset cuda`, `cmake --build --preset cuda`, `ctest --preset cuda --output-on-failure`, `build/cuda/hzl-control --backend cuda --output run-gpu`.

CTest: 10/10 passed, none skipped, total 1.62 seconds. This includes cuda-variants, cuda-control-validation, cuda-observation-validation and cuda-validation.

GPU: NVIDIA GeForce RTX 3080, compute capability 8.6, CUDA runtime 12.1; host nvidia-smi reports driver 580.178.04. The default 65x49 controller ran ten decisions, 60 evaluations each, mean 8.899704 ms, maximum 11.719348 ms, zero one-second deadline misses, final MSE approximately 6.70375e-10 versus fixed baseline 3.14446e-4. Initial setup took approximately 275 ms, excluded from cycle timing. These ten cycles are a smoke/functional result, not a robust latency or speedup benchmark.

Compute Sanitizer, Nsight profiling, broader GPU benchmarks and live desktop interaction remain unverified.
