# Repeated RTX 3080 control benchmarks

Host measurements and raw result files are preserved with the project. Ten configurations, three repeats each, ten control intervals per run: 300 cycles total. All 300 cycles have zero deadline misses. Steady summaries exclude the first decision of each run: 270 steady samples total, 27 per configuration. Initialization, baseline simulation, file output and visualization are outside control-cycle timing; per-process wall time including setup/baseline/output is retained separately.

CPU and GPU use matching grid/model/seed/control configurations and give matching endpoint MSE to floating-point tolerance. CPU measurements were taken earlier on the same host; GPU measurements were taken later in the host terminal. These are indicative ratios between recorded experiments, not a simultaneous, randomized hardware speedup study. Ratio >1 means the GPU control cycle was faster. This matrix uses the global stencil and ordinary launches; it does not establish the best GPU implementation.

| Grid | Endpoint (s) | Budget | Filter radius | Method | CPU mean (ms) | GPU mean (ms) | CPU/GPU ratio | GPU max (ms) |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| 33×25 | 5 | 12 | 0 | finite | 0.36 | 1.01 | 0.36 | 1.50 |
| 65×49 | 10 | 60 | 0 | simplex | 11.23 | 7.30 | 1.54 | 9.06 |
| 65×49 | 10 | 60 | 2 | simplex | 10.34 | 7.26 | 1.42 | 8.95 |
| 65×49 | 10 | 60 | 0 | finite | 7.10 | 5.08 | 1.40 | 6.27 |
| 129×97 | 30 | 12 | 0 | finite | 58.03 | 7.90 | 7.34 | 10.83 |
| 129×97 | 30 | 60 | 0 | simplex | 359.09 | 38.79 | 9.26 | 58.11 |
| 65×49 | 120 | 60 | 0 | simplex | 90.14 | 29.08 | 3.10 | 47.76 |
| 33×25 | 600 | 12 | 0 | finite | 5.14 | 6.17 | 0.83 | 10.29 |
| 256×256 | 5 | 12 | 0 | finite | 270.74 | 26.44 | 10.24 | 32.53 |
| 257×129 | 5 | 60 | 0 | simplex | 524.46 | 61.78 | 8.49 | 87.77 |

The 257×129 case has 0.5s intervals. GPU mean latency is ~61.8ms and the maximum is ~87.8ms, with zero misses; the earlier CPU runs averaged ~524ms and recorded 15 steady-state misses. Default 65×49 simplex control averaged ~7.30ms at a one-second interval, about 137× physical/control-time factor. Final default MSE is ~6.7e-10, compared with the fixed-temperature baseline ~3.14e-4. The 256×256 case averaged ~26.4ms on GPU versus ~270.7ms on CPU. Small workloads can favor CPU: 33×25 finite control at a five-second endpoint was ~0.36ms on CPU versus ~1.01ms on GPU.

Filtering clean images still worsens the final default error (~1.49e-5), so bypass remains the default. Small samples and a deterministic synthetic model do not establish production tail latency or calibrated physical performance.

Reproduce the comparison from stored data:

```sh
python3 tools/summarize_backends.py examples/control-benchmarks/summary.csv \
  examples/gpu-control-benchmarks/summary.csv --output examples/backend-comparison.csv
```

Remaining acceptance: live desktop interactions, broader matched global/tiled/graph comparisons, Nsight Compute hardware counters (permission-blocked), and resulting hardware-specific optimization/release decisions. No new driver installation is indicated by these results.
