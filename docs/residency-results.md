# RTX 3080 forecasting residency results

Host workflow completed 2026-10-26. [Paired report](../examples/gpu-residency-validation/comparison/report.md), raw summaries/cycles, commands, build/device metadata, GUI-action outputs and sanitizer logs are preserved in [examples/gpu-residency-validation](../examples/gpu-residency-validation).

| Grid | Legacy mean cycle | Resident mean cycle | Mean reduction |
|---|---:|---:|---:|
| 65×49 | 6.937 ms | **4.386 ms** | **36.8%** |
| 129×97 | 37.466 ms | **27.894 ms** | **25.5%** |
| 257×129 | 55.205 ms | **33.499 ms** | **39.3%** |

Both modes ran on the same binary with matched model/controller settings, 60-evaluation simplex ceilings, global kernels and ordinary launches. Mode order alternates over three repeats. Each mean covers 27 steady cycles; the first cycle of each run is excluded from these means and retained in the raw logs. Across all 18 runs, **180 cycles had zero deadline misses**. Initialization and complete-process wall time are separately recorded, not silently omitted from the experiment data.

Forecast control uploads fell from about 1.53MB to 27.2KB per cycle at 65×49, 6.01MB to 101.9KB at 129×97, and 15.85MB to 267.0KB at 257×129. Retained-best selection reduced full forecast downloads to one field per decision: 25,480 / 100,104 / 265,224 bytes at the three grids. These counters cover forecasting, not process/image/UI traffic. The resident benchmark verifies zero forecasting buffer allocations during decisions and no unchanged-target reupload. One reconstructed snapshot upload per decision remains.

The host passed expanded global/tiled/graph/fused-reduction/coefficient/retention/cache tests, resident closed-loop tests, and resident GPU-backed GUI action smoke. Variant and resident-control memchecks report **zero errors and zero bytes leaked**; synccheck reports zero errors. Pointwise numerical tests pass, and recorded final endpoint MSE matches between modes to the printed precision for all benchmark cases. Small samples and deterministic synthetic observations do not establish production tail latency or calibrated physical performance.

## Selection

**Adopt resident forecasting for normal CUDA CLI/GUI runs.** `--backend cuda` now selects residency unless `cuda-resident` is explicitly set in the configuration or CLI. `--cuda-resident 0` retains the comparison/rollback path; CPU runs remain nonresident. Direct C++ callers of ControlConfig still choose the mode explicitly. Configuration/CLI override ordering remains respected and is tested without requiring GPU execution.

The generic matrix now follows the application default for CUDA. The historical global/tiled/graph sweep explicitly selects `cuda-resident=0` to preserve its experiment definition. The paired residency script always selects both modes explicitly, so adoption does not alter its comparison.

## Limits and next opportunities

This validates forecasting residency, not a completely GPU-resident application. The process solver, synthetic image path and display snapshots still allocate/transfer fields; observation reconstruction still returns a host snapshot that is uploaded once per decision. Candidate host maps are also expanded for current optimizer/control interfaces. These are concrete next opportunities, but should be measured separately so the gains above are not attributed to features that are not implemented. CUDA–OpenGL interop and hardware-counter analysis remain separate work. Global/ordinary launches stay selected; this paired experiment did not compare every optimized tiled/graph combination at full-controller level.
