# CUDA implementation selection — RTX 3080

The host sweep completed successfully before the Ctrl+C shown after the shell prompt. Preserved [report](../examples/gpu-variant-benchmarks/report.md), [raw summary](../examples/gpu-variant-benchmarks/summary.csv), per-run cycles/metadata/commands and `status.json` record 36 experiments, 360 control cycles, and zero deadline misses. Endpoint errors agree across all four variants.

| Grid | Global launches | Global graphs | Tiled launches | Tiled graphs |
|---|---:|---:|---:|---:|
| 65×49 | **7.412 ms** | 8.150 ms | 7.743 ms | 8.217 ms |
| 129×97 | **38.929 ms** | 47.320 ms | 41.646 ms | 50.419 ms |
| 257×129 | **60.823 ms** | 64.340 ms | 62.636 ms | 65.938 ms |

Values are mean steady full-controller cycle latencies across three repeats, nine steady decisions per repeat. Implementation order rotates across repeats; no profiler is attached. The maximum across all 360 decisions, including first decisions after initialization, is approximately 93.6ms. Initialization remains separately reported.

Decision: **retain global-memory diffusion with ordinary launches as the default** (`cuda-kernel=global`, `cuda-graphs=0`). Tiling adds roughly 3–7% to mean cycle latency here, and global graphs add roughly 6–22%. Tiled graphs have the highest mean latency on all three tested grids. Alternatives remain explicitly selectable for further experiments and other workloads.

This is a decision based on mean full-cycle evidence, not a claim of universal superiority. For example, the 129×97 tiled-launch maximum steady cycle is slightly lower than the global-launch maximum in this sample. Short runs do not establish tail-latency guarantees, nor does this sweep cover every grid/horizon/filter/block configuration. The graph experiment includes optimization-time graph rebuilds; a repeated fixed-control microbenchmark is a different workload. No counter-level cause is asserted because Nsight Compute remains permission-blocked.

The benchmarked numerical/controller sources are unchanged by the subsequent GUI interaction-handler refactor. The new display-independent smoke test shares real GUI keyboard/mouse action code and checks manual temperature application, editing guards, forecast isolation, reset, presets, filtering, target regeneration, endpoint editing and saving. It passes on C++20, host C++23 and ASan/UBSan CPU paths (leak detection disabled under tracing). Actual OS event delivery is outside this test; native window startup was confirmed after the GTK fix.
