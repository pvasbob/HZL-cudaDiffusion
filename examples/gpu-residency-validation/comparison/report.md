# Paired CUDA residency comparison

Same binary and parameters; mode order alternates between repeats. First decisions are excluded from steady means and retained in raw logs. Transfer counters cover forecasting only. No profiler is attached.

| Grid | Mode | Mean cycle ms | Maximum steady ms | Misses | Control upload bytes/cycle | Forecast download bytes/cycle |
|---|---|---:|---:|---:|---:|---:|
| 65x49 | legacy | 6.937 | 9.229 | 0 | 1528800 | 48129 |
| 65x49 | resident | 4.386 | 6.035 | 0 | 27248 | 25480 |
| 129x97 | legacy | 37.466 | 62.052 | 0 | 6006240 | 255821 |
| 129x97 | resident | 27.894 | 46.709 | 0 | 101872 | 100104 |
| 257x129 | legacy | 55.205 | 76.931 | 0 | 15854501 | 766203 |
| 257x129 | resident | 33.499 | 53.266 | 0 | 266985 | 265224 |

Endpoint MSE agrees within relative 1e-5 or absolute 1e-13. This is a controller-comparison guard, not a replacement for pointwise CPU/GPU tests. Workspace allocation and target-reuse invariants passed. Startup and total subprocess timings remain in per-run metadata/summary. No default is changed automatically.
