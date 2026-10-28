# Full-controller CUDA implementation comparison

Three representative grids, rotating variant order between repeats. Times include graph rebuilds during optimization; first decisions are excluded from steady means and retained in raw logs. No profiler is attached. These samples do not establish production tail latency.

| Grid | Implementation | Repeats | Mean cycle (ms) | Maximum steady cycle (ms) | Misses |
|---|---|---:|---:|---:|---:|
| 65x49 | global, graphs=0 | 3 | 7.412 | 9.345 | 0 |
| 65x49 | global, graphs=1 | 3 | 8.150 | 11.232 | 0 |
| 65x49 | tiled, graphs=0 | 3 | 7.743 | 9.530 | 0 |
| 65x49 | tiled, graphs=1 | 3 | 8.217 | 10.843 | 0 |
| 129x97 | global, graphs=0 | 3 | 38.929 | 65.175 | 0 |
| 129x97 | global, graphs=1 | 3 | 47.320 | 73.187 | 0 |
| 129x97 | tiled, graphs=0 | 3 | 41.646 | 62.999 | 0 |
| 129x97 | tiled, graphs=1 | 3 | 50.419 | 77.869 | 0 |
| 257x129 | global, graphs=0 | 3 | 60.823 | 81.597 | 0 |
| 257x129 | global, graphs=1 | 3 | 64.340 | 87.545 | 0 |
| 257x129 | tiled, graphs=0 | 3 | 62.636 | 83.950 | 0 |
| 257x129 | tiled, graphs=1 | 3 | 65.938 | 90.338 | 0 |

All endpoint MSE values agree within the stated comparison tolerance. No default is changed automatically: assess accuracy, repeated latency, setup and deadline data before selecting a workload-specific implementation.
