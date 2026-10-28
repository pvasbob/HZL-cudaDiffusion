# Persistent forecasting and latency work

## Scope and status

The `--cuda-resident 1` path is implemented and validated on the RTX 3080 host. Pointwise/workspace/control and GUI action tests pass; exercised GPU memory checks report zero errors/leaks, and matched measurements show 25.5–39.3% lower mean controller latency on three grids. It is now the default for CUDA CLI/GUI runs. Explicit `--cuda-resident 0` preserves the legacy comparison path; CPU stays nonresident. See [recorded results and limits](residency-results.md).

Local CPU C++20, C++23, minimal and ASan/UBSan tests pass. CUDA tests skip when device access is unavailable; host validation evidence is saved separately.

This change targets the repeated forecasting hot path identified by the Systems trace. It is not a claim of complete device residency for process/image/UI. The ongoing process and measurement ownership and chronological rules are unchanged: control still obtains concentration exclusively through image reconstruction. No prediction endpoint is adopted as process state.

## Changes

- `Experiment` owns an optimized forecasting workspace across automatic decisions. Reset builds its replacement transactionally; failed initialization leaves the existing experiment inspectable. No forecasting `cudaMalloc` or pinned allocations occur inside an optimized candidate/decision. Initial allocation is measured in initialization time. Graph rebuilds may still allocate driver resources if explicitly enabled; graphs remain off by default.
- Each decision updates the immutable reconstructed snapshot. A content-checked target cache avoids re-uploading an unchanged target, including correct invalidation after painted target edits. Same-domain checks reject reuse on a different grid/domain. Snapshot updates invalidate previous forecast and retained-best state.
- Uniform candidate seeds upload one temperature value; simplex candidates upload the coarse tile coefficients (four by default), from a reused pinned staging buffer. GPU code expands the controls and evaluates positive Arrhenius diffusivity. Arbitrary painted/full-map controls and finite candidates use a full temperature upload fallback; this is still CUDA execution, not a CPU solver fallback.
- CPU stability planning uses the maximum coefficient temperature and the existing monotone Arrhenius law to obtain Dmax. It shares exactly the same conservative step bound and interval division helper as the reference solver. Minimum-temperature validation rejects diffusivity underflow. GPU double `exp` may differ in last bits from CPU `exp`, so pointwise CPU/GPU comparisons and final control-error checks are required.
- Improving forecasts are retained by device-to-device copy; only the selected endpoint field is downloaded once at decision completion. The optimizer still receives one scalar objective per candidate, because the installed CPU derivative-free optimizer is sequential. A reused pinned scalar avoids a pageable return staging buffer.
- CUB transform-input iteration computes squared errors during reduction, eliminating the separate full squared-error buffer and producer kernel in the optimized mode. Diffusivity preparation adds a kernel; no reduction in total launch count or actual latency is claimed without measurement.
- Zero-weight CPU objective penalty loops are skipped. Nonzero heating/change/spatial penalties retain their original semantics. This common optimization also affects the legacy path, so compare both modes on the same rebuilt binary rather than attributing all differences against archived runs to residency.

## Instrumentation and bounds

Cycle CSVs now report `control_upload_bytes`, `forecast_download_bytes`, `forecast_device_allocations`, `snapshot_upload_bytes`, `target_upload_bytes`, and `graph_rebuilds` for the forecasting workspace. These are successful explicit allocation/copy counts and byte totals, not GPU hardware counters or whole-application totals. They exclude process/image allocations and internal driver graph resources. Pinned allocations happen only during workspace construction.

Expected resident invariants with an unchanged target: zero forecasting device-buffer allocations per automatic cycle; zero target-upload bytes; one selected forecast-field download. Coarse candidate uploads should be much smaller than one full map per candidate. Snapshot upload remains once per decision. Device storage is bounded to one workspace, one retained best field, two pinned buffers (64 coefficients and one scalar), CUB scratch and at most one cached graph. Each concurrently evaluated workspace needs separate ownership; no global cache or hidden mutable state is introduced.

Both modes use the measured global stencil/ordinary-launch default. Tiled kernels and graphs remain selectable and are included in primitive regression coverage. Workspace/result/target lifetimes and CPU mirror ownership remain explicit.

## Validation and matched measurement

From the normal GPU host terminal:

```sh
cmake --build --preset cuda
tools/validate_residency.sh build/cuda gpu-reports/residency-validation
```

The workflow requires a working GPU; it does not turn unavailable hardware into success. It performs a resident controller run, the expanded global/tiled/graph/coefficient/retention/target-cache variant suite, a resident closed-loop suite, Compute Sanitizer memcheck/synccheck, and an alternating legacy/resident benchmark. No Nsight Compute counter permissions are needed.

The benchmark performs three repeats on each of 65×49, 129×97 and 257×129, alternating mode order between repeats. It compares the same rebuilt executable at identical numerical/controller parameters, checks endpoint MSE agreement (relative 1e-5 or absolute 1e-13), verifies allocation and target-cache invariants, and records transfer-volume counters. It saves raw cycle histories, exact commands, environment/build metadata, binary hashes, subprocess wall time, startup timing, summary.csv, report.md and status.json. Endpoint tolerance is a comparison guard, not a physical-accuracy guarantee. Pointwise numerical tests use tighter tolerances.

Individual commands:

```sh
build/cuda/hzl-control --backend cuda --cuda-resident 1 --output run-resident
python3 tools/benchmark_residency.py --binary build/cuda/hzl-control \
  --repeats 3 --output gpu-reports/residency
```

## Remaining traffic and work

Host temperature maps still exist for manual editing, nonzero penalties and selected control publication; simplex currently also expands candidate host maps. The process solver, synthetic renderer/reconstruction wrapper, display snapshots and baseline comparison still allocate/transfer per cycle. Observation reconstruction returns a host snapshot which is then uploaded to the forecasting workspace. These are explicit remaining residency opportunities, not silently bypassed image processing. Fully resident process→measurement→reconstruction→forecast state, stream-based publication and graphics interop should be designed only after validating and measuring this change. The host measurements support adoption of this forecasting change; they do not validate the remaining residency work listed above.
