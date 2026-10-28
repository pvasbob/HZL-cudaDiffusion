#!/usr/bin/env bash
set -euo pipefail
# Run on the host with a working NVIDIA driver. Never turn absent GPU into success.
build_dir=${1:-build/cuda}
report_dir=${2:-gpu-reports}
mkdir -p "$report_dir"
status_file="$report_dir/validation-status.txt"
printf '%s\n' 'status=running' > "$status_file"
nvidia-smi > "$report_dir/nvidia-smi.txt"
"$build_dir/hzl-control" --backend cuda --endpoint 1 --width 17 --height 13 --budget 8 --output "$report_dir/smoke"
ctest --test-dir "$build_dir" --output-on-failure
compute-sanitizer --error-exitcode 1 --tool memcheck "$build_dir/hzl-cuda-variant-tests" > "$report_dir/memcheck.txt" 2>&1
compute-sanitizer --error-exitcode 1 --tool synccheck "$build_dir/hzl-cuda-variant-tests" > "$report_dir/synccheck.txt" 2>&1
compute-sanitizer --error-exitcode 1 --tool racecheck "$build_dir/hzl-observation-tests" --cuda > "$report_dir/racecheck.txt" 2>&1
"$build_dir/hzl-compare" --quick --cuda > "$report_dir/comparison.csv"
nsys profile --force-overwrite true -o "$report_dir/pipeline" "$build_dir/hzl-control" --backend cuda --endpoint 5 --output "$report_dir/profile-run"
printf '%s\n' 'functional=passed' 'sanitizer_variant_memcheck=passed' 'sanitizer_variant_synccheck=passed' 'sanitizer_observation_racecheck=passed' 'unprofiled_comparison=passed' 'nsys=captured' > "$status_file"
# Profiling perturbs timings: keep this output separate from comparison.csv.
ncu_result=0
ncu --set full --target-processes all --force-overwrite -o "$report_dir/kernels" "$build_dir/hzl-compare" --quick --cuda > "$report_dir/ncu.log" 2>&1 || ncu_result=$?
if grep -q 'ERR_NVGPUCTRPERM' "$report_dir/ncu.log"; then
  printf '%s\n' 'ncu=permission_denied' 'status=partial' >> "$status_file"
  printf '%s\n' 'Nsight Compute counters are blocked; functional tests, sanitizers and the Systems trace completed. See ncu.log and validation-status.txt.'
  exit 2
elif [[ "$ncu_result" -ne 0 ]] || grep -Eq 'No kernels were profiled|==ERROR==' "$report_dir/ncu.log"; then
  printf '%s\n' 'ncu=failed' 'status=partial' >> "$status_file"
  cat "$report_dir/ncu.log"
  exit 2
fi
printf '%s\n' 'ncu=captured' 'status=complete_for_script_scope' >> "$status_file"
printf '%s\n' 'Validation and profiling reports saved. Profiler-run wall timings are not benchmark results.' 
