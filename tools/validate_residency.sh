#!/usr/bin/env bash
set -euo pipefail
build_dir=${1:-build/cuda}
report_dir=${2:-gpu-reports/residency-validation}
mkdir -p "$report_dir"
nvidia-smi > "$report_dir/nvidia-smi.txt"
"$build_dir/hzl-control" --backend cuda --cuda-resident 1 --width 17 --height 13 --endpoint 1 --budget 8 --output "$report_dir/smoke"
if [[ -x "$build_dir/hzl-gui" ]]; then
  "$build_dir/hzl-gui" --backend cuda --cuda-resident 1 --interaction-smoke --width 17 --height 13 --endpoint 2 --interval 1 --budget 8 --method finite --output "$report_dir/gui-actions"
fi
ctest --test-dir "$build_dir" -R 'cuda-(variants|resident-control-validation)' --output-on-failure
compute-sanitizer --error-exitcode 1 --tool memcheck --leak-check full "$build_dir/hzl-cuda-variant-tests" > "$report_dir/memcheck.txt" 2>&1
compute-sanitizer --error-exitcode 1 --tool synccheck "$build_dir/hzl-cuda-variant-tests" > "$report_dir/synccheck.txt" 2>&1
compute-sanitizer --error-exitcode 1 --tool memcheck --leak-check full "$build_dir/hzl-control-tests" --cuda --resident > "$report_dir/control-memcheck.txt" 2>&1
python3 tools/benchmark_residency.py --binary "$build_dir/hzl-control" --output "$report_dir/comparison"
printf '%s\n' 'Residency validation and comparison completed; inspect comparison/report.md.'
