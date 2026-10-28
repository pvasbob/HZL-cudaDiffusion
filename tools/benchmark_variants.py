#!/usr/bin/env python3
"""Compare full controller CUDA implementations, including dynamic graph rebuild costs."""
import argparse
import atexit
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', default='build/cuda/hzl-control')
parser.add_argument('--output', type=Path, default=Path('gpu-reports/variants'))
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
if args.repeats < 1:
    parser.error('--repeats must be positive')
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=True)
for old_report in ('report.md', 'summary.csv'):
    (root/old_report).unlink(missing_ok=True)
completed = False
(root/'status.json').write_text(json.dumps({'status':'running'})+'\n')
def record_status():
    (root/'status.json').write_text(json.dumps({'status':'passed' if completed else 'failed_or_interrupted'})+'\n')
atexit.register(record_status)
variants = [('global', '0'), ('global', '1'), ('tiled', '0'), ('tiled', '1')]
matrix = Path(__file__).with_name('benchmark_matrix.py')
rows = []
for repeat in range(args.repeats):
    # Rotate implementation order between repeats to reduce fixed-order bias.
    for offset in range(len(variants)):
        kernel, graphs = variants[(repeat+offset) % len(variants)]
        destination = root/f'repeat-{repeat:02d}'/f'{kernel}-graphs-{graphs}'
        command = [sys.executable, str(matrix), '--binary', args.binary, '--cuda',
                   '--suite', 'variants', '--repeats', '1', '--output', str(destination),
                   '--cuda-kernel', kernel, '--cuda-graphs', graphs, '--cuda-resident', '0']
        subprocess.run(command, check=True)
        with (destination/'summary.csv').open() as stream:
            for row in csv.DictReader(stream):
                row.update(kernel=kernel, graphs=graphs, repeat=str(repeat))
                rows.append(row)
        with (root/'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
keys = ('width', 'height', 'endpoint_s', 'budget', 'filter_radius', 'method')
report = ['# Full-controller CUDA implementation comparison', '',
          'Three representative grids, rotating variant order between repeats. '
          'Times include graph rebuilds during optimization; first decisions are excluded '
          'from steady means and retained in raw logs. No profiler is attached. '
          'These samples do not establish production tail latency.', '',
          '| Grid | Implementation | Repeats | Mean cycle (ms) | Maximum steady cycle (ms) | Misses |',
          '|---|---|---:|---:|---:|---:|']
for case in dict.fromkeys(tuple(r[k] for k in keys) for r in rows):
    group = [r for r in rows if tuple(r[k] for k in keys) == case]
    reference = next(r for r in group if r['kernel']=='global' and r['graphs']=='0')
    for kernel, graphs in variants:
        runs = [r for r in group if (r['kernel'],r['graphs'])==(kernel,graphs)]
        for run in runs:
            if not math.isclose(float(run['final_mse']), float(reference['final_mse']), rel_tol=1e-6, abs_tol=1e-14):
                raise SystemExit('Accuracy mismatch across variants; inspect raw fields before selecting an implementation')
        mean = statistics.mean(float(r['mean_ms']) for r in runs)
        maximum = max(float(r['max_ms']) for r in runs)
        misses = sum(int(r['deadline_misses']) for r in runs)
        report.append(f'| {case[0]}x{case[1]} | {kernel}, graphs={graphs} | {len(runs)} | {mean:.3f} | {maximum:.3f} | {misses} |')
report += ['', 'All endpoint MSE values agree within the stated comparison tolerance. '
           'No default is changed automatically: assess accuracy, repeated latency, setup '
           'and deadline data before selecting a workload-specific implementation.', '']
(root/'report.md').write_text('\n'.join(report))
(root/'parameters.json').write_text(json.dumps({'repeats':args.repeats, 'variants':variants,
                                              'cuda_resident':0, 'accuracy_relative_tolerance':1e-6,
                                              'accuracy_absolute_tolerance':1e-14}, indent=2)+'\n')
completed = True
print(root/'report.md')
