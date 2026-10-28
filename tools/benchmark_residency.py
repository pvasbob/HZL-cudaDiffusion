#!/usr/bin/env python3
"""Paired legacy/resident CUDA controller comparison on the same current binary."""
import argparse
import atexit
import csv
import json
import hashlib
import math
from pathlib import Path
import statistics
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', default='build/cuda/hzl-control')
parser.add_argument('--output', type=Path, default=Path('gpu-reports/residency'))
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
if args.repeats < 1:
    parser.error('--repeats must be positive')
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=True)
binary=Path(args.binary).resolve()
(root/'parameters.json').write_text(json.dumps({'binary':str(binary), 'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(), 'repeats':args.repeats, 'accuracy_relative_tolerance':1e-5, 'accuracy_absolute_tolerance':1e-13},indent=2)+'\n')
for name in ('summary.csv','report.md'):
    (root/name).unlink(missing_ok=True)
completed = False
(root/'status.json').write_text(json.dumps({'status':'running'})+'\n')
def status():
    (root/'status.json').write_text(json.dumps({'status':'passed' if completed else 'failed_or_interrupted'})+'\n')
atexit.register(status)
rows = []
for repeat in range(args.repeats):
    for resident in ([0,1] if repeat%2==0 else [1,0]):
        destination=root/f'repeat-{repeat:02d}'/f'resident-{resident}'
        command=[sys.executable,str(Path(__file__).with_name('benchmark_matrix.py')),
                 '--binary',args.binary,'--cuda','--suite','variants','--repeats','1',
                 '--cuda-resident',str(resident),'--output',str(destination)]
        subprocess.run(command,check=True)
        for case in sorted(destination.glob('case-*')):
            with (case/'cycles.csv').open() as stream:
                cycles=list(csv.DictReader(stream))
            if resident:
                for cycle in cycles:
                    if int(cycle['forecast_device_allocations'])!=0:
                        raise SystemExit(f'Forecast allocations inside resident decision: {case}')
                    if int(cycle['target_upload_bytes'])!=0:
                        raise SystemExit(f'Unchanged target uploaded in resident decision: {case}')
        with (destination/'summary.csv').open() as stream:
            for row in csv.DictReader(stream):
                matching=[]
                for candidate in sorted(destination.glob('case-*')):
                    with (candidate/'cycles.csv').open() as trace:
                        trace_rows=list(csv.DictReader(trace))
                    # Direct lookup by the fixed matrix order avoids dimensions encoded in directory names.
                    command_data=json.loads((candidate/'command.json').read_text())
                    if command_data[command_data.index('--width')+1]==row['width'] and command_data[command_data.index('--height')+1]==row['height']:
                        matching=trace_rows[1:]
                        break
                if not matching:
                    raise SystemExit('Missing raw cycle data')
                row.update(resident=str(resident),repeat=str(repeat),
                           mean_control_upload_bytes=statistics.mean(int(c['control_upload_bytes']) for c in matching),
                           mean_forecast_download_bytes=statistics.mean(int(c['forecast_download_bytes']) for c in matching),
                           mean_forecast_device_allocations=statistics.mean(int(c['forecast_device_allocations']) for c in matching))
                rows.append(row)
        with (root/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
report=['# Paired CUDA residency comparison','',
        'Same binary and parameters; mode order alternates between repeats. First decisions are excluded from steady means and retained in raw logs. Transfer counters cover forecasting only. No profiler is attached.','',
        '| Grid | Mode | Mean cycle ms | Maximum steady ms | Misses | Control upload bytes/cycle | Forecast download bytes/cycle |',
        '|---|---|---:|---:|---:|---:|---:|']
for width,height in [('65','49'),('129','97'),('257','129')]:
    group=[r for r in rows if r['width']==width and r['height']==height]
    baseline=next(r for r in group if r['resident']=='0')
    for mode in ('0','1'):
        runs=[r for r in group if r['resident']==mode]
        for run in runs:
            if not math.isclose(float(run['final_mse']),float(baseline['final_mse']),rel_tol=1e-5,abs_tol=1e-13):
                raise SystemExit('Endpoint accuracy mismatch; inspect raw results before selecting the resident path')
        report.append(f"| {width}x{height} | {'resident' if mode=='1' else 'legacy'} | {statistics.mean(float(r['mean_ms']) for r in runs):.3f} | {max(float(r['max_ms']) for r in runs):.3f} | {sum(int(r['deadline_misses']) for r in runs)} | {statistics.mean(float(r['mean_control_upload_bytes']) for r in runs):.0f} | {statistics.mean(float(r['mean_forecast_download_bytes']) for r in runs):.0f} |")
report+=['','Endpoint MSE agrees within relative 1e-5 or absolute 1e-13. This is a controller-comparison guard, not a replacement for pointwise CPU/GPU tests. Workspace allocation and target-reuse invariants passed. Startup and total subprocess timings remain in per-run metadata/summary. No default is changed automatically.','']
(root/'report.md').write_text('\n'.join(report)); completed=True
print(root/'report.md')
