#!/usr/bin/env python3
"""Compare matching recorded CPU/GPU matrix cases without rerunning benchmarks."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('cpu', type=Path, help='CPU summary.csv')
parser.add_argument('gpu', type=Path, help='GPU summary.csv')
parser.add_argument('--output', type=Path, default=Path('backend-comparison.csv'))
args = parser.parse_args()
keys = ('width','height','endpoint_s','budget','filter_radius','method')

def read(path):
    grouped = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            grouped[tuple(row[key] for key in keys)].append(row)
    return grouped

def mean(rows):
    count = sum(int(r['steady_samples']) for r in rows)
    if not count:
        raise ValueError('No steady-state samples')
    return sum(float(r['mean_ms'])*int(r['steady_samples']) for r in rows)/count

cpu, gpu = read(args.cpu), read(args.gpu)
if cpu.keys() != gpu.keys():
    raise SystemExit('Case sets differ; compare matching experiment configurations')
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow([*keys, 'cpu_mean_ms','gpu_mean_ms','cpu_over_gpu_ratio',
                     'cpu_steady_misses','gpu_steady_misses','gpu_max_ms',
                     'gpu_final_mse','fixed_baseline_mse'])
    for key in cpu:
        a, b = cpu[key], gpu[key]
        if not math.isclose(float(a[0]['final_mse']), float(b[0]['final_mse']), rel_tol=1e-9, abs_tol=1e-18):
            raise ValueError(f'Endpoint accuracy differs: {key}')
        cm, gm = mean(a), mean(b)
        writer.writerow([*key, cm, gm, cm/gm,
                         sum(int(r['deadline_misses']) for r in a),
                         sum(int(r['deadline_misses']) for r in b),
                         max(float(r['max_ms']) for r in b),
                         b[0]['final_mse'], b[0]['fixed_baseline_mse']])
print(args.output)
