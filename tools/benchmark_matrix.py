#!/usr/bin/env python3
"""Reproducible end-to-end controller experiments, using only Python's stdlib."""
import argparse
import csv
import datetime
import hashlib
import json
import math
import pathlib
import platform
import statistics
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', default='build/cpu/hzl-control')
parser.add_argument('--output', default='benchmarks')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--quick', action='store_true')
parser.add_argument('--suite', choices=['standard', 'variants'], default='standard')
parser.add_argument('--cuda-resident', choices=['0', '1'], default=None, help='default: 1 for CUDA, 0 for CPU')
parser.add_argument('--cuda-kernel', choices=['global', 'tiled'], default='global')
parser.add_argument('--cuda-graphs', choices=['0', '1'], default='0')
parser.add_argument('--keep-fields', action='store_true', help='retain potentially large per-run field CSVs')
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
if args.cuda_resident is None:
    args.cuda_resident = '1' if args.cuda else '0'
if args.repeats < 1:
    parser.error('--repeats must be positive')
if not args.cuda and (args.cuda_kernel != 'global' or args.cuda_graphs != '0' or args.cuda_resident != '0'):
    parser.error('CUDA implementation options require --cuda')
binary = pathlib.Path(args.binary).resolve()
root = pathlib.Path(args.output).resolve()
root.mkdir(parents=True, exist_ok=True)
# Representative workloads rather than an unhelpful Cartesian product.
cases = [(33,25,5,12,0,'finite'), (65,49,10,60,0,'simplex'),
         (65,49,10,60,2,'simplex'), (65,49,10,60,0,'finite')]
if not args.quick:
    cases += [(129,97,30,12,0,'finite'), (129,97,30,60,0,'simplex'),
              (65,49,120,60,0,'simplex'), (33,25,600,12,0,'finite'),
              (256,256,5,12,0,'finite'), (257,129,5,60,0,'simplex')]
if args.suite == 'variants':
    cases = [(65,49,10,60,0,'simplex'), (129,97,30,60,0,'simplex'),
             (257,129,5,60,0,'simplex')]
metadata = {'cuda_resident':args.cuda_resident, 'cuda_kernel': args.cuda_kernel, 'cuda_graphs': args.cuda_graphs,
            'suite': args.suite, 'measured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'platform': platform.platform(), 'processor': platform.processor(),
            'python': platform.python_version(), 'binary': str(binary),
            'backend': 'cuda' if args.cuda else 'cpu', 'cases': cases,
            'note': 'First control decision excluded from steady-state latency; retained in run CSVs.'}
cache = binary.parent/'CMakeCache.txt'
if cache.exists():
    metadata['cmake_cache'] = {line.split('=',1)[0]: line.split('=',1)[1]
                               for line in cache.read_text().splitlines()
                               if '=' in line and not line.startswith(('#','//'))
                               and line.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_FLAGS',
                                                    'CMAKE_CUDA_FLAGS', 'CMAKE_CUDA_ARCHITECTURES:', 'HZL_'))}
for tool, command in [('compiler',['g++','--version']), ('cuda',['nvcc','--version']),
                      ('gpu',['nvidia-smi'])]:
    try:
        run = subprocess.run(command, text=True, capture_output=True, check=False)
        metadata[tool] = {'returncode': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr}
    except OSError as error:
        metadata[tool] = str(error)
(root/'environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
with (root/'summary.csv').open('w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['width','height','endpoint_s','budget','filter_radius','method','repeat',
                     'steady_samples','mean_ms','median_ms','p95_ms','max_ms','deadline_misses',
                     'rtf','final_mse','fixed_baseline_mse','wall_s_including_setup_output_baseline'])
    for index,(width,height,endpoint,budget,radius,method) in enumerate(cases):
        for repeat in range(args.repeats):
            destination = root/f'case-{index:02d}-repeat-{repeat:02d}'
            command = [str(binary), '--width',str(width),'--height',str(height),
                       '--endpoint',str(endpoint),'--interval',str(endpoint/10),
                       '--budget',str(budget),'--filter-radius',str(radius),
                       '--method',method,'--backend','cuda' if args.cuda else 'cpu',
                       '--output',str(destination)]
            if args.cuda:
                command += ['--cuda-kernel', args.cuda_kernel, '--cuda-graphs', args.cuda_graphs, '--cuda-resident', args.cuda_resident]
            start = time.perf_counter()
            run = subprocess.run(command, text=True, capture_output=True, check=False)
            wall = time.perf_counter()-start
            destination.mkdir(parents=True, exist_ok=True)
            (destination/'command.json').write_text(json.dumps(command, indent=2)+'\n')
            (destination/'stdout.txt').write_text(run.stdout)
            (destination/'stderr.txt').write_text(run.stderr)
            if run.returncode:
                raise SystemExit(f'Experiment failed ({run.returncode}): {destination}\n{run.stderr}')
            with (destination/'cycles.csv').open() as stream:
                cycles = list(csv.DictReader(stream))
            steady = cycles[1:]
            latency = sorted(float(c['cycle_ms']) for c in steady)
            physical = sum(float(c['next_observation_s'])-float(c['observation_s']) for c in steady)
            writer.writerow([width,height,endpoint,budget,radius,method,repeat,len(latency),
                             statistics.mean(latency),statistics.median(latency),
                             latency[math.ceil(.95*len(latency))-1],max(latency),
                             sum(int(c['deadline_miss']) for c in steady),physical*1000/sum(latency),
                             cycles[-1]['target_mse'],cycles[-1]['baseline_mse'],wall])
            file.flush()
            if not args.keep_fields:
                (destination/'fields.csv').unlink()
            print(f'Completed {destination.name}: final MSE={cycles[-1]["target_mse"]}', flush=True)
print(root/'summary.csv')
