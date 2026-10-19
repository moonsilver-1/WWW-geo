"""Start CPU gates and the sole shared-memory coordinator on the experiment host."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        if (proc/'cwd').resolve() != ROOT: continue
        args = (proc/'cmdline').read_bytes().split(b'\0')
        assert not any(x in args for x in (b'parallel_coordinator.py', b'validate_frozen_processor.py')), f'Existing project process {proc.name}'
    except (FileNotFoundError, PermissionError):
        continue
registry = json.loads((ROOT/'model_registry.json').read_text())
env = dict(os.environ, PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'])
started = {}
for key in ('qwen3vl8b', 'smolvlm2_22b'):
    command = [sys.executable, '-u', 'validate_frozen_processor.py', '--model', key,
               '--manifest', 'data/www_multisite_v1/manifest.json', '--checkpoint', f'models/{key}',
               '--revision', registry[key]['revision'], '--output', f'results/www/{key}_multisite_processor_check.json']
    with (ROOT/'logs'/f'{key}_multisite_processor.log').open('a') as log:
        p = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    started[key+'_cpu'] = p.pid
with (ROOT/'logs/gpu_queue.log').open('a') as log:
    p = subprocess.Popen([sys.executable, '-u', 'parallel_coordinator.py', '--gpu-policy', 'shared',
                          '--poll-seconds', '10', '--idle-samples', '2', '--reserve-gib', '4'],
                         env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(ROOT/'logs/queue.pid').write_text(str(p.pid)+'\n')
started['coordinator'] = p.pid
print(json.dumps(started), flush=True)
