"""Validate vendor inputs on CPU, then prepare weights for the single GPU queue."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_registry import load_registry

ROOT = Path(__file__).resolve().parent
KEYS = ['smolvlm2_22b', 'pixtral12b', 'phi4vision15b']


def main():
    os.chdir(ROOT)
    lock = (ROOT/'locks/vendor_prepare.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Vendor preparation already running', flush=True)
        return
    state = {}; registry = load_registry(); passed = []
    for key in KEYS:
        checkpoint = registry[key]
        check = ROOT/'results/www'/f'{key}_processor_check.json'
        if check.exists() and json.loads(check.read_text()).get('passed'):
            assert json.loads(check.read_text())['revision'] == checkpoint['revision']
            code = 0
        else:
            command = [sys.executable, '-u', 'validate_processor.py', '--model', key,
                       '--repository', checkpoint['repository'], '--revision', checkpoint['revision']]
            with (ROOT/'logs'/f'{key}_processor_check.log').open('a') as log:
                try:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240).returncode
                except subprocess.TimeoutExpired:
                    code = 124
        state[key] = {'cpu_check_exit': code, 'status': 'processor_passed' if code == 0 else 'processor_failed',
                      'revision': checkpoint['revision'], 'updated_at': time.time()}
        (ROOT/'results/www/vendor_prepare_state.json').write_text(json.dumps(state, indent=2))
        print(key, state[key]['status'], flush=True)
        if code == 0:
            passed.append(key)
    if passed:
        subprocess.run([sys.executable, 'download_models.py', '--models', *passed], check=True)


if __name__ == '__main__':
    main()
