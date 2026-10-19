"""Recover one vendor preparation without restarting active model workers."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from model_registry import load_registry
from download_models import download

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(load_registry()))
    args = ap.parse_args(); info = load_registry()[args.model]
    lock = (ROOT/'locks'/f'prepare_{args.model}.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Model preparation already running', flush=True)
        return
    subprocess.run([sys.executable, '-u', str(ROOT/'validate_processor.py'), '--model', args.model,
                    '--repository', info['repository'], '--revision', info['revision']], check=True)
    record = download(args.model, info['repository'], ROOT/'models'/args.model)
    print(json.dumps({'model': args.model, 'status': 'ready', 'revision': record['revision']}), flush=True)


if __name__ == '__main__':
    main()
