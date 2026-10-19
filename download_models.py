"""Download pinned public checkpoints into the dedicated project directory."""
import json
import argparse
import fcntl
import os
from pathlib import Path
import time

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '180')
from huggingface_hub import snapshot_download
from requests.exceptions import RequestException
from model_registry import load_registry
from weight_validation import IncompleteSnapshotError, weight_files

ROOT = Path(__file__).resolve().parent
REGISTRY = load_registry()
MODELS = {key: record['repository'] for key, record in REGISTRY.items()}
REVISIONS = {key: record['revision'] for key, record in REGISTRY.items()}


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def download(key, repo, target):
    revision = REVISIONS[key]
    model_lock = (ROOT/'locks'/f'download_{key}.lock').open('a+')
    fcntl.flock(model_lock, fcntl.LOCK_EX)
    if (target/'download_complete.json').exists():
        record = json.loads((target/'download_complete.json').read_text())
        assert record['revision'] == revision
        return record
    for attempt in range(1, 13):
        save(target/'download_state.json', {'revision': revision, 'attempt': attempt,
                                          'status': 'downloading', 'updated': time.time()})
        print(f'Downloading {key}; revision {revision}; attempt {attempt}/12', flush=True)
        try:
            # local_dir retains the completed files and partial download cache.
            snapshot_download(repo, revision=revision, local_dir=target, max_workers=2,
                              allow_patterns=['*.json', '*.jinja', '*.txt', '*.model', '*.safetensors', '*.py'])
            shards = weight_files(target)
            record = {'repository': repo, 'revision': revision, 'path': f'models/{key}',
                      'shard_bytes': {name: (target/name).stat().st_size for name in shards}}
            save(target/'download_complete.json', record)
            save(target/'download_state.json', {'revision': revision, 'status': 'complete',
                                              'updated': time.time()})
            return record
        except (RequestException, IncompleteSnapshotError) as error:
            response = getattr(error, 'response', None)
            status = getattr(response, 'status_code', None)
            transient = status is None or status in (408, 429) or status >= 500
            save(target/'download_state.json', {'revision': revision, 'attempt': attempt,
                                              'status': 'retrying' if transient and attempt < 12 else 'failed',
                                              'error_type': type(error).__name__, 'http_status': status,
                                              'updated': time.time()})
            if not transient or attempt == 12:
                raise
            delay = min(30 * attempt, 300)
            print(f'{key}: {type(error).__name__}; resuming in {delay}s', flush=True)
            time.sleep(delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', choices=list(MODELS), default=list(MODELS))
    args = ap.parse_args()
    (ROOT/'locks').mkdir(exist_ok=True)
    lock = (ROOT/'locks'/'model_download.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Model downloader already running', flush=True)
        return
    existing = ROOT/'models'/'manifest.json'
    manifest = json.loads(existing.read_text()) if existing.exists() else {}
    for key in args.models:
        repo = MODELS[key]
        target = ROOT/'models'/key
        target.mkdir(parents=True, exist_ok=True)
        if (target/'download_complete.json').exists():
            manifest[key] = json.loads((target/'download_complete.json').read_text())
            assert manifest[key]['revision'] == REVISIONS[key]
            continue
        manifest[key] = download(key, repo, target)
        save(ROOT/'models'/'manifest.json', manifest)
        print(f'Ready: {key}', flush=True)
    save(ROOT/'models'/'manifest.json', manifest)


if __name__ == '__main__':
    main()
