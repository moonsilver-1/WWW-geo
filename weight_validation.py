"""Reject incomplete local snapshots returned after a Hub network failure."""
import json
from pathlib import Path


class IncompleteSnapshotError(RuntimeError):
    pass


def weight_files(target: Path):
    index = target/'model.safetensors.index.json'
    shards = (sorted(set(json.loads(index.read_text())['weight_map'].values()))
              if index.exists() else sorted(p.name for p in target.glob('*.safetensors')))
    missing = [name for name in shards if not (target/name).is_file() or (target/name).stat().st_size == 0]
    if not shards or missing:
        raise IncompleteSnapshotError('Missing weight files: '+(', '.join(missing) if shards else 'all'))
    return shards
