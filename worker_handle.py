"""Observe project workers that survive a coordinator replacement."""
import json
from pathlib import Path


class ExistingWorker:
    def __init__(self, root, key, state):
        self.root, self.key, self.state = Path(root), key, state
        self.proc = Path(f'/proc/{state["pid"]}')
        if self.proc.exists():
            argv = self.proc.joinpath('cmdline').read_bytes().split(b'\0')
            assert self.proc.joinpath('cwd').resolve() == self.root.resolve()
            assert b'run_web_eval.py' in argv and key.encode() in argv
            self.start_ticks = self.proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
        else:
            self.start_ticks = None

    def poll(self):
        try:
            stat = self.proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] != 'Z' and stat[19] == self.start_ticks:
                return None
        except FileNotFoundError:
            pass
        completion = self.root/'results/www'/f'{self.key}_completion.json'
        if completion.exists() and completion.stat().st_mtime >= self.state['started_at_unix']:
            return 0
        records = self.root/'results/www'/f'{self.key}.jsonl'
        if records.exists():
            rows = records.read_text().splitlines()
            if rows and 'OutOfMemoryError' in json.loads(rows[-1]).get('error', ''):
                return 76
        return 1
