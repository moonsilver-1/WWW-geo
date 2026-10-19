"""Linux integration: retain a live worker and detect its real completion."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from worker_handle import ExistingWorker


@unittest.skipUnless(Path('/proc/self').exists(), 'Linux process ownership check')
class WorkerAdoptionTests(unittest.TestCase):
    def test_adoption_keeps_worker_pid_and_finds_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); (root/'results/www').mkdir(parents=True)
            (root/'run_web_eval.py').write_text(
                "import time\nfrom pathlib import Path\nPath('ready').touch()\ntime.sleep(.8)\n"
                "Path('results/www/test_completion.json').write_text('{}')\n")
            started = time.time()
            worker = subprocess.Popen([sys.executable, 'run_web_eval.py', 'test'], cwd=root)
            try:
                deadline = time.monotonic() + 5
                while not (root/'ready').exists():
                    if worker.poll() is not None or time.monotonic() > deadline:
                        self.fail('Fixture worker did not reach Python entry point')
                    time.sleep(.01)
                handle = ExistingWorker(root, 'test', {'pid': worker.pid, 'started_at_unix': started})
                self.assertIsNone(handle.poll())
                worker.wait(timeout=5)
                self.assertEqual(handle.poll(), 0)
            finally:
                if worker.poll() is None:
                    worker.terminate(); worker.wait()

    def test_adoption_rejects_another_project(self):
        with self.assertRaises(AssertionError):
            ExistingWorker('/tmp', 'test', {'pid': subprocess.os.getpid(), 'started_at_unix': 0})


if __name__ == '__main__':
    unittest.main()
