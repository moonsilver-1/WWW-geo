import json
from pathlib import Path
import tempfile
import unittest
from weight_validation import IncompleteSnapshotError, weight_files


class WeightValidation(unittest.TestCase):
    def test_metadata_only_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(IncompleteSnapshotError):
                weight_files(Path(tmp))

    def test_index_requires_every_positive_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a':'one.safetensors','b':'two.safetensors'}}))
            (root/'one.safetensors').write_bytes(b'x')
            with self.assertRaises(IncompleteSnapshotError):
                weight_files(root)
            (root/'two.safetensors').write_bytes(b'')
            with self.assertRaises(IncompleteSnapshotError):
                weight_files(root)
            (root/'two.safetensors').write_bytes(b'xx')
            self.assertEqual(weight_files(root), ['one.safetensors','two.safetensors'])

    def test_single_file_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'model.safetensors').write_bytes(b'x')
            self.assertEqual(weight_files(root), ['model.safetensors'])


if __name__ == '__main__':
    unittest.main()
