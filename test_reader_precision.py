"""Check image precision without changing discrete tokenizer inputs."""
from types import SimpleNamespace
import json
from pathlib import Path
import unittest

import torch
from transformers import BatchFeature
from run_web_eval import Reader, ROOT
from vendor_inputs import tokenizer_options


class CPUProbe(BatchFeature):
    def to(self, *args, **kwargs):
        # Exercise the native conversion while substituting a CPU test device.
        if args and isinstance(args[0], str):
            args = ('cpu',) + args[1:]
        if 'device' in kwargs:
            kwargs['device'] = 'cpu'
        return super().to(*args, **kwargs)


class ProcessorProbe:
    def apply_chat_template(self, *args, **kwargs):
        return 'Encoded prompt'

    def __call__(self, **kwargs):
        return CPUProbe({'pixel_values': torch.ones(1, 3, 2, 2),
                         'input_ids': torch.tensor([[11, 12]], dtype=torch.long),
                         'attention_mask': torch.ones(1, 2, dtype=torch.long)})


class ReaderPrecision(unittest.TestCase):
    def inputs(self, key):
        reader = Reader.__new__(Reader)
        reader.key = key
        reader.device = 'cuda:0'
        reader.torch = torch
        reader.model = SimpleNamespace(dtype=torch.bfloat16)
        reader.processor = ProcessorProbe()
        return reader.inputs([{'role': 'user', 'content': 'Question'}], True)

    def test_pixtral_pixels_match_bfloat16_vision_weights(self):
        inputs = self.inputs('pixtral12b')
        vision_layer = torch.nn.Conv2d(3, 2, 1).to(dtype=torch.bfloat16)
        self.assertEqual(inputs['pixel_values'].dtype, vision_layer.weight.dtype)
        self.assertTrue(torch.isfinite(vision_layer(inputs['pixel_values'])).all())

    def test_pixtral_discrete_inputs_are_preserved(self):
        inputs = self.inputs('pixtral12b')
        self.assertEqual(inputs['input_ids'].dtype, torch.long)
        self.assertEqual(inputs['input_ids'].tolist(), [[11, 12]])
        self.assertEqual(inputs['attention_mask'].dtype, torch.long)

    def test_phi_precision_remains_bfloat16(self):
        self.assertEqual(self.inputs('phi4vision15b')['pixel_values'].dtype,
                         torch.bfloat16)

    @unittest.skipUnless((ROOT/'models/pixtral12b/tokenizer.json').exists(),
                         'Pinned checkpoint is available on the GPU server')
    def test_pixtral_split_matches_native_tekken_pattern(self):
        from transformers import AutoProcessor, AutoTokenizer
        native = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
                  r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?"
                  r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
                  r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
        options = tokenizer_options('pixtral12b')
        checkpoint = ROOT/'models/pixtral12b'
        processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, **options)
        oracle = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, **options)
        for tokenizer in (processor.tokenizer, oracle):
            state = json.loads(tokenizer.backend_tokenizer.pre_tokenizer.__getstate__())
            self.assertEqual(state['pretokenizers'][0]['pattern']['Regex'], native)
        self.assertEqual(processor.tokenizer("'The'", add_special_tokens=False)['input_ids'],
                         oracle("'The'", add_special_tokens=False)['input_ids'])


if __name__ == '__main__':
    unittest.main()
