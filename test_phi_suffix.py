"""The native Phi expands image positions before decoder scoring."""
import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
from run_web_eval import Reader


class PhiSuffix(unittest.TestCase):
    def test_answer_is_scored_at_expanded_tail(self):
        reader=Reader.__new__(Reader);reader.key='phi4vision15b';reader.torch=torch
        reader.processor=SimpleNamespace(tokenizer=lambda *a, **kw:{'input_ids':[3,4]})
        ids=torch.tensor([[11,12,3,4,5]])
        reader.inputs=lambda messages, generation:SimpleNamespace(input_ids=ids[:,:2] if generation else ids)
        # Vision expansion inserts six positions before the assistant suffix.
        full=torch.zeros(1,11,16)
        full[0,7,3]=4;full[0,8,4]=5;full[0,9,5]=6
        class NativeProbe:
            def prepare_inputs_labels_for_multimodal(self, ids, pos, mask, cache, labels, images):
                expanded=torch.cat([torch.full((1,8),-100), labels[:,-3:]],dim=1)
                return None,None,None,None,torch.zeros(1,11,4),expanded
            def __call__(self, **kw):
                assert kw['logits_to_keep']==4 and kw['use_cache'] is False
                return SimpleNamespace(logits=full[:,-4:])
        reader.model=NativeProbe()
        # BatchFeature supports attribute lookup and mapping expansion.
        from transformers import BatchFeature
        reader.inputs=lambda messages,generation:BatchFeature({'input_ids':ids[:,:2] if generation else ids,
                                                              'attention_mask':torch.ones_like(ids[:,:2] if generation else ids)})
        with patch.dict(sys.modules, {'engines':SimpleNamespace(blocks_to_messages=lambda *a,**kw:None)}):
            result=reader.score([], 'question', 'fact')
        independent=torch.log_softmax(full[0,7:10],-1)[torch.arange(3),torch.tensor([3,4,5])]
        self.assertAlmostEqual(result['A_j'],float(independent.mean()))
        self.assertAlmostEqual(result['A_reference'],float(independent[:2].mean()))
        self.assertEqual(result['response_token_ids'],[3,4,5])


if __name__=='__main__':unittest.main()
