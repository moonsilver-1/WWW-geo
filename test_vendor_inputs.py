"""Keep instruction content and fact boundaries through vendor templates."""
import unittest

try:
    import vendor_inputs
except ModuleNotFoundError:
    vendor_inputs = None


class VendorInputTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{'role': 'system', 'content': 'Exact scoring instructions'},
                         {'role': 'user', 'content': [{'type': 'text', 'text': '[EV-1]'},
                                                     {'type': 'image', 'image': 'page.png'}]},
                         {'role': 'assistant', 'content': 'Fixed fact.'}]

    def test_pixtral_keeps_system_instruction_in_scored_and_generation_inputs(self):
        self.assertIsNotNone(vendor_inputs)
        full = vendor_inputs.prepare_messages('pixtral12b', self.messages)
        prefix = vendor_inputs.prepare_messages('pixtral12b', self.messages[:2])
        self.assertEqual(full[:-1], prefix)
        self.assertEqual(full[0]['content'][0]['text'], 'Exact scoring instructions\n\n')
        self.assertEqual(self.messages[0]['role'], 'system')

    def test_smolvlm_uses_typed_content_for_all_roles(self):
        self.assertIsNotNone(vendor_inputs)
        out = vendor_inputs.prepare_messages('smolvlm2_22b', self.messages)
        self.assertTrue(all(isinstance(m['content'], list) for m in out))
        self.assertEqual(vendor_inputs.reference_text('smolvlm2_22b', 'Fixed fact.'), ' Fixed fact.')

    def test_phi_preserves_system_images_and_nonthinking_fact_prefix(self):
        self.assertIsNotNone(vendor_inputs)
        prefix = vendor_inputs.phi_prompt(self.messages[:2], True)
        full = vendor_inputs.phi_prompt(self.messages, False)
        self.assertEqual(full, prefix + 'Fixed fact.<|im_end|>')
        self.assertIn('Exact scoring instructions', full)
        self.assertEqual(full.count('<image>'), 1)
        self.assertTrue(prefix.endswith('<|im_start|>assistant<|im_sep|><nothink>'))

    def test_existing_model_inputs_keep_their_format(self):
        self.assertIsNotNone(vendor_inputs)
        self.assertEqual(vendor_inputs.prepare_messages('qwen3vl8b', self.messages), self.messages)
        self.assertEqual(vendor_inputs.reference_text('qwen3vl8b', 'Fixed fact.'), 'Fixed fact.')


if __name__ == '__main__':
    unittest.main()
