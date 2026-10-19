"""Independent AutoTokenizer reference IDs for the frozen source statements."""
import argparse
import json
from pathlib import Path
from frozen_cohort import FrozenCohort
from vendor_inputs import reference_text, tokenizer_options
from model_registry import load_registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    cohort = FrozenCohort(args.manifest)
    info = load_registry()[args.model]
    tokenizer = AutoTokenizer.from_pretrained(Path('models')/args.model, local_files_only=True,
                                              trust_remote_code=True, **tokenizer_options(args.model))
    expected = {page: {'statement': spec['statement'], 'question': spec['question'], 'split': spec['split'],
                       'slot_order': [x-1 for x in spec['original_tags_in_display_order']],
                       'reference_token_ids': tokenizer(reference_text(args.model, spec['statement']),
                                                        add_special_tokens=False)['input_ids']}
                for page,spec in cohort.specs.items()}
    Path(args.output).write_text(json.dumps({'revision': info['revision'], 'expected': expected,
                    'cohort_sha256': cohort.sha256, 'tokenizer_options': tokenizer_options(args.model)}, indent=2)+'\n')
    print(f'Independent oracle: {len(expected)} pages')


if __name__ == '__main__': main()
