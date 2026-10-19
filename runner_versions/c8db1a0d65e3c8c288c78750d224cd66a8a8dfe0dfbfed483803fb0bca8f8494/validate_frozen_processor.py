"""Local-only CPU token checks for a frozen cohort; never downloads or queues."""
import argparse
import hashlib
import json
from pathlib import Path
from frozen_cohort import FrozenCohort
from run_web_eval import ROOT, Reader
from vendor_inputs import reference_text, tokenizer_options


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--checkpoint', required=True, help='Existing local checkpoint path')
    ap.add_argument('--revision', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoProcessor
    from engines import blocks_to_messages
    cohort = FrozenCohort(args.manifest)
    assert args.model in cohort.manifest['models']
    checkpoint = Path(args.checkpoint).resolve()
    metadata = json.loads((checkpoint/'download_complete.json').read_text())
    assert metadata['revision'] == args.revision
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True,
                                             trust_remote_code=True, **tokenizer_options(args.model))
    if args.model == 'qwen3vl8b':
        processor.image_processor.min_pixels = 256 * 28 * 28
        processor.image_processor.max_pixels = 768 * 28 * 28
    reader = Reader.__new__(Reader)
    reader.processor, reader.device, reader.key = processor, 'cpu', args.model
    torch.set_num_threads(4)
    checks = []
    for page, spec in cohort.specs.items():
        for state in ([1] * 6, [-1] * 6):
            blocks = cohort.state_blocks(page, state, ROOT / 'results/www/intervention_cache/multisite')
            enc = reader.inputs(blocks_to_messages(blocks, spec['question'], spec['statement']), False)
            prefix = reader.inputs(blocks_to_messages(blocks, spec['question']), True)
            n = prefix.input_ids.shape[1]
            ids = processor.tokenizer(reference_text(args.model, spec['statement']), add_special_tokens=False)['input_ids']
            assert torch.equal(enc.input_ids[0, :n], prefix.input_ids[0]), (page, state, 'prefix')
            assert enc.input_ids[0, n:n+len(ids)].tolist() == ids, (page, state, 'reference')
            checks.append({'page': page, 'state': state, 'reference_tokens': len(ids),
                           'response_tokens': enc.input_ids.shape[1] - n})
    report = {'model': args.model, 'revision': args.revision, 'cohort_sha256': cohort.sha256,
              'checks': checks, 'passed': True, 'tokenizer_options': tokenizer_options(args.model),
              'runner_sha256': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                for name in ('frozen_cohort.py', 'validate_frozen_processor.py', 'run_web_eval.py', 'vendor_inputs.py')}}
    output = Path(args.output)
    assert not output.resolve().is_relative_to(cohort.base)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(f'{len(checks)} CPU prefix/reference checks passed')


if __name__ == '__main__':
    main()
