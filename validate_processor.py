"""CPU-only validation of multimodal prefixes and reference-token boundaries."""
from pathlib import Path
import argparse
import json
import sys
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'src'/'blackbox_eval'), str(ROOT/'src'/'evidence_state')]
from run_web_eval import Reader, neutralize
from engines import blocks_to_messages
from realweb_data import SPEC as development
from realweb_extension import SPEC as transfer, build_blocks
from run_realweb import build_blocks_real
from vendor_inputs import reference_text, tokenizer_options
from slot_permutation import permute_blocks, slot_order, check_suffix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--repository', required=True)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--neutral-radius',type=int,choices=(12,24),default=12)
    ap.add_argument('--permutation-seed',type=int)
    args = ap.parse_args()
    target = ROOT/'models'/args.model
    if not (target/'config.json').exists() or not (target/'tokenizer_config.json').exists():
        snapshot_download(args.repository, revision=args.revision, local_dir=target,
                          allow_patterns=['*.json', '*.txt', '*.jinja', '*.model', '*.py'], max_workers=2)
    processor = AutoProcessor.from_pretrained(target, local_files_only=True, trust_remote_code=True,
                                            **tokenizer_options(args.model))
    reader = Reader.__new__(Reader); reader.processor = processor; reader.device = 'cpu'; reader.key = args.model
    torch.set_num_threads(4)
    records = []
    for page, spec in {**development, **transfer}.items():
        statement = spec['statement']; question = spec.get('question', spec.get('query'))
        for state in [[1]*6, [-1]*6]:
            original_state = state if args.permutation_seed is None else [1]*6
            blocks = build_blocks_real(page, original_state) if page in development else build_blocks(page)
            if args.permutation_seed is not None:
                blocks = permute_blocks(blocks,page,args.permutation_seed)
            blocks = neutralize(blocks, state,args.neutral_radius)
            enc = reader.inputs(blocks_to_messages(blocks, question, statement), False)
            prefix = reader.inputs(blocks_to_messages(blocks, question), True)
            n = prefix.input_ids.shape[1]
            assert torch.equal(enc.input_ids[0, :n], prefix.input_ids[0]), (page, state, 'prefix')
            ids = processor.tokenizer(reference_text(args.model, statement), add_special_tokens=False)['input_ids']
            assert enc.input_ids[0, n:n+len(ids)].tolist() == ids, (page, state, 'reference boundary')
            records.append({'page': page, 'state': state, 'reference_tokens': len(ids),
                            'response_tokens': enc.input_ids.shape[1]-n})
        print(f'{args.model} CPU spans: {page} ({len(records)}/36)', flush=True)
    suffix=check_suffix(args.neutral_radius,args.permutation_seed)
    path = ROOT/'results'/'www'/f'{args.model}{suffix}_processor_check.json'
    path.write_text(json.dumps({'model': args.model, 'revision': args.revision,
                               'tokenizer_options': tokenizer_options(args.model),
                               'neutral_radius':args.neutral_radius,
                               'permutation_seed':args.permutation_seed,
                               'slot_orders':{p:slot_order(p,args.permutation_seed) for p in {**development,**transfer}},
                               'checks': records, 'passed': True}, indent=2)+'\n')
    print(f'{args.model}: 36 CPU prefix/reference checks passed', flush=True)


if __name__ == '__main__':
    main()
