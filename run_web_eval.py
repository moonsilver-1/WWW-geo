"""Pinned-model, fact-matched Web evidence inference on archived page states.

The displayed --device records the physical GPU selected by the queue; a UUID
visibility mask maps it to runtime cuda:0. Both channels target one exact fact.
Raw responses, parse failures and both scoring spans remain in the archive.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import time
from model_registry import load_registry
from vendor_inputs import prepare_messages, reference_text, phi_prompt, tokenizer_options
from parallel_tasks import candidate_groups, output_stem, support_keys
from slot_permutation import permute_blocks, slot_order

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE/'src').exists() else HERE.parents[1]
sys.path[:0] = [str(ROOT/'src'/'blackbox_eval'), str(ROOT/'src'/'evidence_state')]


def parse_credit(text):
    match = re.search(r'\{[^{}]*\}', text)
    try:
        data = json.loads(match.group()) if match else {}
        expected = {f'EV-{i}' for i in range(1, 7)}
        if set(data) != expected or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                       or not 0 <= x <= 1 for x in data.values()):
            return None, False
        return [float(data[f'EV-{i}']) for i in range(1, 7)], True
    except (ValueError, AttributeError):
        return None, False


def neutralize(blocks, state, radius=12):
    from p0_data import neutral_text
    from PIL import Image, ImageFilter
    out = []
    for b, present in zip(blocks, state):
        item = dict(b)
        if present < 0:
            if b['type'] in ('I', 'C'):
                image = Path(b['image']); target = image.with_name(image.stem+'_neutral.png')
                if radius != 12:
                    cache = ROOT/'results/www/intervention_cache'/f'r{radius}'
                    cache.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256(image.read_bytes()).hexdigest()
                    target = cache/f'{digest}.png'
                if not target.exists():
                    tmp = target.with_name(target.stem+f'.{os.getpid()}.tmp.png')
                    with Image.open(image) as im:
                        im.convert('RGB').filter(ImageFilter.GaussianBlur(radius)).save(tmp)
                    os.replace(tmp, target)
                item['image'] = str(target)
            else:
                base = b['text'][:400]
                item['text'] = neutral_text(base if len(base) > 60 else base+' '*(60-len(base)))
        out.append(item)
    return out


class Reader:
    def __init__(self, key, device):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from transformers import Qwen3VLForConditionalGeneration
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.key = key
        self.metadata = json.loads((ROOT/'models'/key/'download_complete.json').read_text())
        model = str(ROOT/self.metadata['path'])
        self.processor = AutoProcessor.from_pretrained(model, local_files_only=True, trust_remote_code=True,
                                                       **tokenizer_options(key))
        # Match the established image budget for the Qwen processor.
        if key == 'qwen3vl8b':
            self.processor.image_processor.min_pixels = 256*28*28
            self.processor.image_processor.max_pixels = 768*28*28
        loader = load_registry()[key]['loader']
        cls = {'qwen3vl': Qwen3VLForConditionalGeneration,
               'auto_image': AutoModelForImageTextToText, 'auto_causal': AutoModelForCausalLM}[loader]
        limits = json.loads(os.environ.get('WWW_MAX_MEMORY_MIB', '{}'))
        options = {'device_map': 'balanced',
                   'max_memory': {int(i): int(mib)*2**20 for i, mib in limits.items()}} if limits else {}
        if limits and key == 'phi4vision15b':
            from phi_sharding import checkpoint_sizes, device_map
            mapping, sizes = device_map(checkpoint_sizes(model), limits)
            options['device_map'] = mapping
            print('Exact Phi BF16 placement MiB: '+json.dumps(sizes), flush=True)
        self.model = cls.from_pretrained(model, torch_dtype=torch.bfloat16,
                                        attn_implementation='sdpa', local_files_only=True,
                                        trust_remote_code=True, **options)
        if limits:
            mapping = getattr(self.model, 'hf_device_map', {})
            assert mapping and all(str(v) not in ('cpu', 'disk') for v in mapping.values()), mapping
            print('GPU layer map: '+json.dumps(mapping), flush=True)
        else:
            self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        if key == 'phi4vision15b':
            from phi_sharding import validate_loaded_weights
            self.weight_check = validate_loaded_weights(self.model, model)
            print('Phi loaded-weight audit: '+json.dumps(self.weight_check), flush=True)

    def inputs(self, messages, add_generation_prompt):
        from PIL import Image
        key = getattr(self, 'key', '')
        if key == 'phi4vision15b':
            text = phi_prompt(messages, add_generation_prompt)
        else:
            text = self.processor.apply_chat_template(prepare_messages(key, messages), tokenize=False,
                                                      add_generation_prompt=add_generation_prompt,
                                                      enable_thinking=False)
        images = []
        for message in messages:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if item.get('type') == 'image':
                        with Image.open(item['image']) as im:
                            images.append(im.convert('RGB'))
        encoded = self.processor(text=[text], images=images or None, return_tensors='pt').to(self.device)
        if key in ('phi4vision15b', 'pixtral12b') and self.device != 'cpu':
            encoded = {name: value.to(dtype=self.torch.bfloat16) if self.torch.is_tensor(value)
                       and value.is_floating_point() else value for name, value in encoded.items()}
            from transformers import BatchFeature
            encoded = BatchFeature(encoded)
        return encoded

    def generate(self, blocks, question, system=None, suffix=None):
        from engines import blocks_to_messages
        enc = self.inputs(blocks_to_messages(blocks, question, system=system, suffix=suffix), True)
        with self.torch.inference_mode():
            output = self.model.generate(**enc, max_new_tokens=200, do_sample=False)
        return self.processor.tokenizer.decode(output[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def score(self, blocks, question, statement):
        from engines import blocks_to_messages
        enc = self.inputs(blocks_to_messages(blocks, question, statement), False)
        prefix = self.inputs(blocks_to_messages(blocks, question), True)
        n_prefix = prefix.input_ids.shape[1]
        assert self.torch.equal(enc.input_ids[0, :n_prefix], prefix.input_ids[0]), 'Unaligned assistant prefix'
        answer_ids = enc.input_ids[0, n_prefix:]
        ref_ids = self.processor.tokenizer(reference_text(self.key, statement), add_special_tokens=False)['input_ids']
        n_ref = len(ref_ids)
        assert answer_ids[:n_ref].tolist() == ref_ids, 'Reference span tokenization differs at boundary'
        with self.torch.inference_mode():
            if self.key == 'phi4vision15b':
                # Phi replaces each image token with a variable-length feature block.
                # The text-only assistant suffix retains its order at the expanded tail.
                from transformers import BatchFeature
                labels = self.torch.full_like(enc.input_ids, -100)
                labels[0, n_prefix:] = answer_ids
                images = BatchFeature({name: enc[name] for name in
                    ('pixel_values', 'pixel_attention_mask', 'spatial_shapes') if name in enc})
                prepared = self.model.prepare_inputs_labels_for_multimodal(
                    enc.input_ids.clone(), None, enc.attention_mask, None, labels, images)
                ids, positions, mask, cache, embeds, expanded_labels = prepared
                target = expanded_labels[0]
                assert self.torch.equal(target[-len(answer_ids):], answer_ids), 'Phi expanded suffix differs'
                assert int((target != -100).sum()) == len(answer_ids), 'Phi answer positions differ'
                output = self.model(input_ids=ids, position_ids=positions, attention_mask=mask,
                    past_key_values=cache, inputs_embeds=embeds,
                    logits_to_keep=len(answer_ids)+1, use_cache=False)
                assert output.logits.shape[1] == len(answer_ids)+1, 'Phi response suffix truncated'
                logits = output.logits[0, :-1].float()
            else:
                logits = self.model(**enc).logits[0, n_prefix-1:enc.input_ids.shape[1]-1].float()
            lp = self.torch.log_softmax(logits, -1).gather(1, answer_ids.to(logits.device)[:, None]).squeeze(1)
        return {'A_j': float(lp.mean()), 'A_reference': float(lp[:n_ref].mean()),
                'n_response_tokens': len(answer_ids), 'n_reference_tokens': n_ref,
                'response_token_ids': answer_ids.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(load_registry()))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--pages', default='all')
    ap.add_argument('--limit-states', type=int, default=0)
    ap.add_argument('--task-id')
    ap.add_argument('--neutral-radius', type=int, choices=(12,24), default=12)
    ap.add_argument('--permutation-seed', type=int)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--cohort-manifest')
    args = ap.parse_args()
    from realweb_data import SPEC as development
    from realweb_extension import SPEC as transfer, build_blocks
    from run_realweb import build_blocks_real
    from run_absolute_control import NOCTX_SYSTEM, NOCTX_SUFFIX
    specs = {**development, **transfer}
    cohort = None
    if args.cohort_manifest:
        from frozen_cohort import FrozenCohort
        cohort = FrozenCohort(args.cohort_manifest)
        assert args.task_id and args.permutation_seed is None
        assert args.model in cohort.manifest['models']
        assert args.neutral_radius == cohort.manifest['neutral_radius']
        specs = cohort.specs
        development = {}
    selected = list(specs) if args.pages == 'all' else args.pages.split(',')
    if args.permutation_seed is not None:
        assert args.task_id, 'Permutation control requires a separate task archive'
    assert set(selected) <= set(specs)
    work = [(page, state) for page in selected for state in itertools.product((-1, 1), repeat=6)]
    if args.limit_states:
        work = work[:args.limit_states]
    data_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for folder in [ROOT/'data'/'realweb', ROOT/'data'/'realweb_extension']
                   for p in folder.rglob('*') if p.is_file()}
    if cohort:
        data_hashes = {str((cohort.base/name).relative_to(ROOT)): value
                       for name,value in cohort.file_hashes.items()}
    destination = ROOT/'results'/'www'; destination.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args.model, args.task_id)
    output = destination/f'{stem}.jsonl'
    previous = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if cohort:
        assert all(r.get('cohort_sha256') == cohort.sha256 for r in previous), 'Resume cohort mismatch'
        if not args.dry_run:
            check = json.loads((destination/f'{args.model}_multisite_processor_check.json').read_text())
            assert check.get('passed') and check.get('cohort_sha256') == cohort.sha256 and len(check['checks']) == 48
    errors = [r for r in previous if r.get('error')]
    if errors:
        archive = destination/'startup_failures'/f'{stem}_{time.time_ns()}'
        archive.mkdir(parents=True)
        (archive/'runtime_attempts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in errors))
        previous = [r for r in previous if not r.get('error')]
        output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
        old_manifest = destination/f'{stem}_manifest.json'
        if old_manifest.exists(): (archive/'manifest.json').write_bytes(old_manifest.read_bytes())
    done = support_keys(previous)
    work = [(p, s) for p, s in work if (p, s) not in done]
    if args.dry_run:
        for page in selected:
            blocks = cohort.blocks(page) if cohort else (build_blocks_real(page, [1]*6) if page in development else build_blocks(page))
            assert len(blocks) == 6
            assert all(Path(b['image']).exists() for b in blocks if b['type'] in ('I', 'C'))
        print(f'{len(selected)} pages; {len(work)} pending states; all assets found', flush=True)
        return
    policy = os.environ.get('WWW_GPU_POLICY', 'exclusive')
    allocation = None
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        from gpu_policy import snapshot, eligible
        visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        allocations = [next(g for g in snapshot() if g['uuid'] == uuid) for uuid in visible]
        allocation = allocations[0] if len(allocations) == 1 else allocations
        required = int(os.environ.get('WWW_MIN_FREE_MIB', '30720'))
        reserve = int(os.environ.get('WWW_RESERVE_FREE_MIB', '4096'))
        allowed = (eligible(allocations[0], policy, required) if len(allocations) == 1 else
                   any({g['uuid'] for g in group} == set(visible)
                       for group in candidate_groups(allocations, required, reserve, allow_shard=True)))
        if not allowed:
            print('GPU launch condition changed; returning to memory queue', flush=True)
            raise SystemExit(75)
    import torch
    torch.set_num_threads(args.workers)
    runtime_device = 'cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else args.device
    runtime_devices = list(range(len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))))
    if policy == 'shared':
        reserve_bytes = int(os.environ.get('WWW_RESERVE_FREE_MIB', '4096')) * 1024 * 1024
        for index in runtime_devices:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            fraction = max(0.0, min(1.0, (free_bytes - reserve_bytes) / total_bytes))
            if fraction <= 0: raise SystemExit(75)
            torch.cuda.set_per_process_memory_fraction(fraction, index)
            print(f'Shared allocation cuda:{index}: {free_bytes/2**30:.2f}GiB free; '
                  f'{reserve_bytes/2**30:.2f}GiB reserve; allocator cap {fraction*total_bytes/2**30:.2f}GiB', flush=True)
    try:
        reader = Reader(args.model, runtime_device)
    except torch.cuda.OutOfMemoryError:
        print('Model loading exceeded memory budget; request more headroom', flush=True)
        raise SystemExit(76)
    except ValueError as exc:
        if args.model == 'phi4vision15b' and 'GPU' in str(exc):
            print(str(exc)+'; request more headroom', flush=True)
            raise SystemExit(76)
        raise
    (destination/f'{stem}_manifest.json').write_text(json.dumps({'model': reader.metadata,
         'source_sha256': data_hashes, 'protocol': 'experiments/www/protocol.md',
         'physical_device': args.device, 'visible_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'),
         'torch': torch.__version__, 'legacy_score': 'entire assistant response including chat terminator',
         'gpu_policy': policy, 'prelaunch_driver_snapshot': allocation,
         'reserve_free_mib': int(os.environ.get('WWW_RESERVE_FREE_MIB', '4096')),
         'vendor': load_registry()[args.model]['vendor'],
         'tokenizer_options': tokenizer_options(args.model),
         'task_id': args.task_id, 'neutral_radius': args.neutral_radius,
         'cohort_sha256': cohort.sha256 if cohort else None,
         'cohort_manifest': str(cohort.path.relative_to(ROOT)) if cohort else None,
         'permutation_seed': args.permutation_seed,
         'slot_orders': {p: [x-1 for x in specs[p]['original_tags_in_display_order']] if cohort else slot_order(p,args.permutation_seed) for p in selected},
         'selected_pages': selected, 'device_map': getattr(reader.model, 'hf_device_map', {}),
         'scoring_alignment': 'expanded-sequence assistant suffix' if args.model == 'phi4vision15b' else 'processor token positions',
         'loaded_weight_check': getattr(reader, 'weight_check', None),
         'max_memory_mib': json.loads(os.environ.get('WWW_MAX_MEMORY_MIB','{}')),
         'reference_template_prefix': ' ' if args.model == 'smolvlm2_22b' else '',
         'runner_sha256': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                           for name in ['run_web_eval.py', 'vendor_inputs.py', 'model_registry.json', 'parallel_tasks.py', 'phi_sharding.py', 'slot_permutation.py'] + (['frozen_cohort.py'] if cohort else [])},
         'reference_score': 'reference-statement tokens only'}, indent=2)+'\n')
    started = time.time()
    print(f'{args.model}: {len(work)} pending states on {args.device}', flush=True)
    for index, (page, state) in enumerate(work, 1):
        spec = specs[page]; question = spec.get('question', spec.get('query')); statement = spec['statement']
        record = {'engine': args.model, 'checkpoint': reader.metadata['repository'],
                  'revision': reader.metadata['revision'], 'page': page, 'state': list(state),
                  'question': question, 'statement': statement,
                  'split': 'development' if page in development else 'transfer',
                  'task_id': args.task_id, 'neutral_radius': args.neutral_radius,
                  'permutation_seed': args.permutation_seed,
                  'slot_order': slot_order(page,args.permutation_seed)}
        try:
            original_state = list(state) if args.permutation_seed is None else [1]*6
            if cohort:
                blocks = cohort.state_blocks(page, state, ROOT/'results/www/intervention_cache/multisite')
                record['cohort_sha256'] = cohort.sha256
                record['slot_order'] = [x-1 for x in specs[page]['original_tags_in_display_order']]
            else:
                blocks = build_blocks_real(page, original_state) if page in development else build_blocks(page)
                if args.permutation_seed is not None:
                    blocks = permute_blocks(blocks,page,args.permutation_seed)
                blocks = neutralize(blocks, state, args.neutral_radius)
            record.update(reader.score(blocks, question, statement))
            raw = reader.generate(blocks, question, system=NOCTX_SYSTEM, suffix=NOCTX_SUFFIX.format(ans=statement))
            scores, valid = parse_credit(raw)
            record.update({'gen': raw, 'p': scores, 'parse_ok': valid})
            if all(s == 1 for s in state):
                record['answer_generation'] = reader.generate(blocks, question)
        except Exception as exc:
            record.update({'parse_ok': False, 'error': f'{type(exc).__name__}: {exc}'})
        record['elapsed'] = time.time()-started
        record['gpu_memory_mib'] = {str(i): {'allocated': torch.cuda.max_memory_allocated(i)/2**20,
                                          'reserved': torch.cuda.max_memory_reserved(i)/2**20} for i in runtime_devices}
        record['peak_allocated_mib'] = sum(v['allocated'] for v in record['gpu_memory_mib'].values())
        record['peak_reserved_mib'] = sum(v['reserved'] for v in record['gpu_memory_mib'].values())
        with output.open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n'); handle.flush()
        if index == 1 or index % 16 == 0:
            print(f'{index}/{len(work)} {page} parse={record["parse_ok"]} error={record.get("error", "none")} elapsed={record["elapsed"]:.0f}s', flush=True)
        if 'OutOfMemoryError' in record.get('error', ''):
            print('Inference exceeded memory budget; request more headroom', flush=True)
            raise SystemExit(76)
        if index <= 2 and 'error' in record:
            raise RuntimeError('Initial model/scoring validation failed; inspect logged error')
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    unique = {(r['page'], tuple(r['state'])): r for r in rows if r.get('parse_ok') and 'A_j' in r}
    counts = Counter(r['page'] for r in unique.values())
    report = {'pages': len(selected), 'eligible_cubes': sum(counts[p] == 64 for p in selected),
              'parsed_state_counts': dict(counts), 'raw_rows': len(rows)}
    (destination/f'{stem}_completion.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
