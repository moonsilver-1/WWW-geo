"""Exact BF16 checkpoint layout for Phi's delayed native vision loader."""
import json
from pathlib import Path
import struct


def checkpoint_sizes(folder):
    sizes = {}
    for shard in Path(folder).glob('*.safetensors'):
        with shard.open('rb') as handle:
            length = struct.unpack('<Q', handle.read(8))[0]
            header = json.loads(handle.read(length))
        for name, item in header.items():
            if name == '__metadata__':
                continue
            if item['dtype'] != 'BF16':
                raise ValueError('Phi placement requires the pinned BF16 checkpoint')
            module = ('.'.join(name.split('.')[:3]) if name.startswith('model.layers.')
                      else 'lm_head' if name.startswith('lm_head.')
                      else '.'.join(name.split('.')[:2]))
            sizes[module] = sizes.get(module, 0) + item['data_offsets'][1] - item['data_offsets'][0]
    if not sizes:
        raise ValueError('Missing checkpoint headers')
    return sizes


def device_map(sizes, budgets_mib):
    """Place contiguous decoder layers with vision/embedding on the input card."""
    budgets = [int(budgets_mib[str(i)] if str(i) in budgets_mib else budgets_mib[i])*2**20
               for i in range(len(budgets_mib))]
    layers = sorted((k for k in sizes if k.startswith('model.layers.')),
                    key=lambda k: int(k.rsplit('.', 1)[1]))
    first = {'model.embed_tokens', 'model.mm_projector', 'model.vision_tower'}
    last = {'model.norm', 'lm_head'}
    if set(sizes) != first | last | set(layers):
        raise ValueError('Unexpected pinned Phi parameter layout')
    used = [0]*len(budgets)
    mapping = {}
    for names, index in [(first, 0), (last, len(budgets)-1)]:
        for name in names:
            used[index] += sizes[name]
            mapping[name] = index
    index = 0
    for name in layers:
        while index < len(budgets) and used[index] + sizes[name] > budgets[index]:
            index += 1
        if index == len(budgets):
            raise ValueError('Exact Phi layers exceed the available GPU weight budgets')
        mapping[name] = index
        used[index] += sizes[name]
    if any(a > b for a, b in zip(used, budgets)):
        raise ValueError('Phi non-decoder modules exceed the GPU budget')
    mapping['model.rotary_emb'] = 0
    return mapping, [a/2**20 for a in used]


def validate_loaded_weights(model, folder):
    """Check complete BF16 storage and checkpoint samples in the delayed vision tower."""
    import torch
    from safetensors import safe_open
    parameters=dict(model.named_parameters())
    expected=checkpoint_sizes(folder)
    assert all(not p.is_meta and p.dtype==torch.bfloat16 and p.device.type=='cuda'
               for p in parameters.values()), 'Phi parameter dtype/device mismatch'
    actual=sum(p.numel()*p.element_size() for p in parameters.values())
    assert actual==sum(expected.values()), ('Phi parameter count mismatch',actual,sum(expected.values()))
    checked=0
    for shard in Path(folder).glob('*.safetensors'):
        with safe_open(shard,framework='pt',device='cpu') as handle:
            for name in handle.keys():
                if not name.startswith('model.vision_tower.vision_tower.'):
                    continue
                reference=handle.get_tensor(name).reshape(-1)
                target=parameters[name].detach().reshape(-1)
                indices=[0,len(reference)//2,len(reference)-1]
                assert torch.equal(reference[indices],target[indices].cpu()), ('Phi vision checkpoint mismatch',name)
                checked+=1
    assert checked>0, 'Missing Phi vision weight audit'
    return {'bf16_parameter_bytes':actual,'vision_tensors_sampled':checked,'samples_per_tensor':3,
            'gpu_only':True,'passed':True}
