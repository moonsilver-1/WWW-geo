"""Bounded GPU groups and disjoint page tasks for one project coordinator."""
from itertools import combinations
import re


def candidate_groups(gpus, required_mib, reserve_mib, occupied=(), allow_shard=False,
                     preferred_gpu=None):
    available = [g for g in gpus if g['uuid'] not in occupied]
    singles = [[g] for g in available if g['free_mib'] >= required_mib]
    singles.sort(key=lambda group: (group[0]['index'] != preferred_gpu,
                                   -group[0]['free_mib']))
    pairs = []
    if allow_shard:
        for pair in combinations(available, 2):
            if min(g['free_mib'] for g in pair) < 14*1024:
                continue
            if sum(g['free_mib']-reserve_mib for g in pair) >= required_mib-reserve_mib:
                pairs.append(sorted(pair, key=lambda g: -g['free_mib']))
        pairs.sort(key=lambda group: -sum(g['free_mib'] for g in group))
        if not pairs and not singles:
            for group in combinations(available,3):
                if min(g['free_mib'] for g in group)>=12*1024 and sum(g['free_mib']-reserve_mib for g in group)>=required_mib-reserve_mib:
                    pairs.append(sorted(group,key=lambda g:-g['free_mib']))
            pairs.sort(key=lambda group:-sum(g['free_mib'] for g in group))
    return singles + pairs


def weight_budgets(group, reserve_mib):
    budgets = {i: g['free_mib']-reserve_mib-1536-(2048 if i == 0 else 0)
               for i, g in enumerate(group)}
    if min(budgets.values()) <= 0:
        raise ValueError('Insufficient per-device workspace')
    return budgets


def make_control_tasks(pages):
    return [{'task_id': f'smol_blur24_g{i}', 'model': 'smolvlm2_22b',
             'pages': list(pages[i::8]), 'neutral_radius': 24,
             'preferred_gpu': i, 'min_free_gib': 11}
            for i in range(8) if pages[i::8]]


def output_stem(model, task_id):
    stem = task_id or model
    if not re.fullmatch(r'[a-zA-Z0-9_\-]{1,80}', stem):
        raise ValueError('Invalid task output name')
    return stem


def support_keys(rows):
    keys = set()
    for r in rows:
        if r.get('error') or 'A_j' not in r or 'A_reference' not in r:
            continue
        key = (r['page'], tuple(r['state']))
        if key in keys:
            raise ValueError(f'Duplicate support observation: {key}')
        keys.add(key)
    return keys
