"""Outcome-independent permutation of page evidence; tags follow displayed slots."""
import hashlib


def slot_order(page, seed):
    if seed is None:
        return list(range(6))
    return sorted(range(6), key=lambda i: hashlib.sha256(
        f'{seed}:{page}:{i}'.encode('utf-8')).digest())


def permute_blocks(blocks, page, seed):
    assert len(blocks) == 6
    return [dict(blocks[old], tag=new+1) for new, old in enumerate(slot_order(page, seed))]


def check_suffix(radius, seed):
    return ('' if radius == 12 else f'_r{radius}') + ('' if seed is None else f'_perm{seed}')
