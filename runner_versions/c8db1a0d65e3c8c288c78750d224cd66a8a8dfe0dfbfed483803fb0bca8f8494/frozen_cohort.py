"""Read-only frozen cohort adapter shared by future inference and CPU checks."""
import hashlib
import json
from pathlib import Path


class FrozenCohort:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        self.base = self.path.parent
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        assert self.sha256 == self.path.with_suffix('.sha256').read_text().strip()
        self.manifest = json.loads(self.path.read_text(encoding='utf8'))
        self.specs = {}
        self.file_hashes = {'manifest.json': self.sha256}
        for page in self.manifest['pages']:
            assert page['id'] not in self.specs, 'Duplicate frozen page'
            folder = self.base / page['id']
            assert folder.resolve().is_relative_to(self.base)
            for name, expected in page['files'].items():
                target = (folder / name).resolve()
                assert target.is_relative_to(folder.resolve())
                assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, target
                self.file_hashes[f"{page['id']}/{name}"] = expected
            spec = json.loads((folder / 'blocks.json').read_text(encoding='utf8'))
            assert spec['id'] == page['id'] and spec['split'] == page['split']
            assert spec['status'] == 'frozen_before_inference'
            assert len(spec['units']) == 6 and [u['tag'] for u in spec['units']] == list(range(1, 7))
            self.specs[page['id']] = spec
        assert len(self.specs) == self.manifest['n_pages']

    def blocks(self, page):
        # Evaluation metadata (answer/semantic labels/roles) is not supplied to readers.
        blocks = []
        for unit in self.specs[page]['units']:
            block = {'tag': unit['tag'], 'type': unit['type']}
            if unit['type'] == 'I':
                image = (self.base / page / unit['image']).resolve()
                assert image.is_relative_to(self.base / page) and image.is_file()
                block['image'] = str(image)
            else:
                assert unit['type'] == 'T'
                block['text'] = unit['text']
            blocks.append(block)
        return blocks

    def state_blocks(self, page, state, cache):
        """Match frozen neutralization specification without writing among source assets."""
        from p0_data import neutral_text
        from PIL import Image, ImageFilter
        import os
        assert len(state) == 6 and all(x in (-1, 1) for x in state)
        cache = Path(cache).resolve()
        assert not cache.is_relative_to(self.base), 'Cache must be outside immutable source assets'
        blocks = self.blocks(page)
        radius = self.manifest['neutral_radius']
        for present, block in zip(state, blocks):
            if present == 1:
                continue
            if block['type'] == 'T':
                base = block['text'][:400]
                block['text'] = neutral_text(base if len(base) > 60 else base + ' ' * (60 - len(base)))
            else:
                source = Path(block['image'])
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                target = cache / f'r{radius}' / f'{digest}.png'
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_name(f'{target.stem}.{os.getpid()}.tmp.png')
                    with Image.open(source) as im:
                        im.convert('RGB').filter(ImageFilter.GaussianBlur(radius)).save(temp)
                    os.replace(temp, target)
                block['image'] = str(target)
        return blocks
