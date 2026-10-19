# WWW-geo

Geo / vision-language evaluation project transferred from the internal server
(`/data/cyf/codes/TauTrajDiff/WWW`).

## Contents

- `src/` — core library code
- `experiments/` — experiment scripts and configs
- `data/` — datasets and inputs
- `results/` — evaluation outputs
- `logs/` — run logs
- `download_models.py` — model weight downloader
- `protocol.md` — project protocol notes

## Model weights

`models/` (about 94.6 GB of safetensors checkpoints for qwen3vl8b, pixtral12b,
smolvlm2_22b, phi4vision15b, internvl35_8b) is **not** tracked in this
repository because of its size. Re-create it locally with:

```bash
python download_models.py
```

`.venv/` (Linux virtual environment) is also excluded; create your own
environment from `dependency_wheels/` or your preferred package source.

## License

MIT — see [LICENSE](LICENSE).
