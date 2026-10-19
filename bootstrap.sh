#!/usr/bin/env bash
set -euo pipefail
cd /data/cyf/codes/TauTrajDiff/WWW
mkdir -p logs models results/www locks
if [ ! -x .venv/bin/python ]; then
  /data/miniconda/envs/ABS/bin/python -m venv --system-site-packages .venv
fi
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=180
python -m pip install 'transformers>=4.57,<5' 'accelerate>=1.0' 'huggingface_hub<1' pillow scipy scikit-learn nvitop 'num2words==0.5.14'
python -m pip freeze > logs/environment.txt
python -c 'import torch,transformers; from transformers import Qwen3VLForConditionalGeneration, AutoModelForImageTextToText; print(torch.__version__,transformers.__version__)'
python download_models.py
python wait_for_gpu.py --gpu-policy shared --min-free-gib 30 --reserve-gib 4 --poll-seconds 10 --idle-samples 2
