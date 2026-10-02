#!/usr/bin/env bash
set -euo pipefail
cd /workspace/motion-valley
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/data/hf-cache
export EXPERIMENT_DEADLINE_UTC=2026-09-23T11:46:10Z
mkdir -p /workspace/outputs/OF001-v2
date -u +%FT%TZ > /workspace/outputs/OF001-v2/setup_started.txt
python3 -m pip install --break-system-packages 'transformers>=4.49,<5' sentencepiece ftfy scipy pytest tqdm huggingface_hub pyyaml
python3 -m pytest tests -q
python3 scripts/download_data.py --root /workspace/data --datasets HumanML3D --revision 97fe404e1b52293b6c8db47619fa7ab9b5459405
python3 -m pip freeze > /workspace/outputs/OF001-v2/pip_freeze.txt
python3 -m motion_valley.overfit --config configs/overfit_v2.json --approval-file approvals/of001-v2.json --data /workspace/data --output /workspace/outputs/OF001-v2
date -u +%FT%TZ > /workspace/outputs/OF001-v2/COMPLETED
