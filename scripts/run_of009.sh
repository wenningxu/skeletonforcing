#!/usr/bin/env bash
set -euo pipefail
cd /workspace/motion-valley
: "${EXPERIMENT_DEADLINE_UTC:?}"
export PYTHONUNBUFFERED=1
trap 'python3 -m pip freeze > /workspace/outputs/OF009-pip_freeze.txt' EXIT
python3 -m pip install --break-system-packages scipy pytest 'transformers==4.57.6' sentencepiece ftfy 'einops==0.8.2' 'omegaconf==2.3.1' 'pydantic==2.13.5' 'hydra-core==1.3.7' 'safetensors==0.8.0'
python3 -m pytest tests -q --basetemp=/tmp/of009-tests -p no:cacheprovider
python3 -m motion_valley.root_first_experiment --config configs/root_first_overfit_v1.json --approval-file approvals/of009-default-valley.json --data /workspace/data --output /workspace/outputs/OF009
