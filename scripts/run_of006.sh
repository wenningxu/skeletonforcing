#!/usr/bin/env bash
set -euo pipefail
cd /workspace/motion-valley
: "${EXPERIMENT_DEADLINE_UTC:?Set allocation deadline, at most 45 minutes}"
export PYTHONUNBUFFERED=1 HF_HOME=/workspace/data/hf-cache
trap 'python3 -m pip freeze > /workspace/outputs/OF006-pip_freeze.txt' EXIT
python3 -m pip install --break-system-packages 'transformers==4.57.6' sentencepiece ftfy scipy pytest tqdm 'huggingface_hub==0.36.2' pyyaml
python3 -m pytest tests -q
python3 -m motion_valley.control_diagnosis
