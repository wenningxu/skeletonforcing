#!/usr/bin/env bash
set -euo pipefail
cd /workspace/motion-valley
: "${EXPERIMENT_DEADLINE_UTC:?Set allocation deadline at most 90 minutes}"
export PYTHONUNBUFFERED=1 HF_HOME=/workspace/data/hf-cache
trap 'python3 -m pip freeze > /workspace/outputs/OF008-pip_freeze.txt' EXIT
python3 -m pip install --break-system-packages 'omegaconf==2.3.1' 'pydantic==2.13.5' 'einops==0.8.2' 'hydra-core==1.3.7' 'safetensors==0.8.0' 'transformers==4.57.6' 'huggingface_hub==0.36.2' sentencepiece ftfy scipy pytest tqdm pyyaml
python3 -m pytest tests -q
python3 -m motion_valley.kimodo_suite
