#!/usr/bin/env bash
set -euo pipefail
cd /workspace/motion-valley
: "${EXPERIMENT_DEADLINE_UTC:?Set from pod allocation, at most 60 minutes}"
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/data/hf-cache
python3 -m pip install --break-system-packages 'transformers==4.57.6' sentencepiece ftfy scipy pytest tqdm 'huggingface_hub==0.36.2' pyyaml
python3 -m pytest tests -q
# Both arms require separate approval records bound to the shared report.
python3 -c "from motion_valley.overfit import approval_check; approval_check('approvals/of003-no_text.json','configs/control_no_text_v4.json'); approval_check('approvals/of003-same_text.json','configs/control_same_text_v4.json')"
python3 -m motion_valley.control_experiment --config configs/control_no_text_v4.json --approval-file approvals/of003-no_text.json --data /workspace/data --output /workspace/outputs/OF003-no_text
python3 -m motion_valley.control_experiment --config configs/control_same_text_v4.json --approval-file approvals/of003-same_text.json --data /workspace/data --output /workspace/outputs/OF003-same_text
python3 -m pip freeze > /workspace/outputs/OF003-pip_freeze.txt
