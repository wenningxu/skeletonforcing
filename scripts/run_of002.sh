#!/usr/bin/env bash
set -euo pipefail
cd /workspace/motion-valley
: "${EXPERIMENT_DEADLINE_UTC:?Supervisor must set deadline from pod allocation time}"
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/data/hf-cache
python3 -m pip install --break-system-packages 'transformers==4.57.6' sentencepiece ftfy scipy pytest tqdm 'huggingface_hub==0.36.2' pyyaml
python3 -m pytest tests -q
# Reuse the verified persistent assets. Missing files fail rather than redownload
# large datasets while a paid GPU idles; this run does not request BABEL.
test -f /workspace/data/download_manifest.json
python3 -m motion_valley.multi_overfit --config configs/overfit_multi_v3.json --approval-file approvals/of002-v3.json --data /workspace/data --output /workspace/outputs/OF002-v3
python3 -m pip freeze > /workspace/outputs/OF002-v3/pip_freeze.txt
