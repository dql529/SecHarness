#!/usr/bin/env bash
# remote_install_offline.sh — run ON the GPU box after the wheel set was rsynced to /root/autodl-tmp/wheels.
# Installs vLLM 0.28.0 (pins torch 2.13.0) + runner deps with no network, then verifies the stack.
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
W=/root/autodl-tmp/wheels
LOG=/root/autodl-tmp/logs/pip_offline_install.log
mkdir -p /root/autodl-tmp/logs
echo "[$(date -Iseconds)] wheels: $(ls $W | wc -l) files, $(du -sh $W | cut -f1)" | tee "$LOG"
pip install --no-index --find-links "$W" "vllm==0.28.0" "scikit-learn==1.7.2" "pandas==2.3.3" "pyyaml>=6.0" "tqdm>=4.66" 2>&1 | tee -a "$LOG"
echo "PIP_OFFLINE_EXIT=${PIPESTATUS[0]}" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import torch, vllm, sklearn, pandas, yaml, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available(),
      "cap", torch.cuda.get_device_capability(), "gpu", torch.cuda.get_device_name(0))
print("vllm", vllm.__version__, "| sklearn", sklearn.__version__, "| pandas", pandas.__version__, "| numpy", numpy.__version__)
import pickle, warnings; warnings.simplefilter("error")
d = pickle.load(open("/root/autodl-tmp/SecHarness/project/logs/E2_beta_ml_model.pkl", "rb"))
print("RF pickle loads without version warning:", type(d["clf"]).__name__, d["clf"].n_estimators, "trees")
print("STACK_OK")
PY
