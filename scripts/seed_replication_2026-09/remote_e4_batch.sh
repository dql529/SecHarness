#!/usr/bin/env bash
# remote_e4_batch.sh — run ON the box. Replicates Table 5's E4 row (lat_E4_unsw: Llama-3.2-3B-Instruct
# + clean LoRA adapter, HF transformers path, harness on, N=200) with several LLM seeds.
# Usage: bash remote_e4_batch.sh <TAG> ["seeds"]      (default seeds: 42 123 456 789 2026)
# Stops the vLLM 14B server first (HF path needs the GPU), checks the base-model symlink the
# original config expects, refuses to overwrite (runner guard), writes a sha manifest per TAG.
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
export HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOT=/root/autodl-tmp/SecHarness; LOGD=/root/autodl-tmp/logs; mkdir -p "$LOGD"
TAG="${1:?TAG}"; SEEDS="${2:-42 123 456 789 2026}"
log(){ echo "[$(date -Iseconds)] $*"; }
# base model path expected by configs/v2/lat_E4_unsw.yaml (a flat dir on the original A800)
BASE=/root/autodl-tmp/models/unsloth-Llama-3.2-3B-Instruct
if [ ! -e "$BASE/config.json" ]; then
  SNAP=$(ls -d /root/autodl-tmp/huggingface/hub/models--unsloth--Llama-3.2-3B-Instruct/snapshots/*/ | head -1)
  mkdir -p /root/autodl-tmp/models; ln -sfn "${SNAP%/}" "$BASE"; log "symlinked $BASE -> $SNAP"
fi
test -f "$ROOT/project/models/qlora_unsw_v2_clean/adapter/adapter_config.json" || { log "adapter missing"; exit 5; }
if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then log "stopping vLLM 14B server to free the GPU"; screen -S vllm_tau -X quit || true; sleep 5; fuser -k 8000/tcp 2>/dev/null || true; sleep 5; fi
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
cd "$ROOT"
for s in $SEEDS; do
  log "--- lat_E4 seed=$s ---"
  python project/scripts/v2/run_experiment.py --config project/configs/v2/lat_E4_unsw.yaml --subsample 200 --seed "$s" --run-tag "$TAG" 2>&1 | tee -a "$LOGD/console_e4_${TAG}.log" || { log "FAILED seed=$s (exit ${PIPESTATUS[0]})"; exit 7; }
  ( sha256sum project/logs/v2/lat_E4_unsw_sub200_seed${s}_${TAG}_audit.jsonl project/results/tables/v2/lat_E4_unsw_seed${s}_${TAG}_* project/results/tables/v2/lat_E4_unsw_sub200_seed${s}_${TAG}_meta.json 2>/dev/null || true ) >> "$LOGD/SHA256SUMS_e4_${TAG}"
done
sort -u "$LOGD/SHA256SUMS_e4_${TAG}" -o "$LOGD/SHA256SUMS_e4_${TAG}"
log "E4 BATCH DONE tag=$TAG; manifest lines: $(wc -l < $LOGD/SHA256SUMS_e4_${TAG})"
