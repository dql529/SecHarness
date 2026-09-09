#!/usr/bin/env bash
# remote_seed_batch.sh — run ON the GPU box (AutoDL RTX PRO 6000, 2026-09-09).
# Starts the vLLM server with the SAME flags the original 14B AWQ runs used
# (run_tau_orchestrator.py MODEL_SPECS["13B"]), waits for readiness, runs a pilot,
# then the seed batch. Every output name carries seed + TAG; the runner refuses to
# overwrite. A sha256 manifest and a console log are written per TAG.
#
# Usage: bash remote_seed_batch.sh pilot            # 5-sample smoke test (writes _pilot5_ outputs)
#        bash remote_seed_batch.sh batch <TAG> ["cond1 cond2"] ["seed1 seed2"]   # default: "full noTools" × "42 123 456 789 2026"
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
export HF_HOME=/root/autodl-tmp/huggingface HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
ROOT=/root/autodl-tmp/SecHarness
MODEL="Qwen/Qwen2.5-14B-Instruct-AWQ"
PORT=8000
LOGD=/root/autodl-tmp/logs
mkdir -p "$LOGD"
MODE="${1:-pilot}"; TAG="${2:-$(date +%Y%m%dT%H%M)}"
log(){ echo "[$(date -Iseconds)] $*"; }

# ---- 1. vLLM server (idempotent) ----
if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  log "starting vLLM server ($MODEL) in screen vllm_tau"
  screen -S vllm_tau -X quit 2>/dev/null || true; fuser -k ${PORT}/tcp 2>/dev/null || true; sleep 1
  screen -dmS vllm_tau bash -c "export PATH=/root/miniconda3/bin:\$PATH HF_HOME=$HF_HOME HF_ENDPOINT=$HF_ENDPOINT HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1; \
    python -m vllm.entrypoints.openai.api_server --model $MODEL --port $PORT --gpu-memory-utilization 0.85 --quantization awq_marlin --dtype auto 2>&1 | tee $LOGD/vllm_tau_${TAG}.log"
  D=$((SECONDS+600))
  until curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    [ $SECONDS -ge $D ] && { log "vLLM not ready after 600s; tail:"; tail -20 $LOGD/vllm_tau_${TAG}.log; exit 4; }
    sleep 5
  done
fi
log "vLLM ready: $(curl -s http://127.0.0.1:${PORT}/v1/models | head -c 200)"
log "vLLM version: $(curl -s http://127.0.0.1:${PORT}/version)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# ---- 2. run ----
cd "$ROOT"
RUN="python project/scripts/v2/run_experiment.py"
case "$MODE" in
  pilot)
    log "PILOT: tau_13B_noTools × 5 samples, seed 42, tag pilot_${TAG}"
    $RUN --config project/configs/v2_tdsc/tau/tau_13B_noTools_unsw.yaml --pilot 5 --seed 42 --run-tag "pilot_${TAG}" 2>&1 | tee "$LOGD/console_pilot_${TAG}.log"
    ;;
  batch)
    CONDS="${3:-full noTools}"; SEEDS="${4:-42 123 456 789 2026}"
    log "BATCH tag=$TAG: 14B [$CONDS] × seeds [$SEEDS] (200 samples each)"
    for cond in $CONDS; do for s in $SEEDS; do
      log "--- $cond seed=$s ---"
      $RUN --config "project/configs/v2_tdsc/tau/tau_13B_${cond}_unsw.yaml" --subsample 200 --seed "$s" --run-tag "$TAG" 2>&1 | tee -a "$LOGD/console_batch_${TAG}.log" || { log "FAILED $cond seed=$s (exit ${PIPESTATUS[0]})"; exit 7; }
      ( cd "$ROOT" && sha256sum project/logs/v2_tdsc/*_seed${s}_${TAG}_audit.jsonl project/results/tables/v2_tdsc/*_seed${s}_${TAG}_* 2>/dev/null ) >> "$LOGD/SHA256SUMS_${TAG}"
    done; done
    sort -u "$LOGD/SHA256SUMS_${TAG}" -o "$LOGD/SHA256SUMS_${TAG}"
    log "BATCH DONE tag=$TAG; manifest lines: $(wc -l < $LOGD/SHA256SUMS_${TAG})"
    ;;
  *) echo "usage: $0 pilot | batch <TAG>"; exit 2;;
esac
