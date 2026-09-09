#!/usr/bin/env bash
# remote_70b_batch.sh — run ON the box. E3-noRF 70B replication (Table S5 70B row; summary-only in the
# archive): Llama-3.3-70B-Instruct-AWQ served by vLLM (awq_marlin), harness on, check_anomaly withdrawn,
# UNSW sub200, several LLM seeds. Usage: bash remote_70b_batch.sh <TAG> ["seeds"]
# Server flags follow the documented Aug-2026 A800 practice (gpu-mem-util 0.85, awq_marlin, dtype auto); the
# April-2026 70B server command itself is unrecorded. max-model-len 8192 (Aug runs used 4096): a larger window
# cannot change outputs for prompts that fit in 4096; on 96GB the default 131072 window may not fit the KV cache.
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
export HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
ROOT=/root/autodl-tmp/SecHarness; LOGD=/root/autodl-tmp/logs; mkdir -p "$LOGD"
TAG="${1:?TAG}"; SEEDS="${2:-42 123 456}"
MODEL=casperhansen/llama-3.3-70b-instruct-awq; SERVED=llama-3.3-70b-instruct-awq; PORT=8000
log(){ echo "[$(date -Iseconds)] $*"; }
# weights complete? (9 shards)
MD=/root/autodl-tmp/huggingface/hub/models--casperhansen--llama-3.3-70b-instruct-awq
SNAP=$(ls -td "$MD"/snapshots/*/ 2>/dev/null | head -1 || true); N=$(ls -1 "$SNAP"*of-00009.safetensors 2>/dev/null | wc -l || true)
[ -n "$SNAP" ] && [ "$N" -eq 9 ] || { log "70B weights incomplete ($N/9 shards in ${SNAP:-<no snapshot>})"; exit 5; }
# HF cache blobs are named by their sha256: verify every large blob against its own name (Mac copy was sha_ok on download).
# An unreadable blob (dangling link, directory) counts as a mismatch; fewer than 9 verified blobs = the scan saw nothing.
BAD=0; NB=0; for b in "$MD"/blobs/*; do n=$(basename "$b"); [ ${#n} -eq 64 ] || continue; NB=$((NB+1)); h=$(sha256sum "$b" 2>/dev/null | cut -d" " -f1 || true); [ "$h" = "$n" ] || { log "BLOB SHA MISMATCH/UNREADABLE $n"; BAD=1; }; done
[ $BAD -eq 0 ] && [ $NB -ge 9 ] || { log "70B blob check failed (bad=$BAD, verified=$NB)"; exit 5; }; log "70B blobs sha-verified against cache names ($NB blobs)"
if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED"; then
  log "starting vLLM 70B server (screen vllm_70b)"; screen -S vllm_tau -X quit 2>/dev/null || true; screen -S vllm_70b -X quit 2>/dev/null || true; fuser -k ${PORT}/tcp 2>/dev/null || true; sleep 20
  D2=$((SECONDS+180)); until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do [ $SECONDS -ge $D2 ] && { log "GPU memory not released"; nvidia-smi; exit 4; }; sleep 5; done
  CMD="python -m vllm.entrypoints.openai.api_server --model $MODEL --served-model-name $SERVED --port $PORT --gpu-memory-utilization 0.85 --quantization awq_marlin --dtype auto --max-model-len 8192"
  log "server cmd: $CMD"
  screen -dmS vllm_70b bash -c "export PATH=/root/miniconda3/bin:\$PATH HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1; $CMD 2>&1 | tee $LOGD/vllm_70b_${TAG}.log"
  DL=$((SECONDS+1500))
  until curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED"; do [ $SECONDS -ge $DL ] && { log "70B server not ready after 1500s"; tail -30 $LOGD/vllm_70b_${TAG}.log; exit 4; }; sleep 10; done
fi
log "70B ready: $(curl -s http://127.0.0.1:${PORT}/version)"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
cd "$ROOT"
log "--- pilot (3 samples, seed 42) ---"
python project/scripts/v2/run_experiment.py --config project/configs/v2/E3_noRF_70B_unsw.yaml --pilot 3 --seed 42 --run-tag "pilot_${TAG}" 2>&1 | tee -a "$LOGD/console_70b_${TAG}.log" || { log "PILOT FAILED (exit ${PIPESTATUS[0]})"; exit 6; }
log "pilot audit lines: $(grep -c . project/logs/v2/E3_noRF_70B_unsw_pilot3_seed42_pilot_${TAG}_audit.jsonl 2>/dev/null || true)"
( sha256sum project/logs/v2/E3_noRF_70B_unsw_pilot3_seed42_pilot_${TAG}_audit.jsonl project/results/tables/v2/E3_noRF_70B_unsw_seed42_pilot_${TAG}_* project/results/tables/v2/E3_noRF_70B_unsw_pilot3_seed42_pilot_${TAG}_meta.json 2>/dev/null || true ) >> "$LOGD/SHA256SUMS_70b_${TAG}"
for s in $SEEDS; do
  log "--- E3_noRF_70B seed=$s ---"
  python project/scripts/v2/run_experiment.py --config project/configs/v2/E3_noRF_70B_unsw.yaml --subsample 200 --seed "$s" --run-tag "$TAG" 2>&1 | tee -a "$LOGD/console_70b_${TAG}.log" || { log "FAILED seed=$s (exit ${PIPESTATUS[0]})"; exit 7; }
  ( sha256sum project/logs/v2/E3_noRF_70B_unsw_sub200_seed${s}_${TAG}_audit.jsonl project/results/tables/v2/E3_noRF_70B_unsw_seed${s}_${TAG}_* project/results/tables/v2/E3_noRF_70B_unsw_sub200_seed${s}_${TAG}_meta.json 2>/dev/null || true ) >> "$LOGD/SHA256SUMS_70b_${TAG}"
done
sort -u "$LOGD/SHA256SUMS_70b_${TAG}" -o "$LOGD/SHA256SUMS_70b_${TAG}"; log "70B BATCH DONE tag=$TAG; manifest lines: $(wc -l < $LOGD/SHA256SUMS_70b_${TAG})"
