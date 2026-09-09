#!/usr/bin/env bash
# a800_setup.sh — Idempotent A800 environment setup for Exp-D-cic (vLLM backend)
#
# PREREQUISITE: rsync_to_a800.sh must be run from Mac BEFORE this script.
#   This script is run ON THE A800 (AutoDL) after rsync.
#
# Usage: bash project/scripts/v2_tdsc/a800_setup.sh
# Exit codes:
#   1 — rsync prerequisite not met (files missing)
#   2 — reserved
#   3 — insufficient disk space (< 10GB free)
#   4 — vLLM server failed to start within timeout
#   5 — LoRA adapter not listed by vLLM /v1/models
#   6 — sanity inference failed

set -euo pipefail

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"

export PATH=/root/miniconda3/bin:$PATH
export HF_HOME="${AUTODL_TMP}/huggingface"
# huggingface.co not reachable from AutoDL (China region); use mirror
export HF_ENDPOINT=https://hf-mirror.com

PAPER5_DIR="${AUTODL_TMP}/SecHarness"
ADAPTER_DIR="${PAPER5_DIR}/project/models/qlora_unsw_v2_clean/adapter"
BASE_MODEL="unsloth/Llama-3.2-3B-Instruct"
BASE_MODEL_LOCAL="${HF_HOME}/hub/models--unsloth--Llama-3.2-3B-Instruct"
VLLM_LOG="${AUTODL_TMP}/vllm_alpha.log"
VLLM_PORT=8000
VLLM_URL="http://127.0.0.1:${VLLM_PORT}/v1"

log() {
    echo "[$(date -Iseconds)] $*"
}

# ---------------------------------------------------------------------------
# Step 0: Prerequisite check — rsync must have been run first
# ---------------------------------------------------------------------------
log "step 0: checking rsync prerequisites"

if ! test -d "${PAPER5_DIR}/project/src"; then
    echo "ERROR: rsync_to_a800.sh not run yet — ${PAPER5_DIR}/project/src not found" >&2
    exit 1
fi

if ! test -f "${ADAPTER_DIR}/adapter_config.json"; then
    echo "ERROR: LoRA adapter config missing: ${ADAPTER_DIR}/adapter_config.json" >&2
    echo "       Run rsync_to_a800.sh from Mac first, then retry." >&2
    exit 1
fi

log "step 0: prerequisites OK"

# ---------------------------------------------------------------------------
# Step 1: Disk housekeeping
# ---------------------------------------------------------------------------
log "step 1: disk housekeeping"

df -h "${AUTODL_TMP}"

log "step 1: HuggingFace hub usage (sorted by size)"
if [ -d "${HF_HOME}/hub" ]; then
    du -sh "${HF_HOME}/hub/models--"* 2>/dev/null | sort -rh || true
fi

DOLPHIN_DIR="${HF_HOME}/hub/models--cognitivecomputations--dolphin-2.9-llama3-8b"
if [ -d "${DOLPHIN_DIR}" ]; then
    log "step 1: removing dolphin-2.9-llama3-8b (~16GB mandatory cleanup)"
    rm -rf "${DOLPHIN_DIR}"
    log "step 1: dolphin model removed"
else
    log "step 1: dolphin model not found, skipping removal"
fi

df -h "${AUTODL_TMP}"

# Assert at least 10GB free
FREE_KB=$(df "${AUTODL_TMP}" | awk 'NR==2 {print $4}')
# Use awk for decimal division (bc not available on AutoDL minimal Ubuntu)
FREE_GB=$(awk "BEGIN { printf \"%.1f\", ${FREE_KB}/1048576 }")
log "step 1: free space = ${FREE_GB} GB"
if [ "${FREE_KB}" -lt 10485760 ]; then
    echo "ERROR: insufficient disk space — need ≥ 10GB free, have ${FREE_GB} GB" >&2
    exit 3
fi

log "step 1: disk check passed (${FREE_GB} GB free)"

# ---------------------------------------------------------------------------
# Step 2: Base model download (skip if already present)
# ---------------------------------------------------------------------------
log "step 2: checking base model ${BASE_MODEL}"

if [ -f "${BASE_MODEL_LOCAL}/config.json" ] && \
   ls "${BASE_MODEL_LOCAL}/"*.safetensors 1>/dev/null 2>&1; then
    log "step 2: base model already downloaded, skipping"
else
    log "step 2: downloading ${BASE_MODEL} (non-gated, matches adapter base_model_name_or_path)"
    /root/miniconda3/bin/huggingface-cli download "${BASE_MODEL}" \
        --local-dir "${BASE_MODEL_LOCAL}" \
        --local-dir-use-symlinks False
    log "step 2: base model download complete"
fi

# ---------------------------------------------------------------------------
# Step 3: Log LoRA adapter placement (already verified in step 0)
# ---------------------------------------------------------------------------
log "step 3: verifying LoRA adapter placement"
ls -la "${ADAPTER_DIR}/"
log "step 3: adapter files confirmed"

# ---------------------------------------------------------------------------
# Step 4: vLLM server start (in screen session)
# ---------------------------------------------------------------------------
log "step 4: starting vLLM server in screen session 'vllm_alpha'"

# Kill existing session if present (idempotent)
if screen -list | grep -q "vllm_alpha"; then
    log "step 4: found existing vllm_alpha screen session, killing it"
    screen -S vllm_alpha -X quit || true
    sleep 2
fi

# Ensure port 8000 is free before launching — screen -X quit alone may leave
# a lingering vLLM process that holds the port across session boundaries.
fuser -k 8000/tcp 2>/dev/null || true
sleep 1

screen -dmS vllm_alpha bash -c "
  export PATH=/root/miniconda3/bin:\$PATH
  export HF_HOME=${AUTODL_TMP}/huggingface
  export HF_ENDPOINT=https://hf-mirror.com
  /root/miniconda3/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model ${BASE_MODEL} \
    --enable-lora \
    --lora-modules alpha_unsw_v2_clean=${ADAPTER_DIR} \
    --max-lora-rank 16 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.75 \
    --port ${VLLM_PORT} \
    --served-model-name unsloth-llama-3.2-3b-instruct \
    2>&1 | tee ${VLLM_LOG}
"

log "step 4: vLLM screen session started"

# ---------------------------------------------------------------------------
# Step 5: Readiness probe — poll /v1/models (timeout 300s)
# ---------------------------------------------------------------------------
log "step 5: waiting for vLLM endpoint (timeout 300s)"

DEADLINE=$((SECONDS + 300))
READY=0
while [ "${SECONDS}" -lt "${DEADLINE}" ]; do
    if curl -sf "${VLLM_URL}/models" -o /tmp/vllm_models.json 2>/dev/null; then
        READY=1
        break
    fi
    sleep 5
    log "step 5: waiting... (${SECONDS}s elapsed)"
done

if [ "${READY}" -eq 0 ]; then
    log "step 5: TIMEOUT — vLLM failed to start within 300s"
    screen -S vllm_alpha -X hardcopy /tmp/vllm_dump.txt 2>/dev/null || true
    cat /tmp/vllm_dump.txt 2>/dev/null || cat "${VLLM_LOG}" 2>/dev/null | tail -50 || true
    exit 4
fi

log "step 5: endpoint reachable, checking LoRA adapter listing"

# Verify the LoRA adapter alias is listed
if ! grep -q '"id":"alpha_unsw_v2_clean"' /tmp/vllm_models.json 2>/dev/null; then
    echo "ERROR: LoRA adapter 'alpha_unsw_v2_clean' not listed by vLLM /v1/models" >&2
    echo "       Check ${VLLM_LOG} for adapter loading errors" >&2
    cat /tmp/vllm_models.json || true
    exit 5
fi

log "step 5: LoRA adapter 'alpha_unsw_v2_clean' confirmed in /v1/models"

# ---------------------------------------------------------------------------
# Step 6: Sanity inference — one hardcoded request
# ---------------------------------------------------------------------------
log "step 6: running sanity inference"

SANITY_PAYLOAD='{
  "model": "alpha_unsw_v2_clean",
  "messages": [
    {"role": "system", "content": "You are a network intrusion detection expert."},
    {"role": "user", "content": "proto=tcp state=FIN dur=0.1 sbytes=1000 dbytes=0 label=benign"}
  ],
  "temperature": 0.1,
  "max_tokens": 150
}'

SANITY_RESP=$(curl -sf \
    --max-time 30 \
    -X POST "${VLLM_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "${SANITY_PAYLOAD}") || {
    echo "ERROR: sanity inference request failed (curl error)" >&2
    exit 6
}

if ! echo "${SANITY_RESP}" | grep -q '"verdict"'; then
    echo "ERROR: sanity inference response missing 'verdict' field" >&2
    echo "Response: ${SANITY_RESP}" >&2
    exit 6
fi

log "step 6: sanity inference passed — response contains 'verdict'"
log "step 6: vLLM server is ready for Exp-D-cic"
log "Setup complete. Run: python3 project/scripts/v2_tdsc/run_E5_a800.py --config project/configs/v2_tdsc/E5_unsw_a800.yaml --pilot-only"
