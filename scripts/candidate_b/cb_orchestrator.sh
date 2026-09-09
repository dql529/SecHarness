#!/bin/bash
# cb_orchestrator.sh — candidate-B matrix, unattended on A800.
# Runs from a script FILE so the controlling process cmdline is "bash cb_orchestrator.sh"
# (contains no "vllm.entrypoints"), making `pkill -f`/bracket-grep vllm self-safe.
#
# Order (high-certainty first):
#   1. multi-seed (123,7) sklearn coverage chain + harness           [3B server]
#   2. scaling: Qwen2.5-7B, 14B  -> harness (seed-42 coverage)       [swap server]
#   3. multi-seed (123,7) selective                                  [3B server, last]
# seed 42 is already complete in candidate_b/ — not repeated.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/cb/project}"
cd "$PROJECT_ROOT" || exit 1
PY="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
MODELS="${MODELS_DIR:-/root/autodl-tmp/models}"
THREEB="${THREEB_MODEL:-$MODELS/LLM-Research/Llama-3.2-3B-Instruct}"
export PATH=/usr/local/bin:/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
ROOTLOG=/root/orch.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ROOTLOG"; }

serve_vllm(){  # $1=model_path  $2=served_name  $3=gpu_util
  ps aux | grep "[v]llm.entrypoints" | awk '{print $2}' | xargs -r kill -9
  sleep 4
  nohup $PY -m vllm.entrypoints.openai.api_server --model "$1" \
    --served-model-name "$2" --port 8000 --gpu-memory-utilization "${3:-0.85}" \
    --max-model-len 4096 --enforce-eager > "/root/vllm_$2.log" 2>&1 &
  for i in $(seq 1 75); do
    curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | grep -q "$2" && { log "vllm $2 READY"; return 0; }
    sleep 8
  done
  log "vllm $2 FAILED to start"; return 1
}

dl(){ $PY -c "from modelscope import snapshot_download; print(snapshot_download('$1', cache_dir='$MODELS'))" 2>/dev/null | tail -1; }

log "ORCH START"
# Wait for any in-flight seed-42 selective run to finish first (avoid vLLM
# contention). Polls process liveness, not the DONE marker, so a crashed run
# does not hang here forever.
while ps aux | grep "[r]un_selective" >/dev/null 2>&1; do
  log "waiting for in-flight selective to finish..."
  sleep 30
done
log "no in-flight selective; proceeding"

# Ensure 3B is serving for the multi-seed phase
curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | grep -q llama3.2-3b || serve_vllm "$THREEB" llama3.2-3b 0.85

# ---- Phase 1: multi-seed sklearn + harness (3B) ----
for S in 123 7; do
  log "SEED $S : sklearn coverage chain"
  CB_SEED=$S $PY scripts/candidate_b/run_coverage_zd_stageA.py    > /root/s${S}_stageA.log 2>&1 || log "s$S stageA FAIL"
  CB_SEED=$S $PY scripts/candidate_b/analyze_coverage_variants.py > /root/s${S}_variants.log 2>&1 || log "s$S variants FAIL"
  CB_SEED=$S $PY scripts/candidate_b/run_multidetector_validation.py > /root/s${S}_multidet.log 2>&1 || log "s$S multidet FAIL"
  log "SEED $S : harness (3B)"
  CB_SEED=$S $PY scripts/candidate_b/run_harness_zd.py --n 200 --workers 32 --tag 3b \
    --model api://localhost:8000/llama3.2-3b > /root/s${S}_three.log 2>&1 || log "s$S three FAIL"
done
log "PHASE1 multi-seed harness DONE"

# ---- Phase 2: scaling harness (seed-42 coverage already in candidate_b/) ----
for SPEC in "Qwen/Qwen2.5-7B-Instruct qwen7b 0.85" "Qwen/Qwen2.5-14B-Instruct qwen14b 0.90"; do
  set -- $SPEC; MID=$1; NAME=$2; GU=$3
  log "SCALING $NAME : download"
  P=$(dl "$MID")
  if [ -z "$P" ] || [ ! -d "$P" ]; then log "$NAME download FAIL ($P)"; continue; fi
  serve_vllm "$P" "$NAME" "$GU" || continue
  log "SCALING $NAME : harness"
  CB_SEED=42 $PY scripts/candidate_b/run_harness_zd.py --n 200 --workers 24 --tag "$NAME" \
    --model "api://localhost:8000/$NAME" > "/root/scale_${NAME}.log" 2>&1 || log "$NAME three FAIL"
  rm -rf "$P"; log "$NAME weights removed (disk freed)"
done
log "PHASE2 scaling DONE"

# Restore 3B server for selective phase
serve_vllm "$THREEB" llama3.2-3b 0.85

# ---- Phase 3: multi-seed selective (last, expensive) ----
for S in 123 7; do
  log "SEED $S : selective (3B)"
  CB_SEED=$S $PY scripts/candidate_b/run_selective.py --n 100 --workers 32 --tag 3b \
    --model api://localhost:8000/llama3.2-3b > /root/s${S}_sel.log 2>&1 || log "s$S selective FAIL"
done
log "PHASE3 selective DONE"

log "ORCH DONE"
touch /root/ORCH_DONE
