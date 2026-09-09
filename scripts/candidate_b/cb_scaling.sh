#!/bin/bash
# cb_scaling.sh — scaling-only re-run (fixed). Robust GPU release + guaranteed cleanup.
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/cb/project}"
cd "$PROJECT_ROOT" || exit 1
PY="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
MODELS="${MODELS_DIR:-/root/autodl-tmp/models}"
DISK="${DISK_CHECK_PATH:-/root/autodl-tmp}"
export PATH=/usr/local/bin:/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
LOG=/root/scaling.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

free_gpu(){
  # kill by GPU compute-apps PID (catches EngineCore worker, not just api_server)
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9
  ps aux | grep "[v]llm" | awk '{print $2}' | xargs -r kill -9
  for i in $(seq 1 30); do
    F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$F" -gt 75000 ] && { log "GPU freed (${F} MiB)"; return 0; }
    sleep 3
  done
  log "WARN GPU not fully free ($F MiB)"; return 0
}

serve(){  # $1 path  $2 name  $3 util
  free_gpu
  nohup $PY -m vllm.entrypoints.openai.api_server --model "$1" \
    --served-model-name "$2" --port 8000 --gpu-memory-utilization "$3" \
    --max-model-len 4096 --enforce-eager > "/root/vllm_$2.log" 2>&1 &
  for i in $(seq 1 75); do
    curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | grep -q "$2" && { log "$2 READY"; return 0; }
    sleep 8
  done
  log "$2 FAILED to start (see /root/vllm_$2.log)"; return 1
}

dl(){ $PY -c "from modelscope import snapshot_download; print(snapshot_download('$1', cache_dir='$MODELS'))" 2>/dev/null | tail -1; }

log "SCALING START (disk: $(df -h "$DISK"|tail -1|awk '{print $4}') free)"
for SPEC in "Qwen/Qwen2.5-7B-Instruct qwen7b 0.45" "Qwen/Qwen2.5-14B-Instruct qwen14b 0.70"; do
  set -- $SPEC; MID=$1; NAME=$2; GU=$3
  log "$NAME: download"
  P=$(dl "$MID")
  if [ -n "$P" ] && [ -d "$P" ]; then
    if serve "$P" "$NAME" "$GU"; then
      log "$NAME: harness (seed 42 coverage)"
      CB_SEED=42 $PY scripts/candidate_b/run_harness_zd.py --n 200 --workers 24 --tag "$NAME" \
        --model "api://localhost:8000/$NAME" > "/root/scale_${NAME}.log" 2>&1 \
        && log "$NAME harness DONE" || log "$NAME harness FAIL"
    fi
  else
    log "$NAME download FAIL (path='$P')"
  fi
  rm -rf "$P" 2>/dev/null   # ALWAYS free disk, success or fail
  log "$NAME cleaned (disk: $(df -h "$DISK"|tail -1|awk '{print $4}') free)"
done
free_gpu
log "SCALING DONE"
touch /root/SCALING_DONE
