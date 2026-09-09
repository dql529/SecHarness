#!/usr/bin/env bash
# pull_box_batch.sh — Mac side. Pull one finished box batch back, verify every file against the box-side sha manifest,
# mirror the box logs, and run the comparability gate against the archived logs.
# Usage: bash pull_box_batch.sh <TAG> <manifest-name> <box-log-subdir> <box-results-subdir> ["<new_log_glob>=<archived_log>" ...]
#   e.g. bash pull_box_batch.sh 20260909T1932 SHA256SUMS_20260909T1932 v2_tdsc v2_tdsc \
#          "tau_13B_noPermissions_unsw_sub200_seed*_20260909T1932_audit.jsonl=tau_13B_noPermissions_unsw_sub200_audit.jsonl"
set -uo pipefail
cd "$HOME/Paper_project/SecHarness"
TAG="${1:?TAG}"; MAN="${2:?manifest}"; LSUB="${3:?logsub}"; RSUB="${4:?ressub}"; shift 4
BOX="${BOX_HOST:?set BOX_HOST=user@host of the GPU box}"; SSH="ssh -p ${BOX_PORT:-22} -o StrictHostKeyChecking=no"; BR="${BOX_REPO:-/root/autodl-tmp/SecHarness}"
L=project/logs/seed_replication_2026-09; mkdir -p "$L/box_logs"
log(){ echo "[$(date -Iseconds)] $*"; }
log "rsync results/logs for tag $TAG"
rsync -a -e "$SSH" --include="*${TAG}*" --exclude="*" "$BOX:$BR/project/logs/$LSUB/" "project/logs/$LSUB/" || { log "rsync logs failed"; exit 2; }
rsync -a -e "$SSH" --include="*${TAG}*" --exclude="*" "$BOX:$BR/project/results/tables/$RSUB/" "project/results/tables/$RSUB/" || { log "rsync results failed"; exit 2; }
rsync -a -e "$SSH" "$BOX:/root/autodl-tmp/logs/" "$L/box_logs/" --exclude="*.safetensors" || { log "rsync box logs failed"; exit 2; }
[ -f "$L/box_logs/$MAN" ] || { log "manifest $MAN not found in box logs"; exit 3; }
log "verifying $(wc -l < $L/box_logs/$MAN) manifest entries"
# manifest paths are relative to the box repo root; shasum -c needs the same relative layout (we are at repo root)
if shasum -a 256 -c "$L/box_logs/$MAN" --quiet; then log "SHA VERIFY: all $(wc -l < $L/box_logs/$MAN | tr -d ' ') files match"; else log "SHA VERIFY: MISMATCH (see above)"; exit 4; fi
RC=0
for pair in "$@"; do
  glob="${pair%%=*}"; ref="${pair##*=}"
  for f in project/logs/$LSUB/$glob; do
    [ -f "$f" ] || { log "no file for $glob"; RC=5; continue; }
    out=$(python3 project/scripts/seed_replication_2026-09/compare_rf_tool.py "$f" "project/logs/$LSUB/$ref" 2>&1); r=$?
    log "gate $(basename $f) vs $ref: $(echo "$out" | tail -1) (exit $r)"; [ $r -ne 0 ] && RC=6
  done
done
log "PULL DONE tag=$TAG gate_rc=$RC"; exit $RC
