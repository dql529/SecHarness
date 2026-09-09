#!/usr/bin/env bash
# mac_31b_followon.sh — Mac side. Wait for the running 31B seed-42 run (PID $1) to exit, then run the remaining
# seeds with the identical command (Ollama gemma4:31b, UNSW sub200), then write a sha manifest.
# Usage: bash mac_31b_followon.sh <PID_of_seed42_run> <TAG> ["seeds"]
set -uo pipefail
cd "$HOME/Paper_project/SecHarness"
PID="${1:?pid}"; TAG="${2:?tag}"; SEEDS="${3:-123 456}"; L=project/logs/seed_replication_2026-09
log(){ echo "[$(date -Iseconds)] $*"; }
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
R42=project/results/tables/v2/E1_rf_context_31B_unsw_seed42_${TAG}_results.csv
[ -f "$R42" ] || { log "seed-42 run (pid $PID) exited but $R42 is missing; aborting"; exit 4; }
log "seed-42 run (pid $PID) exited; results: $R42"
for s in $SEEDS; do
  log "--- 31B seed=$s ---"
  python3 project/scripts/v2/run_experiment.py --config project/configs/v2/E1_rf_context_31B_unsw.yaml --subsample 200 --seed "$s" --run-tag "$TAG" > "$L/mac_31B_seed${s}_${TAG}.log" 2>&1; RC=$?
  log "seed=$s exit=$RC; $(grep -E 'Acc=|Total runtime' $L/mac_31B_seed${s}_${TAG}.log | tail -2 | tr '\n' ' ')"
  [ $RC -ne 0 ] && { log "FAILED seed=$s"; break; }
done
shasum -a 256 project/logs/v2/E1_rf_context_31B_unsw_sub200_seed*_${TAG}_audit.jsonl project/results/tables/v2/E1_rf_context_31B_unsw_seed*_${TAG}_* project/results/tables/v2/E1_rf_context_31B_unsw_sub200_seed*_${TAG}_meta.json 2>/dev/null > "$L/SHA256SUMS_mac31B_${TAG}"
[ -s "$L/SHA256SUMS_mac31B_${TAG}" ] || { log "EMPTY MANIFEST"; exit 5; }
log "MAC31B FOLLOWON DONE tag=$TAG manifest lines: $(wc -l < $L/SHA256SUMS_mac31B_${TAG})"
