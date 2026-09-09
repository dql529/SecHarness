#!/usr/bin/env bash
# remote_queue_after_14b.sh — run ON the box in a screen: wait for the 14B batch (TAG in CURRENT_BATCH_TAG)
# to finish, then run the E4 seed batch, then remove the 14B weights (copies verified on the Mac) to make room
# for the 70B shards. Emits markers the Mac-side waiter greps for.
set -uo pipefail
LOGD=/root/autodl-tmp/logs
log(){ echo "[$(date -Iseconds)] $*"; }
TAG14=$(cat $LOGD/CURRENT_BATCH_TAG 2>/dev/null) || { log "no CURRENT_BATCH_TAG"; exit 2; }
[ -n "$TAG14" ] || { log "empty CURRENT_BATCH_TAG"; exit 2; }
L=$LOGD/batch_${TAG14}_driver.log; DL=$((SECONDS+28800))
# wait until the GPU is free: the driver script logs BATCH DONE on success and its launcher appends DRIVER_EXIT=<rc> on any exit
until grep -qE "BATCH DONE|DRIVER_EXIT" "$L" 2>/dev/null; do [ $SECONDS -ge $DL ] && { log "TIMEOUT waiting for 14B batch ($L)"; exit 3; }; sleep 60; done
log "14B batch finished: $(grep -E 'BATCH DONE|DRIVER_EXIT' "$L" | tail -2 | tr '\n' ' ')"
E4TAG=$(date +%Y%m%dT%H%M); echo "$E4TAG" > $LOGD/CURRENT_E4_TAG
bash /root/autodl-tmp/SecHarness/project/scripts/seed_replication_2026-09/remote_e4_batch.sh "$E4TAG" > $LOGD/e4_${E4TAG}_driver.log 2>&1; RC=$?
log "E4 batch exit=$RC tag=$E4TAG"
# free the disk for the 70B shards only when BOTH batches succeeded (a failed 14B batch still needs these weights to be re-run)
if [ $RC -eq 0 ] && grep -q "BATCH DONE" "$L"; then rm -rf /root/autodl-tmp/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct-AWQ && log "14B weights removed (14B batch BATCH DONE + E4 exit 0)"; else log "14B weights KEPT (e4_exit=$RC, 14B BATCH DONE: $(grep -c 'BATCH DONE' "$L"))"; fi
df -h /root/autodl-tmp | tail -1; log "QUEUE_DONE e4_exit=$RC"
