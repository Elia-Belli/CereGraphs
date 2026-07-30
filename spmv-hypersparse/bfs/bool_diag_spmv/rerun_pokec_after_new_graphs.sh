#!/usr/bin/env bash
# One-shot chained follow-up: waits for the currently-running `snap_sweep`
# screen session (skitter/patents compile+run) to finish, THEN regenerates
# + rebalances + reruns pokec under the direction-fix loader (per
# HANDOFF.md's "Immediate next steps" #1) -- never runs concurrently with
# the appliance job already in flight, since the appliance is a single
# shared allocation (see HANDOFF.md's environment notes).
#
# Meant to be launched inside its own detached screen session (see
# run_snap_sweep_screen.sh for that convention), e.g.:
#   ssh cer-usn-01 'screen -dmS pokec_rerun -L \
#     -Logfile /home/elia/CereGraphs/spmv-hypersparse/bfs/bool_diag_spmv/rerun_pokec.out \
#     bash /home/elia/CereGraphs/spmv-hypersparse/bfs/bool_diag_spmv/rerun_pokec_after_new_graphs.sh'
#
# Check on it: ssh cer-usn-01 'screen -r pokec_rerun'  (Ctrl-A D to detach)

set -uo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

LOG="bfs/bool_diag_spmv/rerun_pokec.log"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

log "=== rerun_pokec_after_new_graphs: waiting for snap_sweep screen session to finish ==="
while screen -list | grep -q '\.snap_sweep[[:space:]]'; do
  sleep 30
done
log "=== snap_sweep finished, proceeding with pokec regeneration ==="

source ~/cs_appliance_sdk/bin/activate

GRID=750

log "regenerating data/snap_pokec.mtx from raw (direction-corrected loader, no flag needed)"
if ! cs_python benchmarks/snap_to_mtx.py data/snap/soc-pokec-relationships.txt.gz data/snap_pokec.mtx 2>&1 | tee -a "$LOG"; then
  log "FAILED regenerating pokec .mtx -- aborting"
  exit 1
fi

log "rebalancing (--shared-perm, with permutation dump)"
if ! ./util/analyze --matrix data/snap_pokec.mtx --shared-perm --fabx "$GRID" --faby "$GRID" \
    --omatrix "data/snap_pokec.balanced${GRID}x${GRID}.mtx" \
    --operm "data/snap_pokec.balanced${GRID}x${GRID}.operm" 2>&1 | tee -a "$LOG"; then
  log "FAILED rebalancing pokec -- aborting"
  exit 1
fi

log "=== pokec regenerated + rebalanced, now compiling+running (source=0) ==="
bash bfs/bool_diag_spmv/run_snap_sweep.sh pokec 2>&1 | tee -a "$LOG"

log "=== rerun_pokec_after_new_graphs done ==="
