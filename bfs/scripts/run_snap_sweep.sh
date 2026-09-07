#!/usr/bin/env bash
# Runs each of the balanced SNAP graphs (see datasets/prep_snap.sh) through
# real-appliance BFS (run_bfs.appliance.py, no --simulator) on a 750x750
# grid. Does NOT stop at the first failure -- these are independent
# real-world graphs, so a failure on one shouldn't skip the others. Each
# graph's outcome (OK / FAILED + why) is logged independently.
#
# Usage: ./bfs/scripts/run_snap_sweep.sh
#
# Order below is smallest-predicted-d2h-transfer first: d2h size for this
# kernel's parent_local_buf readback is ~ grid_width * n * 4 bytes,
# independent of nnz, so vertex count n predicts whether the ~2.15GB gRPC
# message-size ceiling gets hit -- berkstan (n=685231, ~2.06GB) fits just
# under it; pokec/topcats/orkut/livejournal (n 1.6M-4.8M) are predicted over.

set -uo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

LOG="bfs/scripts/run_snap_sweep.log"
GRID=750
SOURCE=0
CHANNELS=16
ARCH=wse3
RUN_SCRIPT="bfs/scripts/run_bfs.appliance.py"

export no_proxy="10.125.8.2,.cerebras.internal,localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# Optional name args, e.g. `run_snap_sweep.sh skitter patents` to run just
# those two without re-running the rest.
NAMES=("$@")
if [ ${#NAMES[@]} -eq 0 ]; then
  NAMES=(berkstan pokec topcats orkut livejournal skitter patents)
fi

log "=== run_snap_sweep starting: ${NAMES[*]} @ ${GRID}x${GRID} ==="

for name in "${NAMES[@]}"; do
  matrix="data/snap_${name}.balanced${GRID}x${GRID}.mtx"
  if [ ! -f "$matrix" ]; then
    log "SKIP $name: $matrix not found (prep_snap.sh didn't produce it)"
    continue
  fi

  log "--- $name ($matrix) @ ${GRID}x${GRID} (channels=${CHANNELS}) ---"

  matrix_base="$(basename "$matrix" .mtx)"
  OUT_ARGS=(
    "--csv=bfs/results/hw/bfs_timing.csv"
    "--out-timing=bfs/results/hw/timing/timing_${matrix_base}_${GRID}x${GRID}_src${SOURCE}_ch${CHANNELS}.png"
  )

  log "compiling $name (writes artifact_path.json)"
  if ! timeout 3600 python "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$GRID" --num_pe_rows="$GRID" \
      --channels="$CHANNELS" --source="$SOURCE" --arch="$ARCH" --notree --compile-only 2>&1 | tee -a "$LOG"; then
    log "FAILED: $name compile step failed -- skipping to next graph."
    continue
  fi

  log "running $name (reads artifact_path.json)"
  if ! timeout 3600 python "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$GRID" --num_pe_rows="$GRID" \
      --channels="$CHANNELS" --source="$SOURCE" --arch="$ARCH" --notree "${OUT_ARGS[@]}" 2>&1 | tee -a "$LOG"; then
    log "FAILED: $name run step failed -- see traceback above."
    continue
  fi

  log "=== $name OK ==="
done

log "=== run_snap_sweep done ==="
