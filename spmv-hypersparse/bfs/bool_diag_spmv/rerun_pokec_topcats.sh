#!/usr/bin/env bash
# Re-run pokec/topcats (which failed at the d2h gRPC ~2GiB ceiling under
# the old code, per run_snap_sweep.log) with the fixed code (Phase A/B of
# the on-device parent resolution plan -- see project memory
# appliance_bfs_scale_limit.md), real appliance hardware, same 750x750 grid.
set -uo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

LOG="bfs/bool_diag_spmv/rerun_pokec_topcats.log"
GRID=750
SOURCE=0
CHANNELS=16
ARCH=wse3
RUN_SCRIPT="bfs/bool_diag_spmv/run_bfs.appliance.py"

export no_proxy="10.125.8.2,.cerebras.internal,localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

for name in pokec topcats; do
  matrix="data/snap_${name}.balanced${GRID}x${GRID}.mtx"
  matrix_base="$(basename "$matrix" .mtx)"
  OUT_ARGS=(
    "--csv=bfs/bool_diag_spmv/results/hw/bfs_timing.csv"
    "--out-timing=bfs/bool_diag_spmv/plots/hw/timing/timing_${matrix_base}_${GRID}x${GRID}_src${SOURCE}_ch${CHANNELS}_postfix.png"
  )

  log "--- $name ($matrix) @ ${GRID}x${GRID} (channels=${CHANNELS}), FIXED code ---"

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

log "=== rerun_pokec_topcats done ==="
