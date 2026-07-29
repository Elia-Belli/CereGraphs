#!/usr/bin/env bash
# Re-run pokec (run-only, reusing the just-compiled artifact), then
# topcats/s20/berkstan (compile+run each) with the memory-reduced 2-buffer
# reduce_select_any design, real appliance hardware, 750x750 grid.
set -uo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

LOG="bfs/bool_diag_spmv/rerun_v2_sweep.log"
GRID=750
SOURCE=0
CHANNELS=16
ARCH=wse3
RUN_SCRIPT="bfs/bool_diag_spmv/run_bfs.appliance.py"

export no_proxy="10.125.8.2,.cerebras.internal,localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

run_case() {
  local matrix="$1"
  local matrix_base
  matrix_base="$(basename "$matrix" .mtx)"
  local OUT_ARGS=(
    "--csv=bfs/bool_diag_spmv/results/hw/bfs_timing.csv"
    "--out-timing=bfs/bool_diag_spmv/plots/hw/timing/timing_${matrix_base}_${GRID}x${GRID}_src${SOURCE}_ch${CHANNELS}_v2.png"
  )
  log "running $matrix_base (reads artifact_path.json)"
  if ! timeout 3600 python "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$GRID" --num_pe_rows="$GRID" \
      --channels="$CHANNELS" --source="$SOURCE" --arch="$ARCH" --notree "${OUT_ARGS[@]}" 2>&1 | tee -a "$LOG"; then
    log "FAILED: $matrix_base run step failed -- see traceback above."
    return 1
  fi
  log "=== $matrix_base OK ==="
}

compile_case() {
  local matrix="$1"
  local matrix_base
  matrix_base="$(basename "$matrix" .mtx)"
  log "compiling $matrix_base (writes artifact_path.json)"
  if ! timeout 3600 python "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$GRID" --num_pe_rows="$GRID" \
      --channels="$CHANNELS" --source="$SOURCE" --arch="$ARCH" --notree --compile-only 2>&1 | tee -a "$LOG"; then
    log "FAILED: $matrix_base compile step failed -- skipping."
    return 1
  fi
}

log "=== rerun_v2_sweep starting: pokec(run-only) topcats s20 berkstan @ ${GRID}x${GRID}, 2-buffer design ==="

# pokec was just compiled (compile-only) right before this script -- run it now
# before artifact_path.json gets overwritten by the next graph's compile.
run_case "data/snap_pokec.balanced${GRID}x${GRID}.mtx"

for matrix in "data/snap_topcats.balanced${GRID}x${GRID}.mtx" \
              "data/rmat_s20_e16.balanced${GRID}x${GRID}.mtx" \
              "data/snap_berkstan.balanced${GRID}x${GRID}.mtx"; do
  compile_case "$matrix" && run_case "$matrix"
done

log "=== rerun_v2_sweep done ==="
