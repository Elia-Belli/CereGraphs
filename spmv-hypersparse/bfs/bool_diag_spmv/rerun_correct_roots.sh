#!/usr/bin/env bash
# Re-run graphs from their correct canonical root vertices (vertex 0 gives
# a shallow, unrepresentative BFS for real-world graphs -- SNAP vertex ids
# are just raw-file record order, not chosen for connectivity -- unlike
# RMAT, which is symmetric-ish enough that vertex 0 is as meaningful a root
# as any).
#
# v3: uses the CORRECTLY re-balanced *.balanced750x750.mtx files
# (benchmarks/prep_snap_v3.sh -- only orkut is genuinely undirected per
# SNAP's own docs, balanced with --symmetric; berkstan/pokec/topcats/
# livejournal are genuinely DIRECTED, balanced with the new --shared-perm
# instead, which preserves vertex identity WITHOUT fabricating structural
# symmetry) plus each graph's own recovered *.operm permutation file.
# v2 wrongly symmetrized all 5 graphs uniformly -- see project memory for
# why that silently fabricated edges for the 4 directed ones (berkstan's
# nnz nearly doubled, which alone caused its h2d failure; not a real scale
# limit). Translated source vertices for this run (original -> v3 balanced,
# via each graph's own .operm file):
#   pokec:       315318  -> 1178437
#   berkstan:    546279  -> 353938
#   livejournal: 772860  -> 3653142
#   topcats:     1405263 -> 1267272
#
# Uses --max-rounds=2 (not the default 10): total_runtime_cycles/
# search_time_cycles are computed from round_trip_start_buffer/
# round_trip_done_buffer (always correct, independent of max_rounds -- see
# bfs_timing.py's decode_phase_row), so a small max_rounds only truncates
# the DETAILED per-round breakdown, not the correctness of the headline
# timing/GTEPS numbers -- worth doing here since these roots may need deep
# BFS traversals and per-PE ts_buf memory scales with max_rounds.
#
# topcats and livejournal are still expected to fail at COMPILE (PE
# static-memory ceiling, independent of source vertex or directedness --
# confirmed on both the v1 and v2 attempts already); included here anyway
# for completeness now that the underlying matrix is finally correct.
set -uo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

LOG="bfs/bool_diag_spmv/rerun_correct_roots.log"
GRID=750
CHANNELS=16
ARCH=wse3
MAX_ROUNDS=2
RUN_SCRIPT="bfs/bool_diag_spmv/run_bfs.appliance.py"

export no_proxy="10.125.8.2,.cerebras.internal,localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

run_one() {
  local name="$1" matrix="$2" source="$3"
  local matrix_base
  matrix_base="$(basename "$matrix" .mtx)"
  local OUT_ARGS=(
    "--csv=bfs/bool_diag_spmv/results/hw/bfs_timing.csv"
    "--out-timing=bfs/bool_diag_spmv/plots/hw/timing/timing_${matrix_base}.png"
  )
  log "--- $name ($matrix) source=$source @ ${GRID}x${GRID} max-rounds=${MAX_ROUNDS} ---"
  log "compiling $name (writes artifact_path.json)"
  if ! timeout 3600 python "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$GRID" --num_pe_rows="$GRID" \
      --channels="$CHANNELS" --source="$source" --max-rounds="$MAX_ROUNDS" --arch="$ARCH" --notree --compile-only 2>&1 | tee -a "$LOG"; then
    log "FAILED: $name compile step failed -- skipping to next graph."
    return 1
  fi
  log "running $name (reads artifact_path.json)"
  if ! timeout 3600 python "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$GRID" --num_pe_rows="$GRID" \
      --channels="$CHANNELS" --source="$source" --max-rounds="$MAX_ROUNDS" --arch="$ARCH" --notree "${OUT_ARGS[@]}" 2>&1 | tee -a "$LOG"; then
    log "FAILED: $name run step failed -- see traceback above."
    return 1
  fi
  log "=== $name (source=$source) OK ==="
}

log "=== rerun_correct_roots v3 starting (correctly directed/undirected + translated sources) ==="
run_one "pokec"       "data/snap_pokec.balanced${GRID}x${GRID}.mtx"       1178437
run_one "berkstan"    "data/snap_berkstan.balanced${GRID}x${GRID}.mtx"    353938
run_one "topcats"     "data/snap_topcats.balanced${GRID}x${GRID}.mtx"     1267272
run_one "livejournal" "data/snap_livejournal.balanced${GRID}x${GRID}.mtx" 3653142
log "=== rerun_correct_roots v3 done ==="
