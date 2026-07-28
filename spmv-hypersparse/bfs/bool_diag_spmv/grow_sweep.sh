#!/usr/bin/env bash
# One-off continuation of sweep_bfs.sh's real-hardware ("appliance") sweep:
# grid is pinned at 750x750 (already near WSE-3's physical fabric-dims cap
# of 762x1172 -- see run_bfs.appliance.py main()'s fabric_dims branch, so
# growing the grid further isn't the axis being tested here) while RMAT
# scale grows one step at a time (edgefactor fixed at 16, matching the
# s12..s19 rows already in results/hw/bfs_timing.csv) until either a
# compile or a run step actually fails, or disk quota can't fit the next
# matrix -- that failure IS the answer to "does it fit", so the loop stops
# there rather than continuing past it.
#
# Usage: ./bfs/bool_diag_spmv/grow_sweep.sh [start_scale]
#   start_scale defaults to 20 (s12-s19 @ 750x750 already have csv rows).
#
# Logs every step to grow_sweep.log next to this script (relative to repo
# root, like bfs_timing.csv itself) and writes a one-line STOP marker there
# explaining exactly why it stopped.

set -uo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

LOG="bfs/bool_diag_spmv/grow_sweep.log"
GRID=750
EDGEFACTOR=16
SOURCE=0
CHANNELS=16
ARCH=wse3
RUN_SCRIPT="bfs/bool_diag_spmv/run_bfs.appliance.py"

export no_proxy="10.125.8.2,.cerebras.internal,localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

scale="${1:-20}"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# Quota-aware: stop before generating a matrix we can't actually fit. RMAT
# file size roughly doubles per scale step (confirmed s19->s20->s21 in
# data/); require 3x the previous balanced matrix's size free as margin
# (raw + balanced + working room) before attempting the next scale.
free_kb() {
  df --output=avail -k . | tail -1
}

log "=== grow_sweep starting at scale=${scale}, grid=${GRID}x${GRID} ==="

prev_size_kb=0

while true; do
  raw="data/rmat_s${scale}_e${EDGEFACTOR}.mtx"
  matrix="data/rmat_s${scale}_e${EDGEFACTOR}.balanced${GRID}x${GRID}.mtx"

  if [ "$prev_size_kb" -gt 0 ]; then
    need_kb=$(( prev_size_kb * 6 ))  # ~2x growth * (raw+balanced) * margin
    avail_kb=$(free_kb)
    if [ "$avail_kb" -lt "$need_kb" ]; then
      log "STOP: disk quota -- only ${avail_kb}KB free, want ${need_kb}KB margin before s${scale}. RMAT growth doesn't fit anymore."
      exit 0
    fi
  fi

  log "--- s${scale} e${EDGEFACTOR} @ ${GRID}x${GRID} (channels=${CHANNELS}) ---"

  if [ ! -f "$matrix" ]; then
    if [ ! -f "$raw" ]; then
      log "generating $raw"
      if ! timeout 3600 cs_python benchmarks/gen_rmat.py "$scale" "$EDGEFACTOR" 0 "$raw" 2>&1 | tee -a "$LOG"; then
        log "STOP: gen_rmat.py failed at s${scale} (see above)."
        exit 1
      fi
    fi
    log "balancing $raw -> $matrix"
    if ! timeout 3600 ./util/analyze --matrix "$raw" --rand 0 --fabx "$GRID" --faby "$GRID" --omatrix "$matrix" 2>&1 | tee -a "$LOG"; then
      log "STOP: util/analyze failed at s${scale} -- matrix doesn't fit/balance on ${GRID}x${GRID}."
      exit 1
    fi
  fi

  prev_size_kb=$(du -k "$matrix" | cut -f1)

  matrix_base="$(basename "$matrix" .mtx)"
  OUT_ARGS=(
    "--csv=bfs/bool_diag_spmv/results/hw/bfs_timing.csv"
    "--out-timing=bfs/bool_diag_spmv/plots/hw/timing/timing_${matrix_base}_${GRID}x${GRID}_src${SOURCE}_ch${CHANNELS}.png"
  )

  log "compiling s${scale} (writes artifact_path.json)"
  if ! timeout 3600 python "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$GRID" --num_pe_rows="$GRID" \
      --channels="$CHANNELS" --source="$SOURCE" --arch="$ARCH" --notree --compile-only 2>&1 | tee -a "$LOG"; then
    log "STOP: compile FAILED at s${scale} @ ${GRID}x${GRID} -- this is the 'doesn't fit' point (device-side buffer/memory limit most likely)."
    exit 1
  fi

  log "running s${scale} (reads artifact_path.json)"
  if ! timeout 3600 python "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$GRID" --num_pe_rows="$GRID" \
      --channels="$CHANNELS" --source="$SOURCE" --arch="$ARCH" --notree "${OUT_ARGS[@]}" 2>&1 | tee -a "$LOG"; then
    log "STOP: run FAILED at s${scale} @ ${GRID}x${GRID} (compiled fine, runtime/on-device failure)."
    exit 1
  fi

  log "=== s${scale} OK ==="
  scale=$((scale + 1))
done
