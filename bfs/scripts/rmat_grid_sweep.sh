#!/usr/bin/env bash
# Two-dimensional RMAT sweep: scale s10..s20 (e16) x grid size, growing grid
# from the smallest that survives compile (PE-memory ceiling) up through
# 512 (doubling), plus 750 (WSE-3's practical max). Continues past
# individual (scale,grid) failures -- never aborts the whole sweep on one
# bad combo (independent data points, not a fail-fast chain).
#
# Balancing uses `--symmetric --operm` -- a balanced<PxP>.mtx without a
# matching .operm sitting next to it is treated as stale and regenerated.
#
# RMAT source vertex is always 0, untranslated -- vertex 0 maps to itself
# under --symmetric --operm rebalancing (unlike SNAP graphs, which do need
# a translated source).
#
# Idempotent: before running a (scale,grid) combo, checks
# results/hw/bfs_timing.csv for an existing row with the same infile_mtx
# basename + pe_grid, and skips if already present.
#
# Meant to run for many hours unattended -- launch it detached in screen,
# not in a blocking foreground shell:
#
#   screen -dmS rmat_grid_sweep bash -c \
#     'source /home/elia/cs_appliance_sdk/bin/activate && \
#      /home/elia/CereGraphs/bfs/scripts/rmat_grid_sweep.sh'
#   screen -r rmat_grid_sweep      # reattach to watch
#   Ctrl-A D                       # detach again without killing it
#
# Logs every step to rmat_grid_sweep.log next to this script (relative to
# repo root, like bfs_timing.csv itself).

set -uo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

# util/analyze is a native binary, not committed -- build (or rebuild, if
# stale) it for this node's toolchain before the sweep's first use.
make -C util

LOG="bfs/scripts/rmat_grid_sweep.log"
CSV="bfs/results/hw/bfs_timing.csv"
EDGEFACTOR=16
SOURCE=0
ARCH=wse3
RUN_SCRIPT="bfs/scripts/run_bfs.appliance.py"
SCALES=(10 11 12 13 14 15 16 17 18 19 20)
GRIDS=(4 8 16 32 64 128 256 512 750)
MAX_LINKER_RETRIES=2
COMPILE_LOG="$(mktemp)"

export no_proxy="10.125.8.2,.cerebras.internal,localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

free_kb() { df --output=avail -k . | tail -1; }

already_ran() {
  local matrix_base="$1" grid="$2"
  [ -f "$CSV" ] || return 1
  # Match on basename of infile_mtx (column 2) + pe_grid (column 5).
  awk -F, -v mb="${matrix_base}.mtx" -v g="${grid}x${grid}" \
    'NR>1 { n=split($2,parts,"/"); base=parts[n]; if (base==mb && $5==g) { found=1 } }
     END { exit !found }' "$CSV"
}

ensure_raw() {
  local scale="$1" raw="$2"
  if [ -f "$raw" ]; then return 0; fi
  log "generating $raw"
  if ! timeout 3600 cs_python datasets/gen_rmat.py "$scale" "$EDGEFACTOR" 0 "$raw" 2>&1 | tee -a "$LOG"; then
    return 1
  fi
}

ensure_balanced() {
  local raw="$1" matrix="$2" operm="$3" grid="$4"
  if [ -f "$matrix" ] && [ -f "$operm" ]; then
    log "reusing existing v3-correct balanced matrix $matrix"
    return 0
  fi
  local raw_kb need_kb avail_kb
  raw_kb=$(du -k "$raw" | cut -f1)
  need_kb=$(( raw_kb * 6 ))
  avail_kb=$(free_kb)
  if [ "$avail_kb" -lt "$need_kb" ]; then
    log "SKIP: disk quota -- only ${avail_kb}KB free, want ${need_kb}KB margin for balancing $raw @ ${grid}x${grid}."
    return 1
  fi
  log "balancing $raw -> $matrix (${grid}x${grid}, --symmetric --operm)"
  timeout 3600 ./util/analyze --matrix "$raw" --symmetric --fabx "$grid" --faby "$grid" \
    --omatrix "$matrix" --operm "$operm" 2>&1 | tee -a "$LOG"
}

# Returns: 0 = ran OK, 1 = failed (any reason), 2 = grid too small (PE memory
# / task table ceiling) -- caller keeps trying larger grids either way, but
# rc==2 is logged distinctly since it's the "not yet at the min feasible
# grid" case rather than a genuine failure.
run_case() {
  local scale="$1" grid="$2" matrix="$3"
  local channels=$(( grid < 16 ? grid : 16 ))
  local matrix_base
  matrix_base="$(basename "$matrix" .mtx)"
  local OUT_ARGS=(
    "--csv=${CSV}"
    "--out-timing=bfs/results/hw/timing/timing_${matrix_base}_${grid}x${grid}_src${SOURCE}_ch${channels}.png"
  )

  local attempt=0
  while true; do
    log "compiling s${scale} @ ${grid}x${grid} (channels=${channels}), attempt $((attempt+1))"
    : > "$COMPILE_LOG"
    if timeout 3600 python "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$grid" --num_pe_rows="$grid" \
        --channels="$channels" --source="$SOURCE" --arch="$ARCH" --notree --compile-only 2>&1 \
        | tee -a "$COMPILE_LOG" | tee -a "$LOG" >/dev/null; then
      break
    fi
    if grep -qE "ran out of PE memory|for task table" "$COMPILE_LOG"; then
      log "SKIP: s${scale} @ ${grid}x${grid} -- grid too small (PE memory/task-table ceiling). Trying next larger grid."
      return 2
    fi
    if grep -qE "cannot open .*\.o: No such file or directory" "$COMPILE_LOG" && [ "$attempt" -lt "$MAX_LINKER_RETRIES" ]; then
      attempt=$((attempt+1))
      log "RETRY: s${scale} @ ${grid}x${grid} -- known linker file-vanished flake (docs/ERRORS.md #12), retry ${attempt}/${MAX_LINKER_RETRIES}"
      continue
    fi
    log "FAILED: s${scale} @ ${grid}x${grid} compile step failed (see log above) -- skipping this grid."
    return 1
  done

  log "running s${scale} @ ${grid}x${grid} (reads artifact_path.json)"
  if ! timeout 3600 python "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$grid" --num_pe_rows="$grid" \
      --channels="$channels" --source="$SOURCE" --arch="$ARCH" --notree "${OUT_ARGS[@]}" 2>&1 | tee -a "$LOG"; then
    log "FAILED: s${scale} @ ${grid}x${grid} run step failed -- see traceback above."
    return 1
  fi
  log "=== s${scale} @ ${grid}x${grid} OK ==="
  return 0
}

log "=== rmat_grid_sweep starting: scales ${SCALES[*]}, grids ${GRIDS[*]} ==="

for scale in "${SCALES[@]}"; do
  raw="data/rmat_s${scale}_e${EDGEFACTOR}.mtx"
  if ! ensure_raw "$scale" "$raw"; then
    log "SKIP scale ${scale}: gen_rmat.py failed"
    continue
  fi

  ran_any=false
  for grid in "${GRIDS[@]}"; do
    matrix="data/rmat_s${scale}_e${EDGEFACTOR}.balanced${grid}x${grid}.mtx"
    operm="data/rmat_s${scale}_e${EDGEFACTOR}.balanced${grid}x${grid}.operm"
    matrix_base="$(basename "$matrix" .mtx)"

    if already_ran "$matrix_base" "$grid"; then
      log "SKIP: s${scale} @ ${grid}x${grid} -- already has a row in ${CSV}"
      ran_any=true
      continue
    fi

    if ! ensure_balanced "$raw" "$matrix" "$operm" "$grid"; then
      continue
    fi

    run_case "$scale" "$grid" "$matrix"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      ran_any=true
    fi
    # rc==1 (failed) or rc==2 (too small): keep trying larger grids either way.
  done

  if [ "$ran_any" = false ]; then
    log "=== s${scale}: NO grid size in ${GRIDS[*]} worked ==="
  fi
done

rm -f "$COMPILE_LOG"
log "=== rmat_grid_sweep done ==="
