#!/usr/bin/env bash
# Converts each downloaded SNAP graph (data/snap/*.txt.gz) to Matrix Market
# via snap_to_mtx.py, then balances it onto a GRID-by-GRID PE grid via
# util/analyze -- same two-step pipeline sweep_bfs.sh/grow_sweep.sh already
# run for RMAT matrices, just pointed at real-world graphs instead. Purely
# host-side (cs_python + util/analyze, no appliance/cluster job involved),
# so safe to run concurrently with an appliance-mode compile/run in
# progress.
#
# Usage: ./benchmarks/prep_snap.sh [grid]   (grid defaults to 750)

set -uo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

GRID="${1:-750}"
LOG="benchmarks/prep_snap.log"

declare -A SNAP_FILES=(
  [berkstan]="web-BerkStan.txt.gz"
  [orkut]="com-orkut.ungraph.txt.gz"
  [pokec]="soc-pokec-relationships.txt.gz"
  [topcats]="wiki-topcats.txt.gz"
  [livejournal]="soc-LiveJournal1.txt.gz"
)

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

for name in berkstan orkut pokec topcats livejournal; do
  raw="data/snap/${SNAP_FILES[$name]}"
  mtx="data/snap_${name}.mtx"
  balanced="data/snap_${name}.balanced${GRID}x${GRID}.mtx"

  if [ ! -f "$mtx" ]; then
    log "converting $raw -> $mtx"
    if ! cs_python benchmarks/snap_to_mtx.py "$raw" "$mtx" 2>&1 | tee -a "$LOG"; then
      log "FAILED converting $name, skipping"
      continue
    fi
  else
    log "$mtx already exists, skipping conversion"
  fi

  if [ ! -f "$balanced" ]; then
    log "balancing $mtx -> $balanced (${GRID}x${GRID} grid)"
    if ! ./util/analyze --matrix "$mtx" --rand 0 --fabx "$GRID" --faby "$GRID" --omatrix "$balanced" 2>&1 | tee -a "$LOG"; then
      log "FAILED balancing $name, skipping"
      continue
    fi
  else
    log "$balanced already exists, skipping balance"
  fi

  log "=== $name ready: $balanced ==="
done

log "=== prep_snap done ==="
