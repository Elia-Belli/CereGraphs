#!/usr/bin/env bash
# v2: symmetrized conversion (snap_to_mtx.py now does A|A^T) + --symmetric
# --operm balancing, so vertex identity survives balancing and the
# permutation is recoverable -- see project memory for why the original
# prep_snap.sh's output (no --symmetric, no symmetrization) didn't have
# either property. Regenerates data/snap_*.mtx (now symmetrized) and
# data/snap_*.balanced${GRID}x${GRID}.mtx + .operm (the recovered
# permutation, one line per original vertex id, 0-based).
set -uo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

GRID="${1:-750}"
LOG="benchmarks/prep_snap_v2.log"

declare -A SNAP_FILES=(
  [berkstan]="web-BerkStan.txt.gz"
  [orkut]="com-orkut.ungraph.txt.gz"
  [pokec]="soc-pokec-relationships.txt.gz"
  [topcats]="wiki-topcats.txt.gz"
  [livejournal]="soc-LiveJournal1.txt.gz"
)

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

for name in berkstan pokec livejournal topcats orkut; do
  raw="data/snap/${SNAP_FILES[$name]}"
  mtx="data/snap_${name}.mtx"
  balanced="data/snap_${name}.balanced${GRID}x${GRID}.mtx"
  operm="data/snap_${name}.balanced${GRID}x${GRID}.operm"

  log "converting (symmetrized) $raw -> $mtx"
  if ! cs_python benchmarks/snap_to_mtx.py "$raw" "$mtx" 2>&1 | tee -a "$LOG"; then
    log "FAILED converting $name, skipping"
    continue
  fi

  log "balancing (symmetric, with permutation dump) $mtx -> $balanced"
  if ! ./util/analyze --matrix "$mtx" --symmetric --fabx "$GRID" --faby "$GRID" \
      --omatrix "$balanced" --operm "$operm" 2>&1 | tee -a "$LOG"; then
    log "FAILED balancing $name, skipping"
    continue
  fi

  log "=== $name ready: $balanced (+ $operm) ==="
done

log "=== prep_snap_v2 done ==="
