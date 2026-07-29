#!/usr/bin/env bash
# v3: only symmetrize genuinely undirected SNAP graphs (checked against each
# dataset's own SNAP documentation -- only com-orkut is actually undirected,
# "com-orkut.ungraph"; berkstan/pokec/topcats/livejournal are all directed).
# The rest are balanced as directed via util/analyze's new --shared-perm
# (vertex-identity-preserving row/column permutation sharing, without
# --symmetric's requirement/fabrication of structural symmetry). See
# benchmarks/snap_to_mtx.py's own module docstring and project memory for
# why the earlier (v2) blanket-symmetrize approach was wrong for 4 of 5
# graphs -- it silently fabricated edges that don't exist in the real graph
# (berkstan's nnz nearly doubled), which for berkstan alone was enough to
# push it over the appliance's h2d transfer-size ceiling.
set -uo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

GRID="${1:-750}"
LOG="benchmarks/prep_snap_v3.log"

declare -A SNAP_FILES=(
  [berkstan]="web-BerkStan.txt.gz"
  [orkut]="com-orkut.ungraph.txt.gz"
  [pokec]="soc-pokec-relationships.txt.gz"
  [topcats]="wiki-topcats.txt.gz"
  [livejournal]="soc-LiveJournal1.txt.gz"
)
# Only genuinely undirected -- everyone else stays directed.
declare -A IS_UNDIRECTED=( [orkut]=1 )

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

for name in berkstan pokec livejournal topcats orkut; do
  raw="data/snap/${SNAP_FILES[$name]}"
  mtx="data/snap_${name}.mtx"
  balanced="data/snap_${name}.balanced${GRID}x${GRID}.mtx"
  operm="data/snap_${name}.balanced${GRID}x${GRID}.operm"

  convert_args=()
  balance_mode="--shared-perm"
  if [ "${IS_UNDIRECTED[$name]:-0}" = "1" ]; then
    convert_args+=("--symmetrize")
    balance_mode="--symmetric"
  fi

  log "converting $raw -> $mtx (${convert_args[*]:-directed, no symmetrize})"
  if ! cs_python benchmarks/snap_to_mtx.py "$raw" "$mtx" "${convert_args[@]}" 2>&1 | tee -a "$LOG"; then
    log "FAILED converting $name, skipping"
    continue
  fi

  log "balancing ($balance_mode, with permutation dump) $mtx -> $balanced"
  if ! ./util/analyze --matrix "$mtx" "$balance_mode" --fabx "$GRID" --faby "$GRID" \
      --omatrix "$balanced" --operm "$operm" 2>&1 | tee -a "$LOG"; then
    log "FAILED balancing $name, skipping"
    continue
  fi

  log "=== $name ready: $balanced (+ $operm, $balance_mode) ==="
done

log "=== prep_snap_v3 done ==="
