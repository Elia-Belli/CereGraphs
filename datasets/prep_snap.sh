#!/usr/bin/env bash
# Only symmetrize genuinely undirected SNAP graphs (checked against each
# dataset's own SNAP documentation -- only com-orkut is actually undirected,
# "com-orkut.ungraph"; berkstan/pokec/topcats/livejournal are all directed).
# The rest are balanced as directed via util/analyze's --shared-perm
# (vertex-identity-preserving row/column permutation sharing, without
# --symmetric's requirement/fabrication of structural symmetry). See
# datasets/snap_to_mtx.py's own module docstring and project memory for why
# an earlier blanket-symmetrize approach was wrong for 4 of 5 graphs -- it
# silently fabricated edges that don't exist in the real graph (berkstan's
# nnz nearly doubled), which for berkstan alone was enough to push it over
# the appliance's h2d transfer-size ceiling.
set -uo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

GRID="${1:-750}"
LOG="datasets/prep_snap.log"

declare -A SNAP_FILES=(
  [berkstan]="web-BerkStan.txt.gz"
  [orkut]="com-orkut.ungraph.txt.gz"
  [pokec]="soc-pokec-relationships.txt.gz"
  [topcats]="wiki-topcats.txt.gz"
  [livejournal]="soc-LiveJournal1.txt.gz"
  [skitter]="as-skitter.txt.gz"
  [patents]="cit-Patents.txt.gz"
)
# Only genuinely undirected -- everyone else stays directed. as-Skitter
# confirmed undirected by SNAP's own dataset page (2026-07-29); cit-Patents
# confirmed directed, so it's left out (defaults to --shared-perm below).
declare -A IS_UNDIRECTED=( [orkut]=1 [skitter]=1 )

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# Optional name args (matches download_snap_graphs.sh's own convention) --
# e.g. `prep_snap.sh 750 skitter patents` to (re)balance just the new two
# without re-converting/re-balancing the other five.
names=("${@:2}")
if [ ${#names[@]} -eq 0 ]; then
  names=(berkstan pokec livejournal topcats orkut skitter patents)
fi

for name in "${names[@]}"; do
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
  if ! cs_python datasets/snap_to_mtx.py "$raw" "$mtx" "${convert_args[@]}" 2>&1 | tee -a "$LOG"; then
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

log "=== prep_snap done ==="
