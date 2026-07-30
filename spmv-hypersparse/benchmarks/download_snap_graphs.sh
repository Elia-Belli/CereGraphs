#!/usr/bin/env bash
#
# Downloads real-world SNAP graph datasets (web-BerkStan, com-orkut,
# soc-pokec, wiki-topcats, soc-LiveJournal1, as-Skitter, cit-Patents) into
# data/snap/ as raw .txt.gz edge lists -- unlike gen_rmat.py's synthetic
# graphs, these are fetched, not generated. Immediately usable via
# --infile_mtx as-is: no MTX conversion needed, bool_diag_spmv/graph_loader.py's
# edge-list support reads a SNAP .txt.gz file directly.
#
# Usage: benchmarks/download_snap_graphs.sh [name ...]
#   No args: downloads all seven datasets below. Otherwise downloads only the
#   named subset (e.g. `download_snap_graphs.sh berkstan` to skip waiting on
#   orkut/livejournal's much larger downloads).
#
# NOTE: the original five URLs (including orkut's SNAP_URL_OVERRIDE path)
# were independently verified against snap.stanford.edu on 2026-07-28 from
# cer-usn-01; skitter/patents were verified (200 OK) on 2026-07-29 -- if a
# fetch 404s again later, check https://snap.stanford.edu/data/ and fix the
# matching entry in SNAP_FILES/SNAP_URL_OVERRIDE below.

set -e

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

declare -A SNAP_FILES=(
  [berkstan]="web-BerkStan.txt.gz"
  [orkut]="com-orkut.ungraph.txt.gz"
  [pokec]="soc-pokec-relationships.txt.gz"
  [topcats]="wiki-topcats.txt.gz"
  [livejournal]="soc-LiveJournal1.txt.gz"
  [skitter]="as-skitter.txt.gz"
  [patents]="cit-Patents.txt.gz"
)

# Per-dataset URL override for entries that don't live directly under
# data/$filename -- confirmed against snap.stanford.edu on 2026-07-28: orkut
# is filed under data/bigdata/communities/, everything else above is at the
# plain top-level path.
declare -A SNAP_URL_OVERRIDE=(
  [orkut]="https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz"
)

names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  names=(berkstan orkut pokec topcats livejournal skitter patents)
fi

mkdir -p data/snap

for name in "${names[@]}"; do
  filename="${SNAP_FILES[$name]:-}"
  if [ -z "$filename" ]; then
    echo "unknown dataset '$name' -- choices: ${!SNAP_FILES[*]}" >&2
    exit 1
  fi

  dest="data/snap/$filename"
  url="${SNAP_URL_OVERRIDE[$name]:-https://snap.stanford.edu/data/$filename}"

  if [ -f "$dest" ]; then
    echo "[$name] already present at $dest, skipping"
  else
    echo "[$name] downloading $url -> $dest"
    # Download to a .part file and only rename on success, so a file at
    # $dest is never partial -- the idempotency check above can trust it,
    # and -C - can resume the .part file itself across interrupted reruns
    # (orkut/livejournal are GB-scale; resumability matters).
    curl -fL -C - -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
  fi

  echo "[$name] $(du -h "$dest" | cut -f1) on disk"
  declared=$(zcat "$dest" | head -20 | grep -m1 -oE 'Nodes:[[:space:]]*[0-9]+[[:space:]]*Edges:[[:space:]]*[0-9]+' || true)
  if [ -n "$declared" ]; then
    echo "[$name] file declares: $declared"
  else
    echo "[$name] no 'Nodes: ... Edges: ...' header line found in first 20 lines"
  fi
done
