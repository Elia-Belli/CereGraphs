#!/usr/bin/env bash
# BFS scale sweep for the poster's main benchmark: runs run_bfs.py (simulator)
# or run_bfs.appliance.py (real hardware, two-phase compile-then-run) across
# a list of (RMAT scale, edgefactor, PE grid) test cases, generating +
# balancing any matrix that doesn't already exist in data/. Each run appends
# its own row to bfs/results/sim|hw/bfs_timing.csv and renders its own
# per-run timing plot -- this script only drives the sweep.
#
# Edit CASES below to whatever (scale, edgefactor, grid) triples you want --
# this is a small smoke-test default. One row per case: "scale edgefactor
# grid" (grid is a single int -- the design requires a square PxP grid).
#
# Usage (wse3 only -- no longer tested/supported on wse2):
#   ./bfs/scripts/sweep_bfs.sh                    # simulator (local, cs_python), wse3
#   ./bfs/scripts/sweep_bfs.sh appliance-sim      # appliance client, simulator backend, wse3
#   ./bfs/scripts/sweep_bfs.sh appliance          # appliance client, REAL hardware, wse3
#
# `appliance-sim`/`appliance` both use run_bfs.appliance.py via plain
# `python`, not cs_python (the ALCF cluster's cerebras.sdk.client talks to
# the job scheduler directly, no local container wrapper). Run
# `appliance-sim` first on a real ALCF node before `appliance` -- it
# validates the same code path against the cluster's software stack without
# spending a real hardware allocation.
#
# Relocates to the repo root, so it's safe to invoke from anywhere.

set -e

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

# util/analyze is a native binary, not committed -- build (or rebuild, if
# stale) it for this node's toolchain before the sweep's first use.
make -C util

MODE="${1:-simulator}"
SOURCE=0

CASES=(
  "18 16 64"
)

case "$MODE" in
  appliance)
    ARCH=wse3
    RUN_SCRIPT="bfs/scripts/run_bfs.appliance.py"
    PYTHON=python
    SIM_FLAG=""
    # SdkRuntime's memcpy gRPC streams get reset if https_proxy/HTTPS_PROXY
    # (needed for pip through ALCF's proxy) also routes this internal cluster
    # traffic -- exclude it; append to (not clobber) any no_proxy already set.
    export no_proxy="10.125.8.2,.cerebras.internal,localhost,127.0.0.1${no_proxy:+,$no_proxy}"
    export NO_PROXY="$no_proxy"
    echo "=== appliance mode: REAL hardware, --arch=$ARCH ==="
    ;;
  appliance-sim)
    ARCH=wse3
    RUN_SCRIPT="bfs/scripts/run_bfs.appliance.py"
    PYTHON=python
    SIM_FLAG="--simulator"
    export no_proxy="10.125.8.2,.cerebras.internal,localhost,127.0.0.1${no_proxy:+,$no_proxy}"
    export NO_PROXY="$no_proxy"
    echo "=== appliance-sim mode: appliance client, simulator backend, --arch=$ARCH ==="
    ;;
  simulator)
    ARCH=wse3
    RUN_SCRIPT="bfs/scripts/run_bfs.py"
    PYTHON=cs_python
    SIM_FLAG=""
    echo "=== simulator mode (local cs_python): --arch=$ARCH ==="
    ;;
  *)
    echo "Unknown mode '$MODE' -- expected 'simulator', 'appliance-sim', or 'appliance'" >&2
    exit 1
    ;;
esac

for case in "${CASES[@]}"; do
  read -r scale edgefactor grid <<< "$case"
  matrix="data/rmat_s${scale}_e${edgefactor}.balanced${grid}x${grid}.mtx"
  raw="data/rmat_s${scale}_e${edgefactor}.mtx"
  # Max I/O channels for this grid size: SDK's documented cap is 16, but we
  # additionally cap at the grid's own edge width/height (channels are
  # assumed to map to fabric-edge columns) since that combo is untested
  # below grid=16.
  channels=$(( grid < 16 ? grid : 16 ))

  echo ""
  echo "=== s${scale} e${edgefactor} @ ${grid}x${grid} (channels=${channels}) ==="

  if [ ! -f "$matrix" ]; then
    if [ ! -f "$raw" ]; then
      echo "-- generating $raw"
      cs_python datasets/gen_rmat.py "$scale" "$edgefactor" 0 "$raw"
    fi
    echo "-- balancing $raw -> $matrix (util/analyze, ${grid}x${grid} grid)"
    ./util/analyze --matrix "$raw" --symmetric --rand 0 --fabx "$grid" --faby "$grid" --omatrix "$matrix"
  fi

  # Real hardware gets its own csv/plot folder, kept separate from
  # simulator/appliance-sim results.
  OUT_ARGS=()
  if [ "$MODE" = "appliance" ]; then
    matrix_base="$(basename "$matrix" .mtx)"
    OUT_ARGS=(
      "--csv=bfs/results/hw/bfs_timing.csv"
      "--out-timing=bfs/results/hw/timing/timing_${matrix_base}_${grid}x${grid}_src${SOURCE}_ch${channels}.png"
    )
  fi

  if [ "$MODE" = "appliance" ] || [ "$MODE" = "appliance-sim" ]; then
    echo "-- compiling (writes artifact_path.json)"
    "$PYTHON" "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$grid" --num_pe_rows="$grid" \
      --channels="$channels" --source="$SOURCE" --arch="$ARCH" $SIM_FLAG --notree --compile-only
    echo "-- running (reads artifact_path.json)"
    "$PYTHON" "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$grid" --num_pe_rows="$grid" \
      --channels="$channels" --source="$SOURCE" --arch="$ARCH" $SIM_FLAG --notree "${OUT_ARGS[@]}"
  else
    "$PYTHON" "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$grid" --num_pe_rows="$grid" \
      --channels="$channels" --source="$SOURCE" --arch="$ARCH" --driver=cslc --notree "${OUT_ARGS[@]}"
  fi
done

echo ""
echo "=== sweep done -- see bfs/results/sim/bfs_timing.csv or bfs/results/hw/bfs_timing.csv ==="
