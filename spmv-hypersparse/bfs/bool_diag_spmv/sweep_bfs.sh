#!/usr/bin/env bash
# BFS scale sweep for the poster's main benchmark: runs run_bfs.py (simulator)
# or run_bfs.appliance.py (real hardware, two-phase compile-then-run -- see
# that script's own module docstring) across a list of (RMAT scale,
# edgefactor, PE grid) test cases, generating + balancing any matrix that
# doesn't already exist in data/. Each run appends its own row to
# bfs/bool_diag_spmv/results/bfs_timing.csv (run_bfs.py's own default) and
# renders its own per-run timing plot (plot_bfs_timing.py) -- this script
# doesn't touch either, it only drives the sweep.
#
# Edit CASES below to whatever (scale, edgefactor, grid) triples you actually
# want -- this is a small smoke-test default. One row per case: "scale
# edgefactor grid" (grid is a single int -- the design requires a square
# PxP grid).
#
# Usage:
#   ./bfs/bool_diag_spmv/sweep_bfs.sh                    # simulator (local, cs_python), wse2
#   ./bfs/bool_diag_spmv/sweep_bfs.sh appliance-sim      # appliance client, simulator backend, wse3
#   ./bfs/bool_diag_spmv/sweep_bfs.sh appliance          # appliance client, REAL hardware, wse3
#
# `simulator` uses run_bfs.py through this repo's local cs_python container
# wrapper -- what every other command_wse*.sh script in this repo already
# does, no cluster access needed.
#
# `appliance-sim` and `appliance` both use run_bfs.appliance.py via plain
# `python` (NOT cs_python -- see that script's own module docstring for why:
# the ALCF cluster's cerebras.sdk.client talks to the job scheduler directly
# over the network, no local container wrapper involved) with only their
# `simulator=`/`--fabric-dims` handling differing. Run `appliance-sim` FIRST
# on a real ALCF login/compute node before ever running `appliance` for
# real -- it validates the exact same appliance-mode code path (SdkCompiler/
# SdkRuntime, artifact_path.json handoff) against the actual cluster's
# software stack, without spending a real hardware allocation, and is the
# closest thing to a dry run this script can offer.
#
# Relocates to the repo root itself (same convention as commands_wse2.sh),
# so it's safe to invoke from anywhere.

set -e

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

MODE="${1:-simulator}"
SOURCE=0

CASES=(
  "12 16 16"
  "14 16 16"
  "16 16 32"
  "18 16 64"
  # "19 16 64"  # pilot -- uncomment once s12-s18 are confirmed clean
  # "20 16 64"  # stretch goal -- only after s19's pilot succeeds
)

case "$MODE" in
  appliance)
    ARCH=wse3
    RUN_SCRIPT="bfs/bool_diag_spmv/run_bfs.appliance.py"
    PYTHON=python
    SIM_FLAG=""
    echo "=== appliance mode: REAL hardware, --arch=$ARCH ==="
    ;;
  appliance-sim)
    ARCH=wse3
    RUN_SCRIPT="bfs/bool_diag_spmv/run_bfs.appliance.py"
    PYTHON=python
    SIM_FLAG="--simulator"
    echo "=== appliance-sim mode: appliance client, simulator backend, --arch=$ARCH ==="
    ;;
  simulator)
    ARCH=wse2
    RUN_SCRIPT="bfs/bool_diag_spmv/run_bfs.py"
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

  echo ""
  echo "=== s${scale} e${edgefactor} @ ${grid}x${grid} ==="

  if [ ! -f "$matrix" ]; then
    if [ ! -f "$raw" ]; then
      echo "-- generating $raw"
      cs_python benchmarks/gen_rmat.py "$scale" "$edgefactor" 0 "$raw"
    fi
    echo "-- balancing $raw -> $matrix (util/analyze, ${grid}x${grid} grid)"
    ./util/analyze --matrix "$raw" --rand 0 --fabx "$grid" --faby "$grid" --omatrix "$matrix"
  fi

  if [ "$MODE" = "appliance" ] || [ "$MODE" = "appliance-sim" ]; then
    echo "-- compiling (writes artifact_path.json)"
    "$PYTHON" "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$grid" --num_pe_rows="$grid" \
      --channels=1 --source="$SOURCE" --arch="$ARCH" $SIM_FLAG --compile-only
    echo "-- running (reads artifact_path.json)"
    "$PYTHON" "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$grid" --num_pe_rows="$grid" \
      --channels=1 --source="$SOURCE" --arch="$ARCH" $SIM_FLAG
  else
    "$PYTHON" "$RUN_SCRIPT" --infile_mtx="$matrix" --num_pe_cols="$grid" --num_pe_rows="$grid" \
      --channels=1 --source="$SOURCE" --arch="$ARCH" --driver=cslc
  fi
done

echo ""
echo "=== sweep done -- see bfs/bool_diag_spmv/results/bfs_timing.csv ==="
