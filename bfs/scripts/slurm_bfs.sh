#!/usr/bin/env bash

#SBATCH --job-name=cerebras-bfs
#SBATCH --output=out/bfs-s15-64x64.txt    
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --account=IscrC_FOCAL_0
#SBATCH --partition=dcgp_usr_prod

module load gcc

set -e

SCALE=15
GRID=64

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$(cd -- "$SLURM_SUBMIT_DIR/../.." &>/dev/null && pwd)"
else
  cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
fi

MTX="data/rmat_s${SCALE}_e16.mtx"
BALANCED="data/rmat_s${SCALE}_e16.balanced${GRID}x${GRID}.mtx"

# 0) Build (or rebuild, if stale) util/analyze for this node's toolchain --
# not committed as a binary, see util/Makefile's own header comment.
make -C util

# 1) Generate the scaleX/edgefactor16/seed0 R-MAT graph
if [ ! -f "$MTX" ]; then
  cs_python datasets/gen_rmat.py ${SCALE} 16 0 "$MTX"
fi

# 2) Load-balance it for a GxG PE grid via util/analyze
if [ ! -f "$BALANCED" ]; then
  util/analyze --matrix "$MTX" --omatrix "$BALANCED" \
      --fabx ${GRID} --faby ${GRID} --symmetric --remove_dup
fi


# 3) Run a single-source BFS on the simulator at GxG PE grid with 16 channels and no buffers
cs_python bfs/scripts/run_bfs.py --arch=wse3 \
    --num_pe_cols=${GRID} --num_pe_rows=${GRID} --channels=16 \
    --width-west-buf=0 --width-east-buf=0 \
    --infile_mtx="$BALANCED" \
    --latestlink bfs/out/s${SCALE}_${GRID}x${GRID} \
    --max-rounds=10 --source=0 --notree
