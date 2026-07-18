#!/usr/bin/env bash

#SBATCH --job-name=cerebras-graph500
#SBATCH --output=out/graph500-s12-64x64.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_FOCAL_0
#SBATCH --partition=dcgp_usr_prod

module load gcc

set -e

SCALE=12
GRID=64

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$(cd -- "$SLURM_SUBMIT_DIR/.." &>/dev/null && pwd)"
else
  cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
fi

MTX="data/rmat_s${SCALE}_e16.mtx"
BALANCED="data/rmat_s${SCALE}_e16.balanced${GRID}x${GRID}.mtx"

# 0) Rebuild util/analyze for this node's toolchain
if [ util/analyze.cpp -nt util/analyze ] || [ util/mmio.c -nt util/analyze ] \
    || [ ! -x util/analyze ]; then
  g++ -Wall -g -I util/include -std=c++17 -O3 -g -c util/analyze.cpp -o util/analyze.o
  gcc -c util/mmio.c -o util/mmio.o
  g++ -Wall -g -I util/include -std=c++17 -O3 -g -o util/analyze util/analyze.o util/mmio.o
fi

# 1) Generate the scaleX/edgefactor16/seed0 R-MAT graph
if [ ! -f "$MTX" ]; then
  cs_python benchmarks/gen_rmat.py ${SCALE} 16 0 "$MTX"
fi

# 2) Load-balance it for a GxG PE grid via util/analyze
if [ ! -f "$BALANCED" ]; then
  util/analyze --matrix "$MTX" --omatrix "$BALANCED" \
      --fabx ${GRID} --faby ${GRID} --symmetric --remove_dup
fi

# 3) Run the full Graph500-style benchmark (one compile, one matrix upload,
#    then --num-searches distinct-root single-source BFS searches, harmonic
#    mean GTEPS across all of them -- see run_graph500.py's own docstring)
#    on the simulator at GxG PE grid with 16 channels and no buffers.
cs_python bool_diag_spmv/run_graph500.py --arch=wse3 \
    --num_pe_cols=${GRID} --num_pe_rows=${GRID} --channels=16 \
    --width-west-buf=0 --width-east-buf=0 \
    --infile_mtx="$BALANCED" \
    --latestlink bool_diag_spmv/out/graph500_s${SCALE}_${GRID}x${GRID} \
    --max-rounds=15 --num-searches=64 --seed=0
