#!/usr/bin/env bash

set -e

# Relocate to the repo root so repo-root-relative paths (src/, data/)
# resolve inside the container's cwd-only bind mount.
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

# Same compiled kernel as run_bfs.py's own smoke-test invocations --
# this just runs run_graph500.py against it: one matrix upload, then
# --num-searches single-source BFS searches from distinct random roots
# (defaults to all 16 vertices here, since the fixture is tiny), timed
# individually and combined via harmonic mean GTEPS. Small matrix/grid here
# purely to keep this a fast sanity check -- see docs/GRAPH500_BENCHMARK.md for
# a real-scale (64-search, larger matrix) run.
cslc bfs/implementation/src/layout_bool.csl --arch wse3 --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=pcols:4,prows:4,blk:4 \
--params=max_local_nnz:8,max_local_nnz_cols:4,max_local_nnz_rows:4,max_rounds:10 \
-o=bfs/out/wse3_graph500 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0

cs_python bfs/scripts/run_graph500.py --arch=wse3 --num_pe_cols=4 --num_pe_rows=4 \
--latestlink bfs/out/wse3_graph500 --channels=1 \
--width-west-buf=0 --width-east-buf=0 --run-only \
--infile_mtx=data/rmat4.4x4.lb.mtx --max-rounds=10 --num-searches=16 --seed=0
