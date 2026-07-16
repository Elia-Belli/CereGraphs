#!/usr/bin/env bash

set -e

# See the matching comment in commands_wse2.sh: relocate to the repo root so
# repo-root-relative paths (src_wse3/, data/) resolve inside the container's
# cwd-only bind mount.
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

# src_wse3/ carries WSE-3-specific kernel changes (see
# ../original_spmv/README.md and
# github.com/Cerebras/sdk-examples/pull/23) that are NOT backwards
# compatible with WSE-2. As in commands_wse2.sh, compile and run happen in
# one call since the compile-time params depend on run_bfs.py's (possibly
# transposed) partition of the matrix.
cs_python bfs_spmv/run_bfs.py --arch wse3 --num_pe_cols=8 --num_pe_rows=8 --channels=1 \
--width-west-buf=0 --width-east-buf=0 --latestlink bfs_spmv/out_wse3 \
--infile_mtx=data/rand600.mtx --source=0
#--infile_mtx=data/rmat4.4x4.lb.mtx --source=0
