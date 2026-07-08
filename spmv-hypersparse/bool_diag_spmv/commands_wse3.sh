#!/usr/bin/env bash

set -e

# See the matching comment in commands_wse2.sh: relocate to the repo root so
# repo-root-relative paths (src/, data/) resolve inside the container's
# cwd-only bind mount.
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

# Unlike original_spmv/bfs_spmv, bool_diag_spmv routes all PE-to-PE traffic
# through the <collectives_2d> stdlib instead of hand-rolled DSR/queue
# assignments, so the same src/ compiles unmodified for both wse2 and wse3
# (verified: identical bool_pe.csl/layout_bool.csl, --arch is the only
# difference) -- no separate src_wse3/ tree needed here.
cslc bool_diag_spmv/src/layout_bool.csl --arch wse3 --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=pcols:4,prows:4,blk:4 \
--params=max_local_nnz:8,max_local_nnz_cols:4,max_local_nnz_rows:4 -o=bool_diag_spmv/out_wse3 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0

cs_python bool_diag_spmv/run_bool.py --arch=wse3 --num_pe_cols=4 --num_pe_rows=4 --latestlink bool_diag_spmv/out_wse3 --channels=1 \
--width-west-buf=0 --width-east-buf=0 --run-only \
--infile_mtx=data/rmat4.4x4.lb.mtx
