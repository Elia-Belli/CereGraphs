#!/usr/bin/env bash

set -e

# See the matching comment in commands_wse2.sh: relocate to the repo root so
# repo-root-relative paths (src/, data/) resolve inside the container's
# cwd-only bind mount.
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

# Same compiled kernel as commands_wse3.sh (f_spmv and f_spmv_iter are both
# exported from src/layout_bool.csl) -- this just runs run_host_driven_bfs.py
# instead of run_single_spmv.py against it, to check the on-device MAX_ITERS
# loop in bool_pe.csl (f_spmv_iter) against MAX_ITERS sequential host-driven
# f_spmv launches.
cslc bool_diag_spmv/src/layout_bool.csl --arch wse3 --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=pcols:4,prows:4,blk:4 \
--params=max_local_nnz:8,max_local_nnz_cols:4,max_local_nnz_rows:4 -o=bool_diag_spmv/out/wse3_iter \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0

cs_python bool_diag_spmv/run_host_driven_bfs.py --arch=wse3 --num_pe_cols=4 --num_pe_rows=4 \
--latestlink bool_diag_spmv/out/wse3_iter --channels=1 \
--width-west-buf=0 --width-east-buf=0 --run-only \
--infile_mtx=data/rmat4.4x4.lb.mtx
