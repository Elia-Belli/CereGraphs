#!/usr/bin/env bash

set -e

# See the matching comment in commands_wse2.sh: relocate to the repo root so
# repo-root-relative paths (src_wse3/, ../../data/) resolve inside the
# container's cwd-only bind mount.
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

# src_wse3/ carries WSE-3-specific kernel changes (queue remapping dropping
# allreduce2R1E-based sync -- see README.md and
# github.com/Cerebras/sdk-examples/pull/23) that are NOT backwards
# compatible with WSE-2, hence the separate source tree and script.
cslc spmv/sdk-hypersparse-spmv/src_wse3/layout.csl --arch wse3 --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=ncols:16,nrows:16,pcols:4,prows:4,max_local_nnz:8 \
--params=max_local_nnz_cols:4,max_local_nnz_rows:4,local_vec_sz:1 \
--params=local_out_vec_sz:1,y_pad_start_row_idx:4 -o=spmv/sdk-hypersparse-spmv/out_wse3 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0

cs_python spmv/sdk-hypersparse-spmv/run.py --arch=wse3 --num_pe_cols=4 --num_pe_rows=4 --latestlink spmv/sdk-hypersparse-spmv/out_wse3 \
--channels=1 --width-west-buf=0 --width-east-buf=0 --is_weight_one --run-only \
--infile_mtx=data/rmat4.4x4.lb.mtx
