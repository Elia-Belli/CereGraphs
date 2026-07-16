#!/usr/bin/env bash

set -e

# The cslc/cs_python container wrapper only binds the invocation's current
# directory into its sandbox, so this script always moves to the repo root
# first (spmv-hypersparse/) and uses repo-root-relative paths throughout --
# that's the one cwd that's an ancestor of both this version's src/ and the
# shared data/ directory. Safe to invoke this script from anywhere.
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

cslc original_spmv/src/layout.csl --arch wse2 --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=ncols:16,nrows:16,pcols:4,prows:4,max_local_nnz:8 \
--params=max_local_nnz_cols:4,max_local_nnz_rows:4,local_vec_sz:1 \
--params=local_out_vec_sz:1,y_pad_start_row_idx:4 -o=original_spmv/out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0

cs_python original_spmv/run.py --num_pe_cols=4 --num_pe_rows=4 --latestlink original_spmv/out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --is_weight_one --run-only \
--infile_mtx=data/rmat4.4x4.lb.mtx
