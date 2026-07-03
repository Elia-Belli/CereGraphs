#!/usr/bin/env bash

set -e

# The cslc/cs_python container wrapper only binds the invocation's current
# directory into its sandbox, so this script always moves to the repo root
# first (spmv-hypersparse/) and uses repo-root-relative paths throughout --
# that's the one cwd that's an ancestor of both this version's src/ and the
# shared data/ directory. Safe to invoke this script from anywhere.
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

cslc bool_diag_spmv/src/layout_bool.csl --arch wse2 --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=pcols:4,prows:4,blk:4 \
--params=max_local_nnz:8,max_local_nnz_cols:4,max_local_nnz_rows:4 -o=bool_diag_spmv/out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0

cs_python bool_diag_spmv/run_bool.py --num_pe_cols=4 --num_pe_rows=4 --latestlink bool_diag_spmv/out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --run-only \
--infile_mtx=data/rmat4.4x4.lb.mtx
