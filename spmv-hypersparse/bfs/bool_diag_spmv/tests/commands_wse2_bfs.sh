#!/usr/bin/env bash
# Simulator-only, hardware-free smoke test for run_bfs.py's parent_local_buf
# readback path -- the fast validation loop for the on-device parent
# resolution plan's Phase A (u32 global-id widening) and, later, Phase B
# (the new reduce_select_any collective). Mirrors commands_wse2.sh's exact
# compile line (same 4x4 grid, blk=4, data/rmat4.4x4.lb.mtx fixture) but
# runs run_bfs.py (which exercises f_spmv_iter/parent_local_buf/
# extract_parent_result end-to-end, including the scipy cross-check)
# instead of commands_wse2.sh's run_single_spmv.py (which never touches
# parent_local_buf at all -- is_iterative is hardcoded false there).
#
# Usage: ./bfs/bool_diag_spmv/tests/commands_wse2_bfs.sh [extra run_bfs.py args...]
#   e.g. ./bfs/bool_diag_spmv/tests/commands_wse2_bfs.sh --infile_mtx=data/collision4x4.mtx

set -e

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"

cslc bfs/bool_diag_spmv/src/layout_bool.csl --arch wse2 --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=pcols:4,prows:4,blk:4 \
--params=max_local_nnz:8,max_local_nnz_cols:4,max_local_nnz_rows:4 -o=bfs/bool_diag_spmv/out/wse2_bfs \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0

cs_python bfs/bool_diag_spmv/run_bfs.py --arch=wse2 --num_pe_cols=4 --num_pe_rows=4 \
--latestlink bfs/bool_diag_spmv/out/wse2_bfs --channels=1 \
--width-west-buf=0 --width-east-buf=0 --run-only --driver=cslc \
--infile_mtx=data/rmat4.4x4.lb.mtx --source=0 "$@"
