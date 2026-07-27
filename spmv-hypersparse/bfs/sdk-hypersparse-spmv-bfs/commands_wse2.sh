#!/usr/bin/env bash

set -e

# The cslc/cs_python container wrapper only binds the invocation's current
# directory into its sandbox, so this script always moves to the repo root
# first (spmv-hypersparse/) and uses repo-root-relative paths throughout --
# that's the one cwd that's an ancestor of both this version's src/ and the
# shared data/ directory. Safe to invoke this script from anywhere.
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

# Unlike sdk-hypersparse-spmv/bool_diag_spmv's commands_wse2.sh, this is a single
# call (no separate `cslc ... -o=...` + `run_bfs.py --run-only` step): the
# compile-time params (max_local_nnz & friends) depend on preprocess()'s
# transposed partition (see run_bfs.py's module docstring, point 2), which
# is only known once run_bfs.py has loaded the matrix -- so compile and run
# happen together here.
cs_python bfs/sdk-hypersparse-spmv-bfs/run_bfs.py --arch wse2 --num_pe_cols=4 --num_pe_rows=4 --channels=1 \
--width-west-buf=0 --width-east-buf=0 --latestlink bfs/sdk-hypersparse-spmv-bfs/out \
--infile_mtx=data/rmat4.4x4.lb.mtx --source=0
