# bool_diag_spmv — boolean-semiring SpMV, diagonal-reduce, iterative

A from-scratch redesign of `../original_spmv` for a different target
workload: boolean-semiring SpMV (`y = OR_j (A[i,j] AND x[j])`) on a *square*
adjacency matrix, as the building block for a future BFS implementation (not
yet built — see "Status" below).

## Files

- `src/layout_bool.csl` — top-level layout: sets up `memcpy` and
  `<collectives_2d>` params, asserts a square PE grid, exports buffers and
  `f_spmv`/timing functions.
- `src/bool_pe.csl` — the whole per-PE kernel in one flat file (no nested
  module-import layer like `original_spmv/src/hypersparse_spmv/`) — modeled
  on the SDK's `gemv-collectives_2d/pe.csl` example.
- `preprocess_bool.py` — structural fork of `../original_spmv/preprocess.py`:
  identical hypersparse compressed-column partitioning
  (`mat_col_idx/loc/len_buf`, `mat_rows_buf`, `y_rows_init_buf`), minus
  `mat_vals_buf` (boolean semiring never uses edge weights).
- `run_bool.py` — host driver: seeds `x` only at the PE grid's diagonal,
  launches `f_spmv` once, reads back the full rectangle, keeps only the
  diagonal entries, verifies against an independent scipy boolean reference.
- `test_iterative.py` — host driver for the on-device iterative entrypoint
  `f_spmv_iter` (see "Status" below): runs one `f_spmv_iter` launch (5
  internal rounds) and, against the same compiled kernel, 5 sequential
  single-shot `f_spmv` launches with the host manually copying each round's
  `y` back into `x` between launches — then checks the two final vectors are
  bit-identical. Logs each run to `iterative_results.jsonl`.
- `commands_wse2.sh` / `commands_wse3.sh` — one-shot compile+run smoke test on
  `../data/rmat4.4x4.lb.mtx` at a 4x4 grid, for WSE-2 and WSE-3 respectively.
  Unlike `original_spmv`/`bfs_spmv`, both scripts compile the *same* `src/` —
  `<collectives_2d>` abstracts the fabric-routing differences between the two
  architectures, so there's no separate `src_wse3/` tree here.
- `commands_wse3_iterative.sh` — same compile as `commands_wse3.sh`, but runs
  `test_iterative.py` instead of `run_bool.py`.

## Design: why the diagonal, and why `<collectives_2d>`

Communication is targeted at the grid diagonal (`pcol_id == prow_id`) instead
of a fixed corner or a full `P^2` scatter (contrast with `original_spmv`):

- **Phase 1 (broadcast)**: only the diagonal PE of a column starts with real
  `x` data (seeded by the host via memcpy); it's broadcast to the rest of the
  column via `mpi_y.broadcast(root=pcol_id, ...)`. One data item moves, one
  round — not an all-gather from every PE.
- **Phase 2 (reduce)**: every PE's local boolean contribution is reduced
  toward *its own row's* diagonal column via `mpi_x.reduce_fadds(root=prow_id,
  ...)`. Boolean OR is implemented as float add + nonzero-check (values are
  0.0/1.0, so `sum > 0 <=> OR`) — the only reduce primitive `<collectives_2d>`
  exposes is `reduce_fadds`.

Both primitives come from the SDK's `<collectives_2d>` stdlib module (see the
bundled `gemv-collectives_2d`/`gemm-collectives_2d`/`topic-11-collectives`
examples), not hand-rolled fabric routing — a hand-rolled 4-color
parity-based routing scheme was drafted and abandoned once this library was
found; it does the same job with far less code to get wrong.

Targeting the diagonal specifically (not a corner) is what makes iterating
cheap: the PE that produces row `py`'s final result is exactly the PE that
needs to *source* the next round's broadcast down column `py` — reusing `y`
as the next round's `x` (`f_spmv_iter`'s `reduce_done()` task in
`bool_pe.csl`) costs nothing beyond running phase 1 again, no transpose or
extra redistribution step, and no host round trip.

## Trade-offs versus `original_spmv` (see `../original_spmv/README.md` first)

This design deliberately gives up the properties `original_spmv` is built
for:

- **No composability with dense-vector solver ops.** `x`/`y` are
  concentrated (redundant copies within a column / everything funneled to
  one diagonal PE per row), not a unique fragment per PE. That's fine for a
  pure boolean OR-collapse, but would leave most of the grid idle for a
  dot-product or AXPY-style update, unlike `original_spmv`'s full `P^2`
  distribution.
- **`P` times more vector memory per PE** (`blk ~= n/P` vs `original_spmv`'s
  `local_vec_sz ~= n/P^2`) — a direct consequence of concentrating instead of
  fully partitioning the vector.
- **No weights.** Only structural nonzero-ness is tracked (`mat_vals_buf` is
  gone entirely) — this also roughly halves the per-nonzero memory footprint
  versus `original_spmv`, which matters for surviving a poorly load-balanced
  matrix (see `../benchmarks/bench_notes.md`: this kernel compiled and ran on
  an unbalanced GRAPH500-style matrix that made `original_spmv`'s linker run
  out of PE memory).

In exchange: measured ~4-8x faster than `original_spmv` on every matrix/grid
combination tried so far (uniform-random, GRAPH500-style RMAT at varying
sparsity, varying grid size, balanced and unbalanced) — see
`../benchmarks/bench_notes.md` for the full set of measurements and caveats
(the two kernels solve related but not identical problems, so treat this as
"cost of this design" rather than a pure implementation bake-off).

## Requirements

- **Square PE grid** (`pcols == prows`) — asserted at compile time. The
  diagonal-target design has no meaning otherwise.
- **Square matrix** (`nrows == ncols`) — asserted by `run_bool.py`.

## Running with a different matrix / grid size

`commands_wse2.sh`/`commands_wse3.sh`/`commands_wse3_iterative.sh` are a
fixed smoke test (`../data/rmat4.4x4.lb.mtx` on a 4x4 grid) split into two
steps — an explicit `cslc` call with hand-computed `--params` (`blk`,
`max_local_nnz*`), then `run_bool.py`/`test_iterative.py --run-only` reusing
that ELF. That split only exists to avoid recompiling on repeat smoke-test
runs; the `--params` values in it are specific to that one matrix+grid
combination and won't work for any other.

For a different matrix or grid size, skip the split and call `run_bool.py`
(or `test_iterative.py`) directly, **without `--run-only`**. Both scripts run
`preprocess_bool.preprocess()` themselves before invoking `cslc`, so `blk`
and the `max_local_nnz*` sizes are computed from the actual matrix and grid
you pass — you never need to work those out by hand:

```sh
cd spmv-hypersparse   # repo-root-relative paths, same as the commands_* scripts

cs_python bool_diag_spmv/run_bool.py --arch=wse3 \
    --num_pe_cols=8 --num_pe_rows=8 --channels=1 \
    --infile_mtx=data/rmat_s6_e4.mtx \
    --latestlink bool_diag_spmv/out_s6_8x8
```

The same flags work for `test_iterative.py` (it shares `cmd_parser.py` with
`run_bool.py`):

```sh
cs_python bool_diag_spmv/test_iterative.py --arch=wse3 \
    --num_pe_cols=8 --num_pe_rows=8 --channels=1 \
    --infile_mtx=data/rmat_s6_e4.mtx \
    --latestlink bool_diag_spmv/out_s6_8x8_iter
```

Notes:

- `--num_pe_cols` **must equal** `--num_pe_rows` (square grid requirement
  above) — any square size works, it isn't required to be a power of 2 or to
  evenly divide the matrix size (`preprocess_bool.py` pads).
- `--infile_mtx` just needs to point at a square `.mtx` file. `../data/`
  already has a range of RMAT sizes to try: `rmat_s5_e4.mtx` (32x32),
  `rmat_s6_e4.mtx` (64x64), `rmat_s7_e4.mtx` (128x128), `rmat_s8_e4.mtx`
  (256x256), up to `rmat_s14_e16.mtx` (16384x16384) — see `../benchmarks/`
  for how these were generated (`gen_rmat.py`) and load-balanced.
- `--fabric-dims`/`--fabric-offsets` are optional — both scripts compute a
  large-enough fabric from the grid size and `--width-west-buf`/
  `--width-east-buf` (default 0) if you omit them.
- Drop `--latestlink` to just use the default `latest/` output dir; pass it
  explicitly (as above) if you want to keep multiple compiled variants
  around side by side instead of overwriting the previous one.
- Compilation cost scales with grid size (the 8x8/64-node example above took
  ~5s to compile vs. <1s for the 4x4 smoke test) — use `--compile-only` to
  split compilation from running if you're iterating on host-side code only,
  same as the two-step `commands_*.sh` scripts do.
- `test_iterative.py`'s `MAX_ITERS = 5` is a hardcoded constant in
  `src/bool_pe.csl` (see "Status" below) — it does not scale with matrix or
  grid size, and there's currently no flag to change it from the host.

## Status

Two entrypoints, both exported from the same compiled kernel:

- `f_spmv` — the original one-shot SpMV: bootstrap `x` at the diagonal via
  host memcpy, broadcast, local boolean multiply, reduce back to the
  diagonal, read back and verify (`run_bool.py`).
- `f_spmv_iter` — on-device iterative version: at the diagonal PEs, each
  round's raw result is masked against a cumulative `visited_buf` before
  being fed back as the next round's `x` (`reduce_done()` in `bool_pe.csl`)
  — only genuinely new discoveries propagate — for a fixed `MAX_ITERS = 5`
  rounds, no host round trip in between. Verified against 5 sequential
  host-driven `f_spmv` launches with the identical mask applied host-side
  (`test_iterative.py`) — both `visited_buf` and the last round's
  new-discoveries are bit-identical, 0 mismatches.

What's still a stub, not yet real (see the `TODO`s in `bool_pe.csl`):

- **Termination condition.** `MAX_ITERS` is a hardcoded fixed count, not a
  real "is `y` globally empty" check. `<collectives_2d>` only exposes
  broadcast/scatter/gather/`reduce_fadds` — no ready-made global AND/OR
  reduce — so a real check needs either an extra `reduce_fadds` pass over a
  single global flag or a small hand-rolled tree. The visited-mask does make
  "did this round find anything new" cheaply computable *per diagonal PE*
  now (`x_buf` all-zero after masking) — what's still missing is turning
  that into a *global* stop signal across all `P` diagonal PEs, which is a
  much smaller reduction than a whole-rectangle allreduce but still unbuilt.
- **No parent tracking.** Deliberately not attempted: `reduce_fadds` is a
  sum, not a selection, so it can't answer "which frontier member reached
  this node" once more than one predecessor fires in the same round — the
  same limitation `bfs_spmv/run_bfs.py`'s own docstring documents (point 3)
  for the host-orchestrated version, which resorts to a separate host-side
  scan against the original CSC structure instead of solving it on-device.
- **No host-settable iteration count.** `MAX_ITERS` lives only as a CSL
  `const`, not a compile or runtime param.

Also worth reconsidering before going further: whether the hypersparse
compressed-column format (inherited unchanged from `original_spmv`) is even
warranted for GRAPH500-scale sparsity — measurements in
`../benchmarks/bench_notes.md` suggest local blocks touch 28-57% of their own
column range even after load-balancing, nowhere near what that format is
optimized for.
