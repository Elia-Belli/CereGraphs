# bool_diag_spmv — boolean-semiring SpMV, diagonal-reduce, one iteration

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
  launches, reads back the full rectangle, keeps only the diagonal entries,
  verifies against an independent scipy boolean reference.
- `commands_wse2.sh` — one-shot compile+run smoke test on
  `../data/rmat4.4x4.lb.mtx` at a 4x4 grid.

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

Targeting the diagonal specifically (not a corner) is what makes a future
BFS loop cheap: the PE that produces row `py`'s final result is exactly the
PE that needs to *source* the next iteration's broadcast down column `py` —
reusing `y` as the next iteration's `x` costs nothing beyond running phase 1
again, no transpose or extra redistribution step.

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

## Status

One SpMV iteration only: bootstrap `x` at the diagonal via host memcpy,
broadcast, local boolean multiply, reduce back to the diagonal, read back and
verify. **BFS looping (masking the diagonal's result and feeding it back as
the next iteration's seed, termination condition, multi-iteration host loop)
is not implemented yet** — by construction (see "why the diagonal" above) it
should require no new communication primitives, just wiring up the loop and
a per-diagonal-PE visited-bitmap mask, but this hasn't been built or tested.
Also worth reconsidering before going further: whether the hypersparse
compressed-column format (inherited unchanged from `original_spmv`) is even
warranted for GRAPH500-scale sparsity — measurements in
`../benchmarks/bench_notes.md` suggest local blocks touch 28-57% of their own
column range even after load-balancing, nowhere near what that format is
optimized for.
