# bool_diag_spmv — boolean-semiring SpMV, diagonal-reduce, iterative

A from-scratch redesign of `../original_spmv` for a different target
workload: boolean-semiring SpMV (`y = OR_j (A[i,j] AND x[j])`) on a *square*
adjacency matrix, as the building block for an on-device BFS. `f_spmv_iter`
now runs the full loop entirely on-device — frontier propagation, a
cumulative visited set, real termination detection, and parent tracking,
with no host round trip and no fixed iteration cap — see "Status" below for
what's genuinely done versus what's still a stub.

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
  `f_spmv_iter` (see "Status" below): runs one `f_spmv_iter` launch (runs
  until the on-device termination relay agrees nothing new was found — no
  round cap) against the same compiled kernel's `f_spmv`, called in a
  host-side loop that applies the identical visited-mask and stops the same
  way — then checks `visited_buf`, the terminating round's raw (unmasked)
  `y_buf`, the assembled parent vector (`extract_parent_result()`, min-reduced
  host-side from the full `parent_local_buf` rectangle), and the number of
  rounds actually run are all identical between the two, plus a sanity
  invariant that the masked `x_buf` is genuinely all-zero when the device
  stops. Logs each run to `iterative_results.jsonl`.
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

### The termination relay: why 4 phases, rooted at the diagonal center

`<collectives_2d>` only gives us per-row (`mpi_x`) or per-column (`mpi_y`)
primitives — nothing addresses "the diagonal" directly, and reaching from
one diagonal PE to another always means relaying through a full row *and* a
full column (`reduce_fadds`'s own implementation is a linear chain across
the entire row/column, confirmed by reading `<collectives_2d>`'s source —
see the design notes referenced from `reduce_done()`). Detecting "did *any*
row find something new this round" therefore takes 4 sequential phases, all
in `reduce_done()`/`term_col_done()`/`term_row_done()`/
`term_row_bcast_done()`/`term_col_bcast_done()`:

1. **Column-reduce** (`mpi_y.reduce_fadds`, root `MID`): each column's
   diagonal flag sums up to row `MID` of that column.
2. **Row-reduce** (`mpi_x.reduce_fadds`, root `MID`, along row `MID`): sums
   all `P` columns' flags into one PE, `(MID, MID)`.
3. **Row-broadcast** (`mpi_x.broadcast`, root `MID`): floods the decision
   back out along row `MID`.
4. **Column-broadcast** (`mpi_y.broadcast`, root `MID`): floods it down
   every column — now every PE agrees.

`MID = pcols/2` is used as the root for *both* axes deliberately: it lands
the aggregator at `(MID, MID)` — itself a diagonal PE — and minimizes each
phase's worst-case chain latency (`max(root, P-1-root)` hops, so a center
root roughly halves the worst case versus a corner root like `0`).

Known inefficiency, left as a `TODO` in `reduce_done()`: at most one PE per
column ever has a nonzero flag (the diagonal one), but `reduce_fadds` routes
every PE's data through its own compute engine (`RAMP`) regardless, so the
other `P-1` PEs in each column pay a real (if tiny, single-element) combine
operation for nothing. A hand-rolled pass-through route could turn phase 1
into a plain send instead — deferred until profiling shows this phase, not
the fixed per-call setup/teardown overhead every phase pays regardless of
hop count, is an actual bottleneck; building it now would reintroduce
exactly the hand-rolled-routing complexity `<collectives_2d>` was adopted to
avoid.

### Parent tracking: computed pre-reduce, min-reduced host-side

`reduce_fadds` can't track *which* frontier member discovered a node either
— same problem as the termination check, sum vs. selection — but unlike the
termination flag, parent identity can't be recovered *after* the reduce at
all: by the time `y_buf` exists, every contributing PE's column identity has
already been summed away. So parent is computed in `compute()`, **before**
the reduce runs, at the only point column identity is still visible:

- Every PE `(i, j)` in row `i` covers a different slice of the column
  (source-vertex) range, so a destination row in row-block `i` can have
  different candidate parents discovered by different PEs in the same row —
  each PE only sees its own slice, so this can't be resolved locally by any
  one PE.
- Each PE keeps `parent_local_buf`, indexed identically to `y_local_buf`
  (dense over its row-block), holding the lowest-global-index local column
  it has *ever* seen touch each local row — persisted across rounds, not
  reset (unlike `y_flags`/`parent_compact`, which are per-round scratch).
  "Lowest index wins" is applied uniformly: within one round (multiple local
  columns hitting the same row), and across rounds (a later round touching a
  row again can still lower its parent if a smaller-index candidate shows
  up).
- Because different PEs in the same row hold different candidates for the
  same destination, the true parent is the **min across all `P` PEs in that
  row** — a row-reduce, structurally just like the termination check's, but
  with `min` instead of `sum`. `<collectives_2d>` has no min-reduce
  primitive, so this step is done host-side instead: `parent_local_buf` is
  exported from *every* PE (not just the diagonal, unlike `x_buf`/`y_buf`/
  `visited_buf`), and `test_iterative.py`'s `extract_parent_result()` reads
  back the full rectangle and takes `.min(axis=...)` across the row's `P`
  column-PEs after `memcpy_d2h`.
- TODO: that final cross-PE min could in principle be done on-device with a
  relay shaped exactly like the termination check's, but there's no
  min-reduce primitive to build it from without hand-rolling one — noted in
  `bool_pe.csl`'s module docstring next to the termination-relay TODO above.

One correctness note, not a bug: this is a **looser** validity notion than
`bfs_spmv/run_bfs.py`'s `find_parents()`, which only considers the frontier
*immediately preceding* a node's first discovery (guaranteeing the parent is
exactly one BFS level closer). The device has no way to enforce that —
`compute()` runs identically whether or not a row was already visited
elsewhere (masking/`visited_buf` only exist at the diagonal), so a node can
pick up a candidate from *any* round it was ever touched in, not just its
discovery round. That's still a fully valid parent by `bfs_spmv`'s own
`verify_bfs()` definition (visited + a real edge, no level check) — see
`test_iterative.py`'s `update_parent_reference()` docstring, which
deliberately mirrors this looser rule rather than `find_parents()`'s
stricter one.

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

## Status

Two entrypoints, both exported from the same compiled kernel:

- `f_spmv` — the original one-shot SpMV: bootstrap `x` at the diagonal via
  host memcpy, broadcast, local boolean multiply, reduce back to the
  diagonal, read back and verify (`run_bool.py`).
- `f_spmv_iter` — on-device iterative version: at the diagonal PEs, each
  round's raw result is masked against a cumulative `visited_buf` before
  being fed back as the next round's `x` (`reduce_done()` in `bool_pe.csl`)
  — only genuinely new discoveries propagate. After masking, a 4-phase
  relay (see "The termination relay" above) checks whether *any* row found
  something new; if not, the whole grid stops. No round-count cap anywhere
  — the loop runs for as many rounds as real BFS convergence takes (bounded
  by the node count, since visited-masking is monotonic) and no host round
  trip anywhere in between. Every PE also tracks a candidate parent
  pre-reduce (see "Parent tracking" above), min-reduced across each row's
  PEs host-side into the final parent vector. Verified against a host-side
  loop of sequential `f_spmv` launches that applies the identical mask, the
  identical stop rule, and an independent host-side parent reference
  (`test_iterative.py`) — `visited_buf`, the terminating round's raw
  (unmasked) `y_buf`, the assembled parent vector, *and* the number of
  rounds actually run are all bit-identical, 0 mismatches, on a 4x4 grid, an
  odd 5x5 grid, and an 8x8/64-node case (with the masked `x_buf`'s
  all-zero-at-stop invariant separately confirmed too).

What's still missing (see the `TODO`s in `bool_pe.csl`):

- **Phase-1 inefficiency in the termination relay.** At most one PE per
  column ever has a nonzero flag, but `reduce_fadds` can't skip the other
  `P-1` — see "The termination relay" above for the full reasoning and why
  it's deferred rather than hand-rolled now.
- **Parent's cross-PE min-reduce is host-side, not on-device.** Same root
  cause as the termination relay (`<collectives_2d>` has no min-reduce), but
  unbuilt here too — see "Parent tracking" above.

Also worth reconsidering before going further: whether the hypersparse
compressed-column format (inherited unchanged from `original_spmv`) is even
warranted for GRAPH500-scale sparsity — measurements in
`../benchmarks/bench_notes.md` suggest local blocks touch 28-57% of their own
column range even after load-balancing, nowhere near what that format is
optimized for.
