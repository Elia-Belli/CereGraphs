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
- `device_io.py` — shared host<->device data-marshaling helpers (hwl<->1d
  layout conversions, diagonal/parent result extraction, the `cslc`
  invocation) used by every driver script below — not runnable on its own.
- `bfs_timing.py` — shared per-round timing constants/decoders (the
  `TS_*`/`PHASES` slot map matching `bool_pe.csl`'s `record_ts()`, the
  `read_tic_toc_delta()`/`decode_round_timestamps()` helpers) — single
  source of truth for `run_bfs.py` (writes `bfs_timing.csv`) and
  `plot_bfs_timing.py` (reads it back), not runnable on its own.
- `bfs_tree_plot.py` — shared BFS-tree-comparison rendering (digraph
  construction, the radial BFS-level layout, panel drawing, parent
  validity checking) — used by `run_bfs.py`, not runnable on its own.
- `run_single_spmv.py` (formerly `run_bool.py`) — host driver for the base,
  one-shot `f_spmv` entrypoint: seeds `x` only at the PE grid's diagonal,
  launches `f_spmv` once, reads back the full rectangle, keeps only the
  diagonal entries, verifies against an independent scipy boolean
  reference. A foundational sanity check independent of any BFS-specific
  machinery (masking, termination, parent tracking) — see "run_bfs.py vs.
  the two test scripts" below for how this relates to `run_bfs.py`.
- `run_host_driven_bfs.py` (formerly `test_iterative.py`) — regression test
  for the on-device iterative entrypoint `f_spmv_iter` (see "Status"
  below): runs one `f_spmv_iter` launch (runs until the on-device
  termination relay agrees nothing new was found — no round cap) against
  the same compiled kernel's `f_spmv`, called in a host-side loop that
  applies the identical visited-mask and stops the same way, seeded with a
  random ~50%-density *multi-source* frontier specifically to stress-test
  masking across many simultaneous discoveries at once — then checks
  `visited_buf`, the terminating round's raw (unmasked) `y_buf`, the
  assembled parent vector (`extract_parent_result()`, min-reduced
  host-side from the full `parent_local_buf` rectangle), and the number of
  rounds actually run are all identical between the two, plus a sanity
  invariant that the masked `x_buf` is genuinely all-zero when the device
  stops. Logs each run to `iterative_results.jsonl`.
- `run_bfs.py` — the main, user-facing driver: a *single-source* BFS, one
  compile + one `f_spmv_iter` launch, reporting three things from that one
  run (each independently toggleable, all on by default): a tree
  comparison plot against `scipy.sparse.csgraph.breadth_first_order`
  (`--notree` to skip), the same scipy cross-check printed as numbers
  (`--nocorrectness` to skip), and per-round phase timing + a Graph500-style
  GTEPS estimate appended to `bfs_timing.csv` plus its own bar-chart plot
  (`--notimings` to skip, which also skips the tsc instrumentation itself
  and its real transfer-time cost). See "`run_bfs.py` vs. the two test
  scripts" below for why this is a separate thing from
  `run_host_driven_bfs.py`, and `GRAPH500_BENCHMARK.md` for the GTEPS
  methodology.
- `plot_bfs_timing.py` — the per-round stacked-bar timing chart
  `run_bfs.py` calls automatically; also runnable standalone
  (`plot_timing_row()`) to re-plot an existing `bfs_timing.csv` row without
  re-running the device.
- `commands_wse2.sh` / `commands_wse3.sh` — one-shot compile+run smoke test on
  `../data/rmat4.4x4.lb.mtx` at a 4x4 grid, for WSE-2 and WSE-3 respectively.
  Unlike `original_spmv`/`bfs_spmv`, both scripts compile the *same* `src/` —
  `<collectives_2d>` abstracts the fabric-routing differences between the two
  architectures, so there's no separate `src_wse3/` tree here.
- `commands_wse3_iterative.sh` — same compile as `commands_wse3.sh`, but runs
  `run_host_driven_bfs.py` instead of `run_single_spmv.py`.

### `run_bfs.py` vs. the two test scripts

`run_single_spmv.py` and `run_host_driven_bfs.py` are correctness tests for
two different *layers*, not redundant with each other or with `run_bfs.py`:
`run_single_spmv.py` validates the base one-shot `f_spmv` primitive
(broadcast + local multiply + reduce, no BFS semantics at all);
`run_host_driven_bfs.py` validates the iterative machinery built on top of
it (masking, termination, parent tracking) under a deliberately adversarial
multi-source stress frontier, and depends on `f_spmv` already being known
correct. Neither does a real single-source BFS run. `run_bfs.py` is the
separate, user-facing "run an actual BFS and show me the result" tool —
its own `--nocorrectness` check is a single-source scipy cross-check, not
`run_host_driven_bfs.py`'s stress test, which stays its own script rather
than being folded in.

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
- A candidate is only ever recorded for a row that **isn't visited yet**
  (`visited_buf[dense_idx] == 0.0`, checked in `compute()`). Since no PE can
  ever see a hit on a row before that row's true global discovery round (a
  hit at any PE immediately makes the row visited, grid-wide, by the end of
  that same round, via the row-reduce all PEs share), this restricts
  candidate recording to *exactly* a row's discovery round — giving a
  genuine, textbook one-hop-closer BFS parent, not merely a valid-but-
  arbitrary predecessor (see "One correctness note" below for the bug this
  fixes). This needs `visited_buf` — otherwise meaningful only at the
  diagonal PE — distributed to every PE in the row before `compute()` runs
  each round: a new broadcast phase (`visited_bcast_done()`, `mpi_x`,
  rooted at each row's own diagonal, same pattern `start_spmv()`'s `x_buf`
  column broadcast already uses) inserted right after the termination
  relay decides to continue (`term_col_bcast_done()`) and right after
  `visited_buf` is first seeded (`start_spmv()`), before `compute()`'s
  first round.
- Each PE keeps `parent_local_buf`, indexed identically to `y_local_buf`
  (dense over its row-block), holding the lowest-global-index local column
  it has recorded for each local row — persisted across rounds, not reset
  (unlike `y_flags`/`parent_compact`, which are per-round scratch). "Lowest
  index wins" is applied within one round only now (multiple local columns
  hitting the same row in that row's own discovery round) — the visited
  gate above means there's nothing left to compare across rounds; once a
  row is visited, no further candidate is ever recorded for it.
- Because different PEs in the same row hold different candidates for the
  same destination, the true parent is the **min across all `P` PEs in that
  row** — a row-reduce, structurally just like the termination check's, but
  with `min` instead of `sum`. `<collectives_2d>` has no min-reduce
  primitive, so this step is done host-side instead: `parent_local_buf` is
  exported from *every* PE (not just the diagonal, unlike `x_buf`/`y_buf`/
  `visited_buf`), and `run_host_driven_bfs.py`'s `extract_parent_result()` reads
  back the full rectangle and takes `.min(axis=...)` across the row's `P`
  column-PEs after `memcpy_d2h`.
- TODO: that final cross-PE min could in principle be done on-device with a
  relay shaped exactly like the termination check's, but there's no
  min-reduce primitive to build it from without hand-rolling one — noted in
  `bool_pe.csl`'s module docstring next to the termination-relay TODO above.

One correctness note, since this was a real bug at one point: without the
visited gate above, `compute()` has no way to tell whether a row it's
touching this round was already visited many rounds ago — a much later
round's frontier member with a real structural edge into an
already-discovered row would silently overwrite its parent whenever that
later member's global index happened to be lower, producing an invalid
(same-level, or even more-hops-away) "parent" that plainly wasn't one BFS
level closer, however plausible it looked as *a* valid predecessor. The
device-side fix (`visited_buf` gate above) plus the matching host-side fix
in `run_host_driven_bfs.py`'s `update_parent_reference()` (gated on the host's
own `visited` array the same way) now give a parent that matches
`bfs_spmv/run_bfs.py`'s stricter `find_parents()` definition (exactly one
BFS level closer), not merely `verify_bfs()`'s looser one (visited + a real
edge, no level check) — confirmed by `run_bfs.py`, whose
`--show-parent-mismatch` tie-break-difference count against scipy's own
`breadth_first_order` dropped to the residual cases where multiple
one-hop-closer predecessors are equally valid and scipy's FIFO-order
tie-break picks a different one than our lowest-index rule.

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
- **Square matrix** (`nrows == ncols`) — asserted by `run_single_spmv.py`.

## Running with a different matrix / grid size

`commands_wse2.sh`/`commands_wse3.sh`/`commands_wse3_iterative.sh` are a
fixed smoke test (`../data/rmat4.4x4.lb.mtx` on a 4x4 grid) split into two
steps — an explicit `cslc` call with hand-computed `--params` (`blk`,
`max_local_nnz*`), then `run_single_spmv.py`/`run_host_driven_bfs.py --run-only` reusing
that ELF. That split only exists to avoid recompiling on repeat smoke-test
runs; the `--params` values in it are specific to that one matrix+grid
combination and won't work for any other.

For a different matrix or grid size, skip the split and call `run_single_spmv.py`
(or `run_host_driven_bfs.py`) directly, **without `--run-only`**. Both scripts run
`preprocess_bool.preprocess()` themselves before invoking `cslc`, so `blk`
and the `max_local_nnz*` sizes are computed from the actual matrix and grid
you pass — you never need to work those out by hand:

```sh
cd spmv-hypersparse   # repo-root-relative paths, same as the commands_* scripts

cs_python bool_diag_spmv/run_single_spmv.py --arch=wse3 \
    --num_pe_cols=8 --num_pe_rows=8 --channels=1 \
    --infile_mtx=data/rmat_s6_e4.mtx \
    --latestlink bool_diag_spmv/out_s6_8x8
```

The same flags work for `run_host_driven_bfs.py` (it shares `cmd_parser.py` with
`run_single_spmv.py`):

```sh
cs_python bool_diag_spmv/run_host_driven_bfs.py --arch=wse3 \
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
  diagonal, read back and verify (`run_single_spmv.py`).
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
  (`run_host_driven_bfs.py`) — `visited_buf`, the terminating round's raw
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
