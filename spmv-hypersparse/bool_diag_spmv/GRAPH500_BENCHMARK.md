# Graph500 BFS benchmark methodology — what we're timing, and why

This is a separate document from `README.md` on purpose: it's about the
*benchmark methodology* (what counts as "the BFS time," how TEPS is
defined) rather than the kernel's own design.

Two scripts implement this methodology, at different scope:
- `run_bfs.py`: single search, single compile + launch, plus a tree plot
  and verbose per-phase breakdown -- the deep-dive tool for one root.
- `run_graph500.py`: the full spec-shaped sweep -- one compile, one matrix
  upload (construction, timed once), then N single-source searches (default
  64) from distinct random roots, each timed individually and combined via
  the harmonic mean (section 1). This is what section 6 below now reports
  against.

## 1. What Graph500 actually defines

Verified directly against the [official spec](https://graph500.org/?page_id=12)
(section 9.1/9.2) and the
[reference implementation](https://github.com/graph500/graph500) source
(`src/main.c`, `src/bfs_reference.c`), not just the prose docs.

**Two kernels, timed completely separately:**

- **Kernel 1 (construction)**: build the graph's data structure once, from
  the raw edge list. Timed (`construction_time`), but **never** part of any
  individual search's reported time.
- **Kernel 2 (BFS search)**: run from 64 different (random) roots. Each
  search is timed individually:

  ```c
  double bfs_start = MPI_Wtime();
  run_bfs(root, &pred[0]);
  double bfs_stop = MPI_Wtime();
  ```

  The timer wraps *only* `run_bfs()` — nothing before, nothing after. The
  spec states the rule directly: *"Start the time for a search immediately
  prior to visiting the search root. Stop the time for that search when
  the output has been written to memory. Do not time any I/O outside of
  the search routine."*

  Practically, that means: whatever it takes to seed the root and to get
  the resulting parent array into host-readable memory **is** part of the
  timed search (it happens fresh for every one of the 64 roots) — but the
  graph's structure, built once and reused across all 64 searches, is not.

  **The output, precisely**: the reference implementation's own signature
  is `void run_bfs(int64_t root, int64_t* pred)` (`mpi/bfs_custom.c`) — the
  predecessor/parent array (`pred[root] = root`, `pred[unreachable] = -1`)
  is the *entire* official output. There is no separate "visited" output in
  the spec at all — visited is nothing more than "has a valid predecessor
  (or is the root)". `pred` is filled in as an ordinary in-memory array
  (host RAM), never touching disk — confirming "memory" in "the output has
  been written to memory" means RAM, and "do not time any I/O outside of
  the search routine" is what excludes disk writes (e.g. this repo's own
  CSV logging, which happens after every tic/toc bracket closes, not
  inside one).

**Root selection** (section 5 of the spec, quoted verbatim): *"The search
keys must be randomly sampled from the vertices in the graph. To avoid
trivial searches, sample only from vertices that are connected to some
other vertex. Their degrees, not counting self-loops, must be at least
one. If there are fewer than 64 such vertices, run fewer than 64
searches."* So: degree >= 1 excluding self-loops, 64 **unique** (no
repeats) roots sampled uniformly at random, and simply run fewer searches
if the graph doesn't have 64 qualifying vertices. Notably, **there is no
giant-connected-component requirement** — a vertex with a single real edge
into a tiny, otherwise-isolated pair still qualifies. `run_graph500.py`'s
`pick_sources()` implements exactly this rule (see section 5 below for the
one place it initially got the self-loop exclusion wrong, now fixed).

**TEPS**: `TEPS(n) = m / bfs_time(n)`, where `m` is the number of edges in
the traversed component — each self-loop counted once, each non-self-loop
edge counted once total (not twice, even though the underlying storage
holds both directions as separate tuples). Reference implementation
(`bfs_reference.c:129-141`):

```c
for (i = 0; i < g.nlocalverts; i++)
  if (pred_glob[i] != -1)                         // i was visited
    for (j = rowstarts[i]; j < rowstarts[i+1]; j++)
      if (COLUMN(j) <= VERTEX_TO_GLOBAL(my_pe(), i))  // dedup: keep one direction
        edge_count++;
```

**Combining the 64 searches**: TEPS is a rate, so the spec uses the
**harmonic mean** (`1 / mean(time_i / edges_i)`), plus quartiles/min/max
reported alongside.

## 2. Mapping onto `bool_diag_spmv`

Here's how each measured piece maps onto the Kernel 1 / Kernel 2 split
above:

| Component | Graph500 analogue | Status |
|---|---|---|
| `h2d_matrix` (`mat_rows_buf`, `mat_col_idx/loc/len_buf`, `y_rows_init_buf`, `local_nnz*`) | **Kernel 1 (construction)** — built once, reused across searches | **Resolved in `run_graph500.py`** — uploaded exactly once per benchmark run, timed separately, excluded from every search's own time (in `run_bfs.py`, which always does one compile + one search, this is logged per-run but not part of `search_time_cycles` either) |
| `h2d_seed` (`x_buf`, the search root) | Part of Kernel 2 — "immediately prior to visiting the search root" | **In scope** |
| on-device BFS rounds (`visited_bcast`, `vertical_bcast`, `local_compute`, `reduce`, `local_term_cond`, `relay_*`, all from `ts_buf`/`record_ts()`) | Kernel 2 itself — the actual `run_bfs()` | **In scope** |
| `d2h` (`parent_local_buf` readback only) | Part of Kernel 2 — "output has been written to memory" (the spec's own output, see section 1) | **Resolved** — folded into `search_time_cycles` (section 3) |

`visited_buf` is **no longer read back at all** by either script — it's
kernel-internal state (used for masking/termination), not part of
Graph500's own output. `device_io.derive_visited_from_parent()` recovers an
identical `visited` array host-side from `parent_local_buf` alone (proven,
not approximated — see its own docstring), so dropping that transfer both
matches the spec more closely (parent is the only real output) and removes
genuinely redundant device-to-host traffic from every search.

## 3. Current "search time" definition (cycles, in scope now)

**Implemented** — both `run_bfs.py` and `run_graph500.py` log this as
`search_time_cycles`, per search:

```
search_time_cycles = h2d_seed_cycles
                    + sum over all profiled rounds r of:
                        visited_bcast_max[r] + vertical_bcast_max[r]
                        + local_compute_max[r] + reduce_max[r]
                        + local_term_cond_max[r] + relay_total_max[r]
                    + d2h_cycles   (parent_local_buf readback only)
```

Using each phase's `_max_cycles` (the straggler PE) per round, summed
across rounds — the same definition `plot_bfs_timing.py`'s stacked bars
already visualize (each round's bar height = sum of its phase segments).
`relay_total` is used directly here rather than re-summing its own 4
sub-phases, to avoid double-counting. `d2h_cycles` is the straggler-PE max
over just the `parent_local_buf` transfer (see section 2) -- smaller than
it used to be, now that `visited_buf` isn't read back at all.

`h2d_matrix` is still **excluded**, correctly -- it's Kernel 1
(construction), never part of any individual search's time.

## 4. `m` — edges traversed, for `bool_diag_spmv`'s own matrix convention

`bool_diag_spmv` stores `A` as row=dest/col=source (`A_csr[r, c] != 0`
means edge `c -> r`; see `generate_boolean_reference` in `run_single_spmv.py`).
The test matrices (`benchmarks/gen_rmat.py`) are explicitly **symmetrized**
before being written out ("symmetrize (undirected graph, standard for BFS
benchmarking)"), so `A_csr[v, u] != 0 <=> A_csr[u, v] != 0` — the same
undirected-with-both-tuples-stored shape Graph500's own reference graphs
have. That means the reference implementation's dedup rule ports directly
(implemented in `run_bfs.py`, vectorized rather than the loop form
below):

```python
m = sum(
    1
    for v in range(n) if visited[v]
    for u in A_csr.indices[A_csr.indptr[v]:A_csr.indptr[v + 1]]
    if u <= v
)
```

**Implemented** — both scripts derive `visited` from `parent_local_buf`
(via `device_io.derive_visited_from_parent()` -- see section 2) and compute
`m` as `np.sum(visited[coo.row] & (coo.col <= coo.row))` on `A_csr.tocoo()`
(`bfs_timing.compute_m_and_gteps`) — one vectorized pass, no Python-level
loop over `n`/`nnz`.

**Caveat confirmed by testing, not just theoretical**: this dedup rule is
only correct for a symmetric (undirected) `A_csr`. Running it against
`rand600.mtx` (an existing test fixture, *not* `gen_rmat.py`-generated)
produced `m=1836 > nnz/2=1800` — impossible for the real quantity, and a
clear tell that the input wasn't actually symmetric. Confirmed directly:
`rand600.mtx` has `A_csr != A_csr.T` and 3 self-loops (`gen_rmat.py`
explicitly avoids both). `run_bfs.py` now checks `A_csr` symmetry up
front and logs a `matrix_symmetric` CSV column.

To be clear about *why* symmetry matters here: it's a property of
Graph500's own edge-counting convention, not of `bool_diag_spmv`'s BFS
kernel — `y = OR_j(A[i,j] AND x[j])` is computed correctly on any square
boolean matrix, directed or not (see the module docstring's own note on
this). So a directed `--infile_mtx` isn't unsupported or invalid, it just
needs a different, still-well-defined `m`:

```python
# row=dest/col=source convention -- coo.col is the edge's SOURCE
m_directed = int(np.sum(visited[coo.col]))
```

Every visited vertex's out-edges get examined exactly once by the SpMV
kernel, in whichever round that vertex is in the active frontier (see
`compute()` in `bool_pe.csl`) — so this counts real, meaningful
algorithmic work for a directed graph, it's just **not** a Graph500-spec-
comparable number (the spec has no defined `m` for directed input at all).
Both `run_bfs.py` and `run_graph500.py` pick the formula automatically
based on `matrix_symmetric` and record which one was used in the
`m_convention` column (`"undirected_dedup"` or `"directed_source_visited"`).

## 5. Known deviations from the full Graph500 protocol (not addressed yet)

- ~~Root selection isn't restricted to the giant connected component~~ --
  **not actually a deviation**. The spec's own rule (section 1 above) only
  requires degree >= 1, not giant-component membership -- a root landing in
  a tiny, otherwise-isolated pair is spec-compliant, not a bug. On
  `data/rmat_s8_e4.mtx`, two of the 64 sampled roots did land in tiny
  (2-vertex) disconnected components, each with `m=1` and GTEPS three
  orders of magnitude below the rest -- harmonic mean is (correctly)
  extremely sensitive to this, dragging the reported harmonic-mean GTEPS
  well below the median for that run. That's exactly what harmonic mean is
  *supposed* to do with a disproportionately slow/tiny search, not a
  methodology gap to fix.
- **Fixed**: `pick_sources()` originally filtered candidates by raw
  out-degree (per-column nnz count), which -- for a matrix with self-loops
  -- could count a vertex whose only "out-edge" is a self-loop as
  qualifying. The spec explicitly excludes self-loops from the degree
  count. `gen_rmat.py`'s own output never has self-loops so this never
  showed up against it, but an arbitrary `--infile_mtx` can have them (e.g.
  `data/rand600.mtx`, 3 self-loops). Now subtracts the diagonal presence
  (`A_csc.diagonal() != 0`) from the raw out-degree before filtering.
- **Fixed**: `d2h` was previously excluded from `search_time_cycles`
  entirely, and even when it was measured, it bundled a `visited_buf`
  transfer that (per section 1's "the output, precisely") was never
  actually part of Graph500's own definition of the search's output.
  `visited_buf` is no longer read back at all (derived from
  `parent_local_buf` instead, see section 2); the resulting -- smaller,
  parent-only -- `d2h_cycles` is now folded into `search_time_cycles`
  (section 3) for every search, in both scripts.
- **Clock frequency is an assumed constant, not calibrated.** `CLOCK_FREQ_HZ
  = 875 MHz` converts `search_time_cycles` -> `search_time_seconds` for
  TEPS, but isn't calibrated against this simulator run in any way -- same
  "not an absolute hardware-calibrated figure" caveat this repo's other
  tsc-based timing already carries (see `bool_pe.csl`'s own tsc comment).
  Revisit if a real reference frequency for the simulator/hardware being
  targeted becomes available.

## 6. Current status

**Implemented, end to end**: `run_graph500.py` runs the full spec-shaped
sweep -- one compile, one matrix upload (timed once, excluded from every
search), then N single-source searches (default 64) from distinct random
roots (sampled per the spec's own degree-based rule, section 1), each
producing `search_time_cycles` (h2d_seed + on-device rounds + d2h parent
readback, section 3), `m_edges_traversed`, `m_convention`, `visited_count`,
`matrix_symmetric`, `search_time_seconds`, and `gteps` (one row per search,
appended to `graph500_searches.csv`), plus one summary row
(`graph500_summary.csv`) with `harmonic_mean_gteps` (the spec's own
aggregation rule, section 1), `min_gteps`, `median_gteps`, `max_gteps`, and
`construction_time_seconds`. Between searches, no explicit host-side reset
is needed beyond re-uploading `x_buf` -- `bool_pe.csl`'s `start_spmv()`
already reinitializes `visited_buf`/`rounds_completed`/`parent_local_buf`/
`ts_round` on every fresh `f_spmv_iter()` call (see its own comments).
`visited_buf` itself is never read back by either script -- `visited` is
derived host-side from `parent_local_buf` alone (`device_io.
derive_visited_from_parent()`, section 2), which is both more spec-faithful
(parent is Graph500's *only* defined output) and strictly less
device-to-host traffic per search. Verified against `data/rmat_s8_e4.mtx`
(256 vertices, 8x8 grid): 64/64 searches passed their per-search scipy
correctness check (`--nocorrectness` to skip, on by default), both before
and after the `d2h`/`visited_buf` change above.

`run_bfs.py` remains the single-search deep-dive tool (tree plot, verbose
per-phase breakdown, `--show-parent-mismatch`) and still logs its own
`search_time_cycles`/`gteps` for that one search into `bfs_timing.csv` --
useful for drilling into one specific root's phase breakdown, not for the
spec's own 64-search aggregate. It shares the same `d2h`/`visited_buf`
fix, so its own numbers moved too (e.g. on `data/rmat4.4x4.lb.mtx`,
`search_time_cycles` went from 18220 to 19688 cycles and `gteps` from
0.005187 to 0.004800 -- lower, but the honest, spec-complete number).

What's left is everything in section 5 above: only the clock-frequency
calibration caveat remains open; root selection and `d2h` are both
resolved.

## 7. `h2d_seed` optimization: one PE, not the whole grid

**The old approach**: `dist_x_to_diag_hwl()` built a dense `(P, P, blk)`
host array (real data only at diagonal positions, zero elsewhere) and
`memcpy_h2d` wrote the *entire* `P x P` grid, every search -- even though a
single-source seed has exactly one nonzero bit, landing in exactly one
diagonal PE's block.

**Why the other `P-1` diagonal PEs never needed that write at all**: traced
through `bool_pe.csl`'s `reduce_done()`, which sets `x_buf[i] = newly` at
every diagonal PE, every round including the last. The loop's own
termination condition (`nz_total == 0`) is a **non-negative sum** over
every diagonal PE's own `nz_local` flag (1.0 iff that PE's own `x_buf` had
any nonzero entry that round) -- a non-negative sum can only be zero if
every term is zero, so `nz_total == 0` *provably* means every diagonal
PE's entire `x_buf` is all-zero at the moment `f_spmv_iter()` returns
control to the host. Combined with `x_buf`'s zero state at kernel load,
this holds even for the very first search. Non-diagonal PEs never needed a
host write in the first place, single- or multi-source: every PE's
`x_buf` is unconditionally overwritten by that round's own column-
broadcast (`visited_bcast_done()`) before `compute()` ever reads it.

**Fixed**: `device_io.single_source_seed_pe(source, blk, P)` returns just
`(px, py, local_x)` for the one owning diagonal PE (`px = py = source //
blk`, a length-`blk` array with a single 1.0), and both `run_bfs.py` and
`run_graph500.py` now do a `1x1`-region `memcpy_h2d` instead of the
full-grid one. Confirmed via the SDK's own `sdkruntimepybind` docs and its
bundled `gemv-06-routes-1` tutorial that a non-origin `(px, py)` with
`w=1, h=1` is a normal, documented way to target exactly one PE (`px` =
column, `py` = row).

**Only valid for this single-source, `f_spmv_iter` case** -- NOT for
`run_host_driven_bfs.py`'s multi-source frontier (several diagonal PEs can
be genuinely live at once there, needs the general `dist_x_to_diag_hwl`
path) or `run_single_spmv.py`'s one-shot `f_spmv` (never touches `x_buf`
itself, so has no such self-zeroing invariant).

**Verified**: 0 correctness failures before and after, on both the 4x4/16-
vertex fixture and `data/rmat_s8_e4.mtx` (256 vertices, 8x8 grid). Cycle
savings were modest in testing (e.g. 4x4 grid: `h2d_seed` max 1485 -> 1401
cycles) -- a fixed per-transfer configure/FSM/teardown cost dominates at
these small payload sizes, the same effect noted elsewhere in this
codebase for the termination relay (see `bool_pe.csl`'s own TODO on
`reduce_fadds` overhead). Still strictly less data moved and less
host-side array construction (previously `O(P^2)`, now `O(1)`), and the
more architecturally correct amount of work regardless of whether it shows
up as a large cycle win at these particular grid sizes.

## 8. Per-PE relay cost investigation

Profiling on `data/rmat_s8_e4.mtx` (8x8 grid) showed the 4-phase
termination relay (section 3) at 46-54% of a round's total time -- over
half. Digging into *why* required per-PE data, not just the aggregate
min/max/avg `bfs_timing.csv` already logs.

**Tooling**: `run_bfs.py --dump-pe-timing` saves the full `(round, height,
width)` cycle grid per phase to a single `.npz` file (via
`bfs_timing.decode_pe_phase_cycles()`/`save_pe_phase_cycles()`), inside
`plots/heatmap/<matrix>_<grid>_src<N>/` -- one file per run, not appended
like the CSVs, since a heatmap needs the grid shape back, not a long/tidy
or wide CSV that has to be pivoted every time, and living in the same
folder as the PNGs it feeds keeps raw data and plots together as one
self-contained bundle. `plot_pe_heatmap.py` reads it back and renders
per-PE heatmaps (magma by default, `--cmap` for any other matplotlib
colormap; diagonal outlined, the relay's `(MID, MID)` aggregation point
starred) into that same folder -- one `round_<r>.png` per profiled round
(not aggregated by default: which PEs are active in a given round depends
on the graph's own structure and the chosen `--source`, not just the
protocol, so per-round is what separates those two effects) plus a
`summary_avg.png` overview (mean over rounds -- typical cost, not one
worst round). `--relay` selects just the 4-phase
termination relay's own sub-phases (`bfs_timing.RELAY_PHASES`) with a
SHARED color scale across them, for comparing their magnitudes directly
(the default per-phase-own-scale view is right for a whole-algorithm
overview, but hides exactly this comparison, which is what motivated
adding it).

**A real measurement caveat, not just a data-format one**: PEs are never
explicitly barrier-synchronized between phases. Tracing `<collectives_2d>/
pe.csl` (the actual `reduce_fadds`/`broadcast` implementation, not just
`bool_pe.csl`'s call sites): the user callback (our `term_col_done` etc.)
only fires once a `C_LOCK` task is BOTH unblocked (this PE's own FSM
reached its `Callback` state) AND activated (this PE actually received an
incoming teardown wavelet -- see `teardown_handler_0`/`_1`). The teardown
cascade itself (`teardown_reduce_network()`) is a separate, asymmetric,
multi-hop propagation from the chain's two physical endpoints inward,
converging at the root, then reflecting back out past it -- a DIFFERENT
topology than the actual data-reduce chain. So every phase's measured
"issue-to-done" interval conflates three things: real data movement,
waiting on an upstream dependency, and this teardown/re-arm handshake --
and since the relay phases carry a payload of exactly one `f32`, (1) is
negligible there, meaning almost all of the observed variance is (2)+(3),
not "real work happening at that PE." Fully separating these would require
instrumenting `<collectives_2d>` itself (a fork, for profiling purposes
only) -- not attempted; the heatmaps below are "total per-call wall-clock
cost at this PE, teardown-inclusive," not "local compute time."

**What the heatmaps actually showed** (`rmat_s8_e4.mtx`, 8x8, source=0) --
reproducible, structured patterns, not noise:
- `local_term_cond`: diagonal uniformly maxed, everywhere else uniformly
  ~0 -- exactly matches the code (only diagonal PEs run the masking loop).
  Confirms the instrumentation is trustworthy for genuine per-PE work.
- `local_compute`: worst PE data-dependent (e.g. (0,0) at 9359 cycles) --
  real partition-size imbalance, unrelated to the collective protocol.
- `visited_bcast`: the diagonal (sender/root) is fast; **every other PE in
  the grid is uniformly ~1521**, regardless of position -- a flat
  per-receiver protocol tax, not a hop-distance gradient.
- `relay_row_bcast` / `relay_col_bcast`: a sharp, clean split -- one side
  of the root (`MID`) is uniformly cheap, the other uniformly expensive,
  every round. This is `<collectives_2d>`'s `POS_DIR` (EAST/SOUTH) vs
  `NEG_DIR` (WEST/NORTH) asymmetry showing up directly: broadcasting
  toward the positive direction from the root costs meaningfully more than
  toward the negative direction.
- `relay_row_reduce`: elevated cost concentrated near columns close to
  `MID` (the root) across many different rows -- consistent with the root
  position needing extra switch-handling (receiving from both directions)
  that edge positions don't pay.
- `--relay`'s shared-scale view confirms `relay_col_reduce` is the clear
  outlier of the four relay sub-phases on this matrix/grid (avg worst-PE
  4404 cycles, vs 2943-3299 for the other three) -- consistent with
  section 3's earlier round-by-round table, now visible directly in one
  image instead of read off separate independently-scaled panels.
- `sparsity.png` (the matrix's own per-PE `local_nnz`/`local_nnz_cols`/
  `local_nnz_rows`, from `preprocess_bool.py`, saved alongside the timing
  grids for exactly this comparison) confirms the split cleanly: PE (0,0)
  is `local_nnz`'s worst PE (~180, vs a handful elsewhere) *and*
  `local_compute`'s worst PE (9359 cycles) -- real work correlating with
  real sparsity, as expected. The termination relay's own hotspots (e.g.
  `relay_col_reduce`'s worst PE at (3,0)) do **not** line up with
  `sparsity.png` at all -- confirming (independently of the protocol-level
  tracing above) that the relay's cost is structural/positional
  (`<collectives_2d>` routing), not workload-dependent.

**Status**: tooling in place and verified; the directional (`POS_DIR` vs
`NEG_DIR`) asymmetry in the two relay broadcast phases is the most
concrete, actionable lead surfaced so far, if this gets picked back up --
not yet investigated further or acted on.

## 9. Row-MID-only relay: implemented, correct, but not a latency win

Phases B/C (row-reduce, row-broadcast, both `mpi_x`, root=column `MID`) are
only ever *meaningful* for row `MID` -- every other row's own copy
combines/broadcasts data nobody reads (see section 8's per-PE heatmap
findings, which first surfaced this). Since `mpi_x` (phases B/C) and
`mpi_y` (phases A/D) use entirely separate fabric queues/colors (`{2,4}`
vs `{3,5}`, see their own instantiation in `bool_pe.csl`), row `MID` being
mid-FSM in `mpi_x` doesn't block it from also participating in `mpi_y`'s
phase D once it gets there, and a non-root `broadcast()` call genuinely
blocks on its own fabric queue waiting for real data -- it doesn't require
the root to have already called `broadcast()` first.

**Implemented** in `term_col_done()`: only `prow_id == MID` now calls
phases B/C's `mpi_x.reduce_fadds`/(via `term_row_done`/
`term_row_bcast_done`) `mpi_x.broadcast`; every other row instead records
zero-duration timestamps for those slots (so the timing tooling reports
the now-literally-true ~0 cost there, not stale leftover `ts_buf` values)
and joins phase D (`mpi_y.broadcast`, needed by everyone) directly, where
it correctly blocks until row `MID`'s real answer arrives.

**Verified correct** at three scales, 0 mismatches each time:
`data/rmat4.4x4.lb.mtx` (4x4), `data/rmat_s8_e4.mtx` (8x8),
`data/rmat_s12_e4.mtx` (16x16, n=4096).

**The honest result, from directly comparing the same matrix/grid/source
before and after** (`rmat_s12_e4.mtx`, 16x16, per-round averages):

| phase | before | after |
|---|---|---|
| `relay_row_reduce` | 4272.2, 6025.2, 4414.7, 3529.7, 3541.2 | 360.6, 522.4, 381.3, 279.6, 280.9 |
| `relay_row_bcast` | 5424.8, 7055.8, 5257.5, 4887.0, 4893.7 | 337.3, 471.8, 332.3, 249.7, 250.0 |
| `relay_col_bcast` | 5896.3, 24901.8, 8830.5, 3500.8, 3505.0 | 14900.4, 36993.5, 17794.0, 11393.2, 11414.1 |
| **`relay_total`** | 24707.1, 69155.5, 30043.3, 18669.2, 18695.7 | 24712.1, 69160.5, 30048.3, 18674.2, 18700.7 |

`relay_row_reduce`/`relay_row_bcast` collapsed (~10-20x) exactly as
expected -- 15 of 16 rows now genuinely do ~0 there. But `relay_col_bcast`
absorbed almost exactly that difference (the skipped rows arrive at phase
D early and simply wait longer there instead), and `relay_total` is
unchanged within noise (~5 cycles) at every scale tested (4x4/8x8/16x16).

**Why**: row `MID`'s own phase B/C work runs entirely on its own physical
row wires, independent of every other row (that's what "separate fabric
queues/colors" means physically, not just logically) -- so it was never
waiting on anyone else, and removing everyone else's wasted work can't
shorten a critical path it was never part of. The round's latency is, and
was always, bounded by row `MID` alone.

**What this change actually buys**: real wasted computation eliminated on
15/16 (or 3/4, at 4x4) rows every round -- less real work done on
hardware that was previously computing and transmitting values nobody
ever reads, which matters for power/resource usage even though it doesn't
show up as round-latency here. It does **not** address the directional
(`POS_DIR`/`NEG_DIR`) asymmetry from section 8, which remains the more
promising lead for an actual latency reduction, since `relay_total` is
still dominated by row `MID`'s own real work being paid twice (once in the
expensive direction for reduce, again for broadcast) rather than by
wasted work on other rows.
