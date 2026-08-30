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

**Superseded by section 14** -- this section described a per-phase-sum
formula (`visited_bcast`/`vertical_bcast`/`reduce`/`relay_total` each
individually timed and summed) that predates, then briefly coexisted with,
then was fully replaced by section 14's `device_time_cycles` = whole-run
`round_trip_start_buffer` → `round_trip_done_buffer` span. Kept here for
historical context (this is what section 4 onward originally cross-
referenced); do not implement against this formula.

`h2d_matrix` is (and remains) **excluded** from `search_time_cycles` --
it's Kernel 1 (construction), never part of any individual search's time.

## 4. `m` — edges traversed, for `bool_diag_spmv`'s own matrix convention

`bool_diag_spmv` stores `A` as row=dest/col=source (`A_csr[r, c] != 0`
means edge `c -> r`; see `generate_boolean_reference` in `run_single_spmv.py`).
The test matrices (`datasets/gen_rmat.py`) are explicitly **symmetrized**
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

## 10. Skew-adjusted heatmaps: how much of the relay's cost is real?

**Superseded by section 12**: this section's `entry_ref` (a flat max over
the *whole* row/column group) turned out to be unsound for `reduce_fadds`-
based phases -- confirmed producing large, impossible negative "adjusted"
values on real `rmat_s12_e4.mtx` data. Section 12 replaces it with a
hop-aware chain reference and, with that fix, `relay_col_bcast` -- reported
here as ~80% pure wait -- turns out to be **~100% real cost**. The
methodology description and `visited_bcast`/`vertical_bcast`/`reduce`
findings below are still broadly accurate; the `relay_col_bcast` verdict
specifically is not -- see section 12 before relying on any specific
number from this section.

Before committing to a hand-rolled diagonal allreduce (the next step being
considered to replace `<collectives_2d>` entirely), it was worth checking
whether the relay's measured cost is real. PEs are **not** explicitly
synchronized at phase boundaries: `record_ts()` captures each PE's own
task-entry/issue point independently, so a phase's raw duration for a PE
that has nothing real to contribute (e.g. a non-diagonal PE before
`relay_col_reduce`, or a non-`MID` row before `relay_row_reduce`/
`relay_row_bcast`) can be almost entirely *wait* for whichever PE in its
dependency group (one row for an `mpi_x` call, one column for an `mpi_y`
call) actually has real work, not real fabric transit.

**Method** (`bfs_timing.compute_skew_adjusted()`, `PHASE_GROUP_AXIS`): with
no independent global clock, there's no way to directly timestamp "when did
this phase really start" -- but every PE in a dependency group already
recorded its own issue timestamp for the phase, and the WSE is one
synchronous clock domain across the whole wafer (already relied on
implicitly every time this investigation has compared timestamps across
different PEs), so those ARE directly comparable. The group's real start is
therefore the MAX of its members' own issue timestamps -- whichever PE was
slowest to even become ready:

```
wait[pe]     = group_max_issue - own_issue[pe]        (>= 0 always)
adjusted[pe] = own_done[pe] - group_max_issue
raw[pe]      = own_done[pe] - own_issue[pe] = wait[pe] + adjusted[pe]
```

Applied to all 7 phases that bracket a real collective call
(`visited_bcast`, `vertical_bcast`, `reduce`, and all four relay phases).
`local_compute` and `local_term_cond` are deliberately excluded: neither
brackets a collective, and their own per-PE variance is genuine local work
(sparsity-dependent multiply, or a diagonal-only masking loop), not wait.
`relay_row_reduce`/`relay_row_bcast` use row `MID`'s own P columns as the
group (post section-9 optimization, that's the only row where this phase is
real) -- this surfaces a skew source section 8/9 never looked at: different
**columns'** `relay_col_reduce` finishing at different real times before
reaching row `MID`.

Also computed: a single honest end-to-end number, independent of the
per-phase split -- `relay_critical_path_cycles` = latest anyone finishes
`relay_col_bcast` anywhere, minus the earliest a real diagonal PE was ready
to even start the relay (only diagonal PEs feed anything real into
`relay_col_reduce`).

**Verified** (`raw == wait + adjusted` asserted exactly, integer cycle
counts) at 4x4 (`rmat4.4x4.lb.mtx`): identity holds for every phase/PE/
round, and the two previously-known zero-wait cases come out exactly 0
(diagonal PE's own `relay_col_reduce_wait`, row `MID`'s own
`relay_col_bcast_wait`).

**Confirmation run**: 16x16 grid, `data/rmat_s6_e4.mtx` (source 0, chosen
over the larger `rmat_s12_e4.mtx` used in section 9 to keep this a cheap
rerun rather than another ~12-minute run -- same PE grid, much smaller
matrix), 0 mismatches. Mean raw duration vs. mean wait, averaged over every
PE and all 3 profiled rounds:

| phase | raw (avg cycles) | wait | adjusted | wait as % of raw |
|---|---|---|---|---|
| `visited_bcast` | 418.8 | 66.9 | 351.9 | 16.0% |
| `vertical_bcast` | 364.7 | 19.8 | 345.0 | 5.4% |
| `reduce` | 829.5 | 288.1 | 541.4 | 34.7% |
| `relay_col_reduce` | 829.8 | 320.2 | 509.6 | 38.6% |
| `relay_row_reduce` | 86.8 | 13.6 | 73.2 | 15.6% |
| `relay_row_bcast` | 66.6 | 12.7 | 53.9 | 19.1% |
| `relay_col_bcast` | 1789.1 | **1422.6** | 366.5 | **79.5%** |

And the headline comparison, per round:

| | round 0 | round 1 | round 2 |
|---|---|---|---|
| `relay_total` avg (previously reported) | 2822.6 | 2763.2 | 2731.0 |
| `relay_total` max (previously reported, the "worst PE") | 3164 | 3156 | 3065 |
| `relay_critical_path_cycles` (this section, skew excluded) | 2787 | 2815 | 2739 |

**Findings**:
- `relay_col_bcast` -- the single most expensive relay phase in every prior
  heatmap -- is **~80% pure wait**. The heatmap (`skew_relay_col_bcast_
  summary_avg.png` in `plots/heatmap/rmat_s6_e4_16x16_src0_skew/`) shows
  `wait` visually near-identical to `raw`, and `adjusted` (real transit)
  uniformly dark almost everywhere. This confirms the user's suspicion
  directly: most of what looked like an expensive final broadcast was
  actually PEs sitting idle waiting for row `MID`, not real fabric cost.
- `relay_col_reduce` and the SpMV `reduce` are the opposite case: real,
  substantial cost even after removing skew (61%/65% of raw survives as
  `adjusted`). These phases are NOT overstated by skew -- they're genuinely
  expensive.
- `relay_row_reduce`/`relay_row_bcast` (row `MID`'s internal cross-column
  skew, not analyzed before this section) carry a smaller but real ~15-19%
  wait fraction -- visible as a horizontal band in `skew_relay_row_reduce_
  summary_avg.png`, distinct column-to-column, not previously visible
  because these phases' timing was only ever looked at relative to their
  own row (which is trivially near-zero for every non-`MID` row).
- The honest end-to-end number, `relay_critical_path_cycles` (2739-2815),
  sits **below** `relay_total`'s previously-reported max (3065-3164, used
  throughout sections 8/9 as "the worst PE") and close to its avg
  (2731-2823) -- i.e. the raw *average* across PEs was already closer to
  the truth than the raw *max* the heatmap-driven analysis leaned on; the
  "worst PE" framing was itself partly a skew artifact.

**Bottom line**: the relay is genuinely still a real cost (`relay_col_reduce`
and the SpMV `reduce` don't go away), but it is **not as bad as the raw
per-PE heatmaps suggested** -- a large fraction of `relay_col_bcast`'s
apparent cost was PEs waiting on row `MID`, not fabric transit. This
tempers, but doesn't eliminate, the case for the diagonal-allreduce
rewrite: it's still worth doing for `relay_col_reduce`/`reduce`'s real cost,
but shouldn't be scoped as if `relay_col_bcast` were an equally real,
independent target.

**Tooling**: `plot_pe_heatmap.py --skew <phase>` (any key in
`bfs_timing.PHASE_GROUP_AXIS`) renders `[raw, wait, adjusted]` side by side,
shared scale, for one phase.

**Scope note**: this pass only touched the per-PE heatmap path
(`bfs_timing.py`/`plot_pe_heatmap.py`), per the heatmap being easy to verify
by eye. The same generalized skew split cascading into the aggregate
bar-plot/CSV path is section 11, below.

## 11. Cascading the skew fix into the bar plots / CSV / GTEPS

Section 10 only changed `plot_pe_heatmap.py`. The same raw-vs-skew problem
applies identically to `decode_phase_row()` (the aggregate min/max/avg
`bfs_timing.py` logs to `bfs_timing.csv`/`graph500_searches.csv`) and its
consumer `plot_bfs_timing.py` (the per-round stacked bar chart) -- both
were built entirely from each phase's raw straggler-PE max, which section
10 showed can be up to ~80% artificial wait for some phases.

**`decode_phase_row()` change**: now calls `decode_pe_phase_cycles()` +
`compute_skew_adjusted()` internally (previously duplicated its own
timestamp-decoding loop) and logs three new columns per phase that has an
adjustment (`{phase}_adjusted_min/max/avg_cycles`, all 7 phases in
`PHASE_GROUP_AXIS`) alongside the existing raw ones -- nothing removed, so
existing raw columns are unchanged. A new `relay_critical_path_cycles`
column logs the same honest end-to-end scalar from section 10.
`device_time_cycles` (feeds `search_time_cycles` -> `search_time_seconds`
-> `gteps`) now sums each `SEARCH_TIME_PHASES` entry's **adjusted** max
where one exists (`visited_bcast`, `vertical_bcast`, `reduce`), the raw max
unchanged for `local_compute`/`local_term_cond` (real work, nothing to
adjust), and `relay_critical_path_cycles` in place of `relay_total`'s own
raw max (`relay_total` itself has no single group axis -- it spans two
different collectives' groups -- so this is its honest substitute, exactly
as used for the headline number in section 10).

**`plot_bfs_timing.py` change**: every stacked-bar segment now uses each
phase's `{phase}_adjusted_max/min_cycles` columns by default (segment
height / tick), falling back to the raw columns for `local_compute`/
`local_term_cond` and for any CSV row logged before these columns existed
-- same "skew-adjusted by default, raw still available" convention as
`plot_pe_heatmap.py`. Suptitle now says so explicitly.

**Verified**: fast 4x4 rerun (`rmat4.4x4.lb.mtx`, scratch `--csv` path, not
the real `bfs_timing.csv`), 0 mismatches. `search_time_cycles` dropped from
19618 to 17962 cycles (GTEPS 0.004817 -> 0.005261) purely from swapping
`relay_total`'s raw max for `relay_critical_path_cycles` in the same
formula -- i.e. the previously-reported GTEPS numbers were understated by
exactly the same relay wait-skew section 10 found, not a new effect.
Re-plotted an existing (pre-this-change) row from the real `bfs_timing.csv`
to confirm the fallback path works unchanged for old data.

**Migration**: `bfs_timing.csv`/`graph500_searches.csv`/`graph500_summary.csv`
each assert their existing on-disk header matches the current run's columns
before appending (by design -- see the assertion's own message,
"delete/rename the old CSV... or pass a different --csv path"). The new
columns above change that schema, so the three old files (11 / 136 / 6
rows) were renamed to `bfs_timing_legacy.csv`/`graph500_searches_legacy.csv`/
`graph500_summary_legacy.csv` -- old data preserved, not lost. The next
`run_bfs.py`/`run_graph500.py` invocation creates a fresh
`bfs_timing.csv`/`graph500_searches.csv`/`graph500_summary.csv` with the new
schema automatically (`write_header = not os.path.exists(csv_path)`).

## 12. Chain-aware skew reference: fixing a real bug, and the honest s12 verdict

**Further superseded by section 13**: even this section's own fix turned
out to be incomplete -- it chained off each chain position's *issue* time
rather than its *done* time, which still let one unrelated straggler's
irrelevant issue-time delay bleed through every position after it. A
second fix (chaining off `done`/`end_grid`, not `start_grid`) resolved the
concrete symptoms this caused (relay phases moving a fixed 1-float payload
now correctly showed *constant* cost across rounds; `reduce`'s heatmap
stopped looking like only the diagonal PE had real work) -- **but the
result of that second fix was, on inspection, judged implausible too**
(most PEs collapsing to near-zero "adjusted" cost), and neither of us could
fully verify it was correct rather than merely no-longer-*obviously*-wrong.
Given two rounds of real bugs found this way, section 13 sets the entire
per-phase/per-PE skew decomposition aside as unverified/exploratory rather
than continuing to patch it -- treat every specific number in sections
10-12 (not just section 10's) as provisional, not authoritative.

Section 10's `entry_ref` -- a flat max over a phase's *whole* row/column
group -- was found to be unsound while investigating a concrete bug report:
skew-adjusted heatmaps on `rmat_s12_e4.mtx` (16x16) showed **large, negative
`_adjusted` values** for `reduce`, `relay_col_reduce`, and `relay_row_reduce`
(up to 48% of cells, magnitudes in the tens of thousands of cycles) --
physically impossible for a "real cost" number.

**Root cause** (confirmed by reading `collectives_2d/pe.csl`'s actual
`transfer_data_reduce()`): a non-root `reduce_fadds` participant's own
"done" callback fires as soon as **its own single hop** completes (receive
from one neighbor, add, forward to the next) -- it does not wait for the
whole collective to finish everywhere. The reduce (and, by the same
`configure_broadcast_network()` routing logic, the broadcast) is a genuine
**two-sided sequential chain**: one side flows `0 -> 1 -> ... -> root`, one
hop at a time, the other flows `(NUM_PES-1) -> ... -> root+1`, and a PE at
chain position `k` only ever causally depends on positions between `k` and
its own side's far edge -- never the other side of root, and never a
position further toward root than itself. A flat group-wide max conflates
both sides (and every position within a side), so a straggler anywhere in
the group can push the reference **past** some other, unrelated PE's own
real completion -- which is exactly what produced the negative values.

**Fix** (`bfs_timing._chain_entry_ref()`): a running (cumulative) max
computed from each side's own far edge inward toward root
(`np.maximum.accumulate`), not a flat max over the whole group or even a
whole side. Every phase now also has an explicit root position
(`PHASE_ROOT_KIND`: `"diagonal"` for `visited_bcast`/`vertical_bcast`/
`reduce`, whose root varies per row/column; `"mid"` for the four relay
phases, whose root is the fixed `MID`). Verified on both the fast
`rmat_s6_e4.mtx` rerun and the real `rmat_s12_e4.mtx` data: **zero**
negative `wait`/`adjusted` values anywhere, identity (`raw == wait +
adjusted`) still holds exactly.

**Round-boundary fix, same pass**: round r's `visited_bcast` issue time is,
for every PE, essentially identical to that same PE's own `relay_col_bcast`
completion from round r-1 (`term_col_bcast_done()` immediately re-issues
the next broadcast, no real work in between). All of `visited_bcast`'s
cross-PE variance at round r>=1 is therefore provably already-reported
`relay_col_bcast` variance from the previous round, not anything new --
giving it a fresh reference would either duplicate that skew or (a literal
backward-pointing reference) produce negative wait. Round r>=1 now reports
`wait=0`/`adjusted=raw` for `visited_bcast` (its whole raw duration is
genuinely new cost); round 0 (no preceding round) keeps the normal chain
reference.

Also persisted: `decode_pe_phase_cycles()`'s raw per-slot timestamps are now
saved into the `.npz` too (prefixed `raw_ts_*`) -- previously discarded
after computing phase diffs, which meant this whole investigation could
only be re-verified by re-running the device. Any future change to the
skew methodology can now be re-checked against already-captured data.

### The honest `rmat_s12_e4.mtx` (16x16) verdict

Rerun end-to-end with the fixed pipeline (`plots/heatmap/
rmat_s12_e4_16x16_src0_v2/`, 0 mismatches). Per-phase average wait
fraction (mean over all PEs and all 5 profiled rounds):

| phase | raw (avg cycles) | wait | adjusted | wait as % of raw |
|---|---|---|---|---|
| `visited_bcast` | 2259.8 | 0.0 | 2259.8 | 0.0% |
| `vertical_bcast` | 623.5 | 16.8 | 606.7 | 2.7% |
| `reduce` | 6810.9 | 3260.4 | 3550.5 | 47.9% |
| `relay_col_reduce` | 13066.9 | 10475.0 | 2591.9 | **80.2%** |
| `relay_row_reduce` | 365.0 | 208.6 | 156.4 | 57.1% |
| `relay_row_bcast` | 328.2 | 0.9 | 327.3 | 0.3% |
| `relay_col_bcast` | 18499.0 | 9.5 | 18489.5 | **0.1%** |

**This overturns section 10's headline finding**: `relay_col_bcast` --
reported there as ~80% pure wait, the single biggest "inflation" -- is
under the corrected model **~100% real cost**
(`skew_relay_col_reduce_relay_col_bcast/summary_avg.png` shows this
directly: `relay_col_bcast`'s `wait` panel is essentially all-black,
`adjusted` visually identical to `raw`). The phases that genuinely *are*
wait-dominated are different ones: `relay_col_reduce` (80.2%) and the plain
SpMV `reduce` (47.9%) -- both `reduce_fadds`-based, both showing the same
real skew (mostly from R-MAT's own highly-skewed per-PE `local_compute`
load feeding into when each PE becomes ready to issue its own reduce call).

**Device-time share, old (raw) vs new (skew-adjusted)**:

| phase | old share | new share |
|---|---|---|
| `local_compute` | 36.6% | **40.8%** |
| `relay` (`relay_total` raw / `relay_critical_path_cycles`) | 33.2% | **26.3%** |
| `reduce` | 20.9% | 22.4% |
| `local_term_cond` | 7.0% | 7.8% |
| `visited_bcast` | 1.9% | 2.1% |
| `vertical_bcast` | 0.5% | 0.5% |

`relay_total`'s own raw max, summed over 5 rounds, is 213191 cycles;
`relay_critical_path_cycles` (the skew-free equivalent) sums to 151802 --
a **28.8% reduction**. `search_time_cycles` drops from 791264 to 724777
cycles purely from this correction; GTEPS rises from 0.01585 to **0.01730**
(+9.2%), with no device or algorithm change at all.

**Bottom line**: the relay genuinely was inflated by measurement artifact,
but not uniformly, and not enough to call it a non-issue. It drops from the
single largest device-time component to the second-largest, behind
`local_compute` (which was always real, data-dependent work and is
untouched by any of this). Within the relay, the honest picture is now
sharper than section 10's: `relay_col_reduce` and the SpMV `reduce` are the
real targets for a diagonal-allreduce rewrite (80.2% / 47.9% wait --
genuine, fixable inefficiency); `relay_col_bcast`/`relay_row_bcast` are not
(~0% wait -- that time is real fabric transit a redesign wouldn't
eliminate, though it might still be reshaped by a different topology).

**Artifacts**: `plots/heatmap/rmat_s12_e4_16x16_src0_v2/` (fresh, corrected
data) is kept alongside the original (now-superseded, pre-chain-fix)
`plots/heatmap/rmat_s12_e4_16x16_src0/` and the pre-optimization
`..._before/` baseline -- nothing deleted, all three stages of this
investigation remain directly comparable.

## 13. Setting per-phase skew decomposition aside; a robust alternative

Section 12's own fix turned out to be incomplete (see that section's own
"further superseded" note): chaining off each chain position's *issue*
time still let an unrelated PE's own irrelevant delay bleed forward
through every later position on the same side. A second fix -- chaining
off each predecessor's actual *done* time instead -- resolved the two
concrete, checkable symptoms this caused: `relay_col_reduce`/
`relay_row_reduce` (which move a fixed 1-float payload every round,
regardless of data) started reporting *exactly constant* adjusted cost
across rounds instead of wildly varying ones, and `reduce`'s per-PE
heatmap stopped collapsing to "only the diagonal PE has real work."

But the resulting heatmap -- most interior PEs collapsing to near-zero
"adjusted" cost, with only the two chain endpoints and the root showing
anything -- was, on inspection, judged implausible too, and neither
further reasoning nor the round-invariance check could fully confirm or
rule this out. Two rounds of real, checkable bugs found this way (both by
inspection, not by the model's own internal checks) is a strong signal
that decomposing a multi-hop collective's per-PE timestamp into "real
cost" vs "cross-PE wait" is a genuinely hard problem -- possibly requiring
full per-hop chain-position modeling (not just a two-sided split) to get
right, which is its own significant undertaking. **Set aside, not solved**
at the time this section was written: `compute_skew_adjusted()`/
`PHASE_GROUP_AXIS`/`plot_pe_heatmap.py --skew` were kept in the codebase
for anyone who wanted to keep pursuing this, and none of sections 10-12's
specific numbers were treated as reliable, with the relay-vs-`local_compute`
"bottleneck" verdict from section 12 retracted pending a trustworthy
decomposition. **Update (section 14): this machinery has since been fully
removed from the codebase, not just set aside** -- `device_time_cycles`
(used for GTEPS everywhere) had never actually adopted this section's own
"robust alternative" below despite predating it; that inconsistency, plus
the judgment that the skew-adjustment math itself was unreliable, is what
section 14 resolves.

### A robust alternative: round-duration + local-only decomposition

Rather than decompose communication phases at all, use only measurements
that don't require any cross-PE reference-point assumption in the first
place -- two straggler-PE spans, both bracketed by hard synchronization
facts already established earlier in this investigation:

- **`total_runtime_cycles`**: `max` over every PE of (last profiled
  round's `TS_TERM_COL_BCAST_DONE` − round 0's `TS_VBCAST_ISSUE`) -- the
  straggler's own first-phase-to-last-phase span. Doesn't account for the
  very first broadcast's one-time fabric fill delay -- acceptable, a
  one-time cost, not a per-round recurring one.
- **`round_duration_cycles`** (per round): `max` over every PE of (that
  same round's `TS_TERM_COL_BCAST_DONE` − `TS_VBCAST_ISSUE`) -- valid
  specifically because the termination relay forces every PE to agree on
  `nz_total` before *any* PE can issue the next round's `visited_bcast`
  (see `bool_pe.csl`'s `term_col_bcast_done()`) -- a genuine, unconditional
  synchronization boundary, unlike the relay's own internal sub-phases.
- **`local_compute`/`local_term_cond`**: already reliably measurable, and
  already excluded from any adjustment (see `PHASE_GROUP_AXIS`'s own
  comment) -- both are bracketed by a single PE's own entry/exit
  timestamps with no cross-PE dependency to reason about at all.
- **`communication`**: the remainder, `round_duration_cycles −
  local_compute_max − local_term_cond_max` -- everything else in the round
  (`visited_bcast`, `vertical_bcast`, the SpMV `reduce`, and the whole
  4-phase termination relay) lumped into one number, deliberately *not*
  decomposed further and *not* claimed to be chronologically contiguous
  within the bar (some of it happens before `local_compute`, some after
  `local_term_cond`).

**No device-side changes** -- every timestamp this needs was already in
`ts_buf`. New: `bfs_timing.compute_round_summary()`, wired into
`decode_phase_row()` (new CSV columns `round_duration_cycles`/
`total_runtime_cycles`), and `plot_bfs_timing.py` rewritten to stack only
these 3 segments per round (`local_compute`, `local_term_cond`,
`communication`) instead of the old 9-phase-plus-relay breakdown, with the
sum-of-rounds-vs-`total_runtime_cycles` comparison printed directly in the
plot's own suptitle as a live sanity check.

**Results, both matrices, 16x16 grid, 0 mismatches**:

| | `rmat_s6_e4.mtx` (n=64) | `rmat_s12_e4.mtx` (n=4096) |
|---|---|---|
| rounds | 3 | 5 |
| round bars sum | 14967 | 311267 |
| `total_runtime_cycles` | 15045 | 311423 |
| gap | +0.5% | +0.05% |

Both matrices confirm the sum-of-rounds check to well under 1% -- strong
validation that `round_duration_cycles` and `total_runtime_cycles` are
measuring the same thing correctly, using only hard synchronization facts
and no contested chain modeling.

The `rmat_s12_e4.mtx` breakdown (`plots/timing/timing_rmat_s12_e4_16x16_src0_ch1.png`)
tells an honest, believable story with none of sections 10-12's
back-and-forth: `local_compute` dominates every round (37161 / 92107 /
45319 / 30296 / 30296 cycles), `local_term_cond` is a real, essentially
*constant* per-round cost (9188 / 9308 / 8836 / 8764 / 8760 -- expected,
since it's a fixed-size masking loop over `blk`, not data-dependent) and
`communication` is comparatively small (13268 / 4496 / 4448 / 4508 / 4512
-- round 0 pays extra, plausibly initial fabric/route configuration cost
rather than a recurring one). This is consistent with -- and considerably
more trustworthy than -- section 12's device-time-share table (`local_compute`
already the largest single component there too), without relying on any
of the contested per-phase skew math.

## 14. Skew-adjustment fully removed; `device_time_cycles` finally adopts section 13's own alternative

Section 13 above set the skew-adjustment machinery aside for the
*plotting* path (`plot_bfs_timing.py`) and proposed a round-duration +
local-only alternative -- but `device_time_cycles` (the quantity every
`GTEPS`/`search_time_cycles*` CSV column is actually built from) never
adopted it. `decode_phase_row()` kept summing skew-adjusted per-phase
maxima (falling back to section 13's `total_runtime_cycles` only when
`rounds_completed > max_rounds`), so the GTEPS numbers this project cites
and the round bars a poster figure draws were, silently, built from two
different methodologies that happened to share column names.

This surfaced concretely: a poster figure's per-round "% communication"
(built from `round_duration_cycles`, section 13's alternative) didn't match
`plot_grid_scale_heatmap.py`'s own "% time in communication" cell (built
from the still-skew-adjusted `device_time_cycles`) for the same
`(scale, grid)` run -- 73% vs. ~59% for `rmat_s19_e16.balanced512x512`
@512x512, a 14-point gap traced directly to this inconsistency, not an
arithmetic bug in either plot. Investigating it settled the question this
project had left open since section 13: the skew-adjustment math itself is
unreliable and should be removed everywhere, not kept dormant for
`plot_pe_heatmap.py --skew` to keep probing.

**What changed, end to end:**

- **`src/bool_pe.csl`**: `ts_buf`/`NUM_TS_SLOTS` shrunk from 12 slots to 6
  (`TS_VBCAST_ISSUE`, `TS_COMPUTE_ENTRY`, `TS_REDUCE_ISSUE`,
  `TS_REDUCE_DONE`, `TS_RELAY_ISSUE`, `TS_TERM_COL_BCAST_DONE`) -- every
  slot that existed only to bracket an individual communication phase
  (`visited_bcast`'s own end, the termination relay's 4 internal sub-phase
  boundaries) or the `local_compute` reset/compact/expand sub-breakdown is
  gone. The two round-boundary markers survive (they feed `round_time`,
  never part of the removed machinery) along with `local_compute`/
  `local_term_cond`'s own entry/exit points. No change to the actual BFS
  algorithm or collective calls -- `record_ts()` calls are standalone
  instrumentation statements, not control flow.
- **`bfs_timing.py`**: `compute_skew_adjusted()`, `_chain_entry_ref()`,
  `PHASE_GROUP_AXIS`, `PHASE_ROOT_KIND`, `PHASE_MID_ONLY`,
  `SEARCH_TIME_PHASES` all deleted. `PHASES` shrunk to just
  `local_compute`/`local_term_cond`. `decode_phase_row()`'s
  `device_time_cycles` is now `total_runtime_cycles`
  (`round_trip_cycles.max()`) **unconditionally** -- not a fallback for
  truncated runs, the one and only definition. New
  `check_round_vs_total_communication()`: per-round `round_time -
  round_compute`, summed across rounds, checked against whole-run
  `device_time - total_compute` (`total_compute` includes the one-time
  `transpose_structure()` cost) -- logs a warning if the delta exceeds a
  5%-or-1000-cycle tolerance, since the two sides are measured two
  structurally different ways (sum of independent per-round stragglers vs.
  one whole-run span) and won't be bit-identical even when both are
  correct.
- **Final methodology** (per-run, and per-round where noted):
  - `compute` = `local_compute` + `local_term_cond` (per round) +
    `transpose` (one-time) -- each timed by its own tsc bracket, max
    across PEs.
  - `parent_resolve` -- unchanged, already an independent tic/toc bracket.
  - `device_time` = the whole-run `round_trip_start_buffer` →
    `round_trip_done_buffer` span (max across PEs) -- excludes
    `parent_resolve` by construction (`round_trip_done_buffer` is captured
    before `parent_resolve` starts, see `bool_pe.csl`'s
    `term_col_bcast_done()`), includes `transpose` (it runs inside that
    same span).
  - `communication` = `device_time - compute` (whole-run); per round,
    `round_time - round_compute` (`round_time` = `round_duration_cycles`,
    unchanged from section 13).
  - `search_time_cycles_no_transfer` = `device_time + parent_resolve`;
    `search_time_cycles` (full) = `h2d_seed + search_time_cycles_no_transfer
    + d2h` -- same CSV column names and on-the-wire meaning as before this
    change, just built from the new `device_time_cycles`.
- **CSV schema**: every `{phase}_adjusted_{min,max,avg}_cycles` column and
  `relay_critical_path_cycles` dropped. All pre-existing result CSVs
  (`results/bfs_timing.csv` and its `_legacy*`/`pre_directional`/
  `pre-parent-resolve-timing` siblings, `results/hw/bfs_timing.csv`,
  `results/graph500_searches*.csv`) were **deleted outright**, not
  migrated -- every row in them was built under the now-abandoned
  methodology, judged not reliable enough to carry forward under the same
  column names. Real-hardware re-validation (recompile + rerun the RMAT
  sweep and SNAP graphs) repopulates these from scratch under the new
  schema.
- **`plot_pe_heatmap.py`**: `--skew` mode and `--relay`/per-communication-
  phase rendering removed entirely (their underlying `ts_buf` slots no
  longer exist) -- it now only ever shows `local_compute`/`local_term_cond`
  per-PE grids, raw (there is no more "adjusted" variant to default to).
- **`plot_bfs_timing.py`**: the 4th panel (`local_compute`'s own
  reset/compact/expand grouped-bar breakdown) removed along with the
  `ts_buf` slots it depended on -- back to 3 panels (h2d, rounds, d2h).
  `plot_bfs_timing_poster.py`/`plot_grid_scale_heatmap.py`/
  `plot_bfs_scaling.py` needed no logic changes (they already read
  `round_duration_cycles`/`local_compute_max_cycles`/
  `local_term_cond_max_cycles`/`search_time_cycles_no_transfer` -- same
  column names, now consistently built end to end).

**Net effect**: the original poster-vs-heatmap discrepancy that started
this investigation is closed by construction -- both now derive
"communication" from the same `device_time`/`round_time` accounting, so a
poster's per-round bars aggregate to the same percentage the heatmap's
cell shows for that `(scale, grid)`, not a coincidentally-close or
wildly-different one.

## 15. `h2d_matrix`/`h2d_seed`/`d2h` timing: fixing (then dropping) a real understatement bug

Checking this project's own `h2d`/`d2h` timing against the Cerebras SDK's
official bandwidth-test example
(`examples/benchmarks/bandwidth-test/src/sync/pe.csl` in the SDK's own
`csl-extras` bundle) surfaced a real bug, distinct from anything sections
1-14 cover (which are all about *round/compute* accounting, not host<->device
transfer timing): `read_tic_toc_delta` computes, per PE, `toc[pe] -
tic[pe]`, then takes `.max()` across PEs. That is a structural **lower
bound** on the true cross-PE transfer span -- for any PE p, `toc[p] -
tic[p] <= max(toc) - min(tic)` always, with equality only when the single
longest-*duration* PE also happens to be both the earliest-starting and
latest-finishing one. The SDK's own example computes the right quantity
(`cycles_send = max(time_end) - min(time_start)`), correcting for
cross-PE clock skew via a one-time reference-clock sync first (PE tsc
counters aren't synchronized at boot).

**Fix, phase 1 (kept both, added the correct one alongside)**: a new
`f_sync_hostdevice()` entrypoint in `bool_pe.csl` reuses the kernel's own
existing `mpi_x`/`mpi_y` `<collectives_2d>` instances -- a single
`mpi_x.broadcast(MID, ...)` reaches the whole grid in one phase, since
`pe_id` in `collectives_2d` is scoped to one axis and every row broadcasts
independently and simultaneously -- rather than porting the SDK's own
hand-rolled sync module verbatim, whose hardcoded task IDs/colors/queues
all collided with `mpi_x`/`mpi_y`'s existing allocation. Task ID 25
(confirmed free by compiling, same empirical-trial convention as section
14's own task IDs). A new `tsc_ref_buffer` captures each PE's reference
timestamp; `bfs_timing.read_sync_corrected_span` reads it back alongside
the raw `tsc_start_buffer`/`tsc_end_buffer` and applies a propagation-delay
correction (`hop_distance = |pcol_id - MID)`, one cycle per hop, mirroring
the SDK example's own `(px+py)` correction for its differently-shaped relay)
before computing `max(corrected_toc) - min(corrected_tic)`. Called three
separate times per run -- before `h2d_matrix`, before `h2d_seed`, before
`d2h` -- rather than one shared upfront sync, since `h2d_seed`/`d2h` are
separated by the entire BFS run.

Validated on real hardware at scale (RMAT s17, `750x750`, WSE-3): the
`span_cycles ≥ max_cycles` proof held in every case, and `d2h` at this
scale showed a **51% understatement** (`max_cycles`=3,044,320 vs.
`span_cycles`=4,613,653) -- `h2d_matrix`/`h2d_seed` showed almost no gap
(~1,700 cycles), consistent with `d2h`'s narrow-column transfer pattern
having more real cross-PE fan-out skew than the two wide, one-shot bulk
uploads. A second back-to-back run of the identical compiled kernel showed
`max_cycles` and `span_cycles` both swinging by the *same* absolute ~431K
cycles between runs (real host<->device network jitter, confirmed stable
and near-zero for the purely-on-device `parent_resolve` bracket run
alongside it) -- i.e. the fix improves **accuracy** (removes a proven
systematic bias), not **precision** (real infrastructure noise is a
property of the transfer itself, not the measurement method).

**Fix, phase 2 (dropped the old one)**: once the new span was validated as
strictly more accurate and never worse, there was nothing left to keep the
old per-PE-max approach alongside for, so it was dropped for `h2d_matrix`/
`h2d_seed`/`d2h` specifically -- `{part}_min_cycles`/`_max_cycles`/
`_avg_cycles` no longer exist for those three; `{part}_span_cycles` is now
the only number recorded, and it feeds `search_time_cycles`/GTEPS directly
(previously the span was logged alongside but not wired into that math).
`read_tic_toc_delta` itself is unchanged and still correct for
`transpose_cycles`/`parent_resolve_cycles`/`round_trip_cycles` -- all
on-device-only quantities that never leave the fabric, where no cross-PE
sync concept applies and the old self-relative-delta approach was never
wrong. `plot_bfs_timing.py`'s h2d/d2h bars lost their per-PE min tick
(nothing to show any more, a single span isn't a per-PE statistic);
`plot_bfs_timing_poster.py` reads `_span_cycles` in place of `_max_cycles`.
`plot_grid_scale_heatmap.py` needed no changes -- its GTEPS/`%
communication` panels never touched `h2d`/`d2h` columns at all, confirmed
by direct comparison against its own already-published `rmat_grid_scale`
heatmap: the new run's GTEPS-excl-transfer and `%` communication for
`s17`/`750x750` matched that heatmap's cell exactly (13 GTEPS, 89%).
