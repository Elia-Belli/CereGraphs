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
| `d2h` (`visited_buf` + `parent_local_buf` readback) | Part of Kernel 2 — "output has been written to memory" | **Placeholder** — measured, logged (both scripts), but not yet folded into `search_time_cycles`/GTEPS |

`d2h` is the one remaining open item: it's real per-search cost (Graph500's
own rule times it as part of the search), but folding it in changes every
existing GTEPS number, so it's deferred until that's a deliberate decision,
not a side effect of this doc update.

## 3. Current "search time" definition (cycles, in scope now)

**Implemented** — `run_bfs.py` logs this as `search_time_cycles`.
For one search (one CSV row from `run_bfs.py`):

```
search_time_cycles = h2d_seed_cycles
                    + sum over all profiled rounds r of:
                        visited_bcast_max[r] + vertical_bcast_max[r]
                        + local_compute_max[r] + reduce_max[r]
                        + local_term_cond_max[r] + relay_total_max[r]
```

Using each phase's `_max_cycles` (the straggler PE) per round, summed
across rounds — the same definition `plot_bfs_timing.py`'s stacked bars
already visualize (each round's bar height = sum of its phase segments).
`relay_total` is used directly here rather than re-summing its own 4
sub-phases, to avoid double-counting.

`h2d_matrix` and `d2h` are **excluded** from this sum for now (see the
placeholders above).

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

**Implemented** — `run_bfs.py` now decodes `visited_buf` (via
`extract_diag_result`/`oned_to_hwl_colmajor`, the shared `device_io.py`
helpers every driver script uses) and computes `m` as
`np.sum(visited[coo.row] & (coo.col <= coo.row))` on `A_csr.tocoo()` — one
vectorized pass, no Python-level loop over `n`/`nnz`.

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
- **Clock frequency is an assumed constant, not calibrated.** `CLOCK_FREQ_HZ
  = 875 MHz` converts `search_time_cycles` -> `search_time_seconds` for
  TEPS, but isn't calibrated against this simulator run in any way -- same
  "not an absolute hardware-calibrated figure" caveat this repo's other
  tsc-based timing already carries (see `bool_pe.csl`'s own tsc comment).
  Revisit if a real reference frequency for the simulator/hardware being
  targeted becomes available.
- **`d2h` placeholder status** (section 2) — needs a decision once the
  core TEPS number has settled: does readback belong in the denominator,
  and if so, amortized how?

## 6. Current status

**Implemented, end to end**: `run_graph500.py` runs the full spec-shaped
sweep -- one compile, one matrix upload (timed once, excluded from every
search), then N single-source searches (default 64) from distinct random
roots, each producing `search_time_cycles`, `m_edges_traversed`,
`m_convention`, `visited_count`, `matrix_symmetric`, `search_time_seconds`,
and `gteps` (one row per search, appended to `graph500_searches.csv`), plus
one summary row (`graph500_summary.csv`) with `harmonic_mean_gteps` (the
spec's own aggregation rule, section 1), `min_gteps`, `median_gteps`,
`max_gteps`, and `construction_time_seconds`. Between searches, no explicit
host-side reset is needed beyond re-uploading `x_buf` -- `bool_pe.csl`'s
`start_spmv()` already reinitializes `visited_buf`/`rounds_completed`/
`parent_local_buf`/`ts_round` on every fresh `f_spmv_iter()` call (see its
own comments). Verified against `data/rmat_s8_e4.mtx` (256 vertices, 8x8
grid): 64/64 searches passed their per-search scipy correctness check
(`--nocorrectness` to skip, on by default).

`run_bfs.py` remains the single-search deep-dive tool (tree plot, verbose
per-phase breakdown, `--show-parent-mismatch`) and still logs its own
`search_time_cycles`/`gteps` for that one search into `bfs_timing.csv` --
useful for drilling into one specific root's phase breakdown, not for the
spec's own 64-search aggregate.

What's left is everything in section 5 above: the giant-component root-
selection question, `d2h`'s denominator status, and (if it ever matters) a
calibrated clock frequency instead of the assumed 875 MHz.
