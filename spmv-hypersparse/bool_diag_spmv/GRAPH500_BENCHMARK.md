# Graph500 BFS benchmark methodology — what we're timing, and why

This is a separate document from `README.md` on purpose: it's about the
*benchmark methodology* (what counts as "the BFS time," how TEPS is
defined) rather than the kernel's own design. Scope right now is narrow —
get a real TEPS number out of `bench_timing.py` — so **construction and
readback are left as explicit placeholders below**, not fully resolved.

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

`bench_timing.py` already measures four separate things. Here's how each
one maps onto the Kernel 1 / Kernel 2 split above:

| bench_timing.py component | Graph500 analogue | Status |
|---|---|---|
| `h2d_matrix` (`mat_rows_buf`, `mat_col_idx/loc/len_buf`, `y_rows_init_buf`, `local_nnz*`) | **Kernel 1 (construction)** — built once, reused across searches | **Placeholder** — measured, logged, but not yet folded into any TEPS number |
| `h2d_seed` (`x_buf`, the search root) | Part of Kernel 2 — "immediately prior to visiting the search root" | **In scope now** |
| on-device BFS rounds (`visited_bcast`, `vertical_bcast`, `local_compute`, `reduce`, `local_term_cond`, `relay_*`, all from `ts_buf`/`record_ts()`) | Kernel 2 itself — the actual `run_bfs()` | **In scope now** |
| `d2h` (`visited_buf` + `parent_local_buf` readback) | Part of Kernel 2 — "output has been written to memory" | **Placeholder** — measured, logged, but not yet folded into any TEPS number |

The immediate priority is the middle two rows — get a real per-search
device time and a real TEPS number out of them. `h2d_matrix` and `d2h`
stay as separately-logged CSV columns for now; whether/how they eventually
join the TEPS denominator is an open question below, not decided yet.

## 3. Current "search time" definition (cycles, in scope now)

**Implemented** — `bench_timing.py` logs this as `search_time_cycles`.
For one search (one CSV row from `bench_timing.py`):

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
means edge `c -> r`; see `generate_boolean_reference` in `run_bool.py`).
The test matrices (`benchmarks/gen_rmat.py`) are explicitly **symmetrized**
before being written out ("symmetrize (undirected graph, standard for BFS
benchmarking)"), so `A_csr[v, u] != 0 <=> A_csr[u, v] != 0` — the same
undirected-with-both-tuples-stored shape Graph500's own reference graphs
have. That means the reference implementation's dedup rule ports directly
(implemented in `bench_timing.py`, vectorized rather than the loop form
below):

```python
m = sum(
    1
    for v in range(n) if visited[v]
    for u in A_csr.indices[A_csr.indptr[v]:A_csr.indptr[v + 1]]
    if u <= v
)
```

**Implemented** — `bench_timing.py` now decodes `visited_buf` (via
`extract_diag_result`/`oned_to_hwl_colmajor`, the same helpers
`test_iterative.py`/`plot_bfs_tree.py` already use) and computes `m` as
`np.sum(visited[coo.row] & (coo.col <= coo.row))` on `A_csr.tocoo()` — one
vectorized pass, no Python-level loop over `n`/`nnz`.

**Caveat confirmed by testing, not just theoretical**: this dedup rule is
only correct for a symmetric (undirected) `A_csr`. Running it against
`rand600.mtx` (an existing test fixture, *not* `gen_rmat.py`-generated)
produced `m=1836 > nnz/2=1800` — impossible for the real quantity, and a
clear tell that the input wasn't actually symmetric. Confirmed directly:
`rand600.mtx` has `A_csr != A_csr.T` and 3 self-loops (`gen_rmat.py`
explicitly avoids both). `bench_timing.py` now checks `A_csr` symmetry up
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
`bench_timing.py` picks the formula automatically based on
`matrix_symmetric` and records which one was used in the `m_convention`
column (`"undirected_dedup"` or `"directed_source_visited"`).

## 5. Known deviations from the full Graph500 protocol (not addressed yet)

- **One search, not 64.** `bench_timing.py` runs a single `--source` per
  invocation. The harmonic-mean-over-64-searches step doesn't apply until
  multiple searches per compiled matrix are actually run and aggregated.
- **Clock frequency is an assumed constant, not calibrated.** `CLOCK_FREQ_HZ
  = 875 MHz` in `bench_timing.py` converts `search_time_cycles` ->
  `search_time_seconds` for TEPS, but isn't calibrated against this
  simulator run in any way -- same "not an absolute hardware-calibrated
  figure" caveat this repo's other tsc-based timing already carries (see
  `bool_pe.csl`'s own tsc comment). Revisit if a real reference frequency
  for the simulator/hardware being targeted becomes available.
- **`h2d_matrix`/`d2h` placeholder status** (section 2) — needs a decision
  once the core TEPS number is working: do they belong in the denominator
  at all for a single-compile/many-searches workload, and if so, amortized
  how?

## 6. Current status

**Implemented, end to end**: `bench_timing.py` logs `search_time_cycles`,
`m_edges_traversed`, `m_convention`, `visited_count`, `matrix_symmetric`,
`clock_freq_hz`, `search_time_seconds`, and `gteps` (GTEPS -- 10^9
edges/s, the conventional Graph500-reporting unit) for every run. `m`/
`gteps` are always real, meaningful numbers for whatever graph you give it
(directed or undirected, see section 4) -- `m_convention` records which
formula was used, and only `"undirected_dedup"` (i.e. `matrix_symmetric`
True) is directly comparable to a Graph500-spec TEPS number.
`"directed_source_visited"` is real algorithmic-work-done information, not
a lesser or invalid result, just a different (and, for this kernel,
arguably more natural) counting convention. What's left is everything in
section 5 above: running all 64 searches instead of 1, deciding
whether/how `h2d_matrix`/`d2h` join the denominator, and (if it ever
matters) a calibrated clock frequency instead of the assumed 875 MHz.
