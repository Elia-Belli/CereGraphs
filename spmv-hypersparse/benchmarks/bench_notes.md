# Benchmark notes: original hypersparse_spmv vs bool_pe.csl (diagonal-reduce)

Raw per-run timing data lives in `bench_results.jsonl` (append-only, one JSON
record per `bench_orig_timing.py`/`bench_bool_timing.py` run: kernel, input
matrix, n, nnz, PE grid, raw cycles, time_us). This file is the narrative/
interpretation layer on top of that log, plus the load-balancing and
GRAPH500-legality investigation that isn't captured there.

Methodology throughout: memcpy excluded from timing; raw `max(tsc_end) -
min(tsc_start)` across all PEs, no cross-PE clock-skew correction (same
simplified methodology applied to both kernels, so the *difference* is
meaningful even though neither number is a hardware-calibrated absolute
figure). Clock assumed at 850MHz for the cycles->us conversion.

## Test matrices

All matrices live in `../data/`. RMAT ones are regenerable via
`cs_python benchmarks/gen_rmat.py <scale> <edgefactor> <seed> data/<out>.mtx`
— run from the **repo root** (`spmv-hypersparse/`), not from inside
`benchmarks/`: `cs_python`'s container only binds the invocation's cwd, so a
`../data/...` output path would be invisible to it even though it exists on
disk (see the top-level `README.md`). Standard GRAPH500 parameters (A=.57
B=.19 C=.19 D=.05), symmetrized, deduped, self-loops removed, seed=0.
Balanced variants are produced from the unbalanced file via `util/analyze`
(a plain native binary, not containerized, so no such restriction — can be
run from anywhere with ordinary relative paths; see the load imbalance
section below for the exact invocation).

| file | n | nnz | avg deg | max deg / notes |
|---|---|---|---|---|
| `rand600.mtx` | 600 | 3600 | 6.0 | uniform random, not skewed |
| `rmat4.4x4.lb.mtx` (repo's existing test matrix) | 16 | 108 | 6.75 | small, hand-curated |
| `rmat_s12_e16.mtx` | 4096 | 97194 | 23.7 | 1336 |
| `rmat_s12_e16.balanced8x8.mtx` | 4096 | 97194 | 23.7 | balanced for an 8x8 grid |
| `rmat_s12_e16.balanced16x16.mtx` | 4096 | 97194 | 23.7 | balanced for a 16x16 grid |
| `rmat_s12_e4.mtx` | 4096 | 28666 | 7.0 | sparser (edgefactor=4) |
| `rmat_s12_e4.balanced8x8.mtx` | 4096 | 28666 | 7.0 | balanced for an 8x8 grid |
| `rmat_s14_e16.mtx` | 16384 | 425218 | 26.0 | 3582, bigger — balancing/benchmark incomplete (see below) |

## Load imbalance (local_nnz max/mean across PEs, from `preprocess_bool.py`)

| matrix | grid | before balancing | after `util/analyze.cpp` |
|---|---|---|---|
| uniform random (n=600) | 6x6 | 1.27x | not needed |
| RMAT s12_e16 (n=4096, avg deg 23.7) | 4x4 | 4.24x | not tested |
| RMAT s12_e16 (n=4096, avg deg 23.7) | 8x8 | **8.38x** | **1.05x** |
| RMAT s12_e4 (n=4096, avg deg 7.0, sparser) | 8x8 | not measured | 1.18x |
| RMAT s14_e16 (n=16384, avg deg 26.0) | 8x8 | not measured | 1.05x |

`util/analyze.cpp` (repo's existing "load balance analysis tool", searches
permutations P,Q minimizing `var(nnz(A_ij))`) works very well on these
synthetic RMAT graphs — needed one host-side fix to compile on this GCC
version (`util/include/argparse/argparse.hpp` was missing `#include
<utility>` for `std::as_const`; one-line fix, unrelated to any kernel logic).

Side finding: even after balancing for nnz *count*, the fraction of a local
block's own column range actually touched (`local_nnz_cols / bx`) is 28%
(sparser matrix) to 45-57% (denser matrix) — nowhere near what "hypersparse"
usually implies (a tiny fraction of the local dimension touched). RMAT/
GRAPH500-style graphs don't have the kind of local block structure the
compaction format is optimized for, at these grid sizes.

## Timing comparison (memcpy excluded; see `bench_results.jsonl` for raw records)

| matrix | grid | original (f32) | bool (ours) | ratio (bool vs original) |
|---|---|---|---|---|
| `rmat4.4x4.lb.mtx` | 4x4 | 6308 cyc / 7.42us | 1634 cyc / 1.92us | 3.9x faster |
| uniform random n=600 | 6x6 | 38188 cyc / 44.93us | 7305 cyc / 8.59us | 5.2x faster |
| RMAT s12_e16, **unbalanced** | 8x8 | **compile failure** (linker: out of PE memory, `max_local_nnz=12728` x 6B/nnz > 48KB SRAM) | 148044 cyc / 174.17us (survives — no `mat_vals_buf`, half the per-nnz footprint) | n/a (original didn't run) |
| RMAT s12_e16, balanced | 8x8 | 302738 cyc / 356.16us | 38470 cyc / 45.26us | **7.9x faster** |
| RMAT s12_e4, balanced (sparser, avg deg 7.0) | 8x8 | 132461 cyc / 155.84us | 19854 cyc / 23.36us | **6.7x faster** |
| RMAT s14_e16, balanced (bigger, n=16384) | 8x8 | not completed — hit an infra-level linker error (`cannot open .../*.o`, likely host memory pressure during a large parallel compile, not a kernel-design issue) on repeated attempts | not attempted | incomplete, worth retrying later |
| RMAT s12_e16, balanced (same matrix as the 8x8 row, bigger grid) | 16x16 | 108018 cyc / 127.08us | 14734 cyc / 17.33us | 7.3x faster |

**Effect of a bigger PE grid on the same matrix** (n=4096, nnz=97194, avg deg
23.7; balanced separately for each grid size, imbalance 1.05x at 8x8 and
1.15x at 16x16): going from 8x8 to 16x16, *both* kernels got substantially
faster in absolute terms (original: 302738 -> 108018 cycles; bool: 38470 ->
14734 cycles) — local blocks shrink faster than the extra communication
rounds cost, for both designs, in this range. The ratio between them stayed
roughly flat (7.87x -> 7.33x, if anything very slightly narrower), not a
widening or a reversal. So "a bigger PE grid makes the original relatively
better" doesn't show up as a trend at 8x8 -> 16x16 for this graph — 10x10 and
12x12 were skipped since 16x16 already ran successfully and showed no such
trend emerging.

**Answer to "does the original do relatively better on sparser matrices?"**
On the one controlled comparison we have (same n=4096, same 8x8 grid, only
edge factor differs: 16 vs 4), the gap *narrows slightly* (7.9x -> 6.7x) but
does not come close to reversing — ours stays substantially faster. Likely
mechanism: the original's phase-2 reduce transmits real sparse (row,val)
lists sized by local nnz, so its communication cost (not just its compute)
shrinks with sparsity too; our phase-1/phase-2 communication is a *fixed*
`blk`-sized transfer regardless of sparsity, so we don't get that same
scaling benefit — but we're starting from a big enough lead (fewer
communication *rounds*, `O(1)` vs `O(P)`) that this doesn't close the gap
within the sparsity range tested (avg degree 7-24).

## GRAPH500 rules: is balancing/reordering the input legal?

Researched via the official spec (graph500.org/?page_id=12) — quoting the
relevant rules directly:
- "The first kernel may transform the edge list to any data structures (held
  in internal or external memory) that are used for the remaining kernels."
- "The graph may be represented in any manner, but it may not be modified by
  or between subsequent kernels."

Reading: Kernel 1 (construction) is explicitly allowed to transform/reorder
the graph into whatever internal representation it wants — that's exactly
what "any data structures... used for the remaining kernels" describes. The
constraint is that this representation must then stay **fixed** across all
of Kernel 2's BFS searches (not re-tuned per search root). A load-balancing
reorder like `util/analyze.cpp`'s — computed once from graph topology alone,
before any BFS root is chosen, and reused identically for every search —
fits squarely within that: it's a Kernel 1 transform, not a per-search
modification.

Caveat: the spec text doesn't use the words "load balancing" explicitly, so
this is an inference from the general "any data structure" + "no
modification between kernels" language, not a verbatim confirmation of this
specific technique. It would also need to be counted inside Kernel 1's timed
construction phase (GRAPH500 reports construction and BFS search time
separately), not excluded from timing. Worth a second read of the full spec
(and the reference implementation's own preprocessing, if it does any) before
relying on this for an actual compliant submission.

Sources:
- [Benchmark Specification - Graph 500](https://graph500.org/?page_id=12)
- [graph500/Graph500.org (reference spec text mirror)](https://github.com/graph500/graph500/blob/master/Graph500.org)
