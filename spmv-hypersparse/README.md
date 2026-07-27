# spmv-hypersparse

Sparse matrix-vector multiplication on the Cerebras Wafer-Scale Engine, as a
building block toward a future graph BFS implementation. This directory
holds multiple **versions** of the SpMV kernel side by side — each is
self-contained (its own CSL source, its own host driver, its own
compile+run script) and independently comparable via the shared benchmark
harness. Versions are grouped into `spmv/` (SpMV-only designs) and `bfs/`
(BFS-capable designs); `data/`, `util/`, and `benchmarks/` are shared across
both groups.

## Layout

- **`spmv/sdk-hypersparse-spmv/`** — the original SDK `hypersparse_spmv`
  example: real-valued (f32), general (possibly weighted, rectangular)
  matrix, fully `P^2`-distributed vectors so it composes with dense-vector
  solver operations (dot products, AXPY) in a larger iterative solver.
  Supports both WSE-2 (`src/`) and WSE-3 (`src_wse3/`, ported from
  [Cerebras/sdk-examples#23](https://github.com/Cerebras/sdk-examples/pull/23)).
  See `spmv/sdk-hypersparse-spmv/README.md`.
- **`spmv/fp32_diag_spmv/`** — real-valued (f32) fork of `bool_diag_spmv`'s
  diagonal-broadcast/reduce design: square matrix, genuine multiply-accumulate
  (not a boolean OR-via-add surrogate), single SpMV round (`f_spmv`, host
  round-trip capable) or a fixed number of rounds entirely on-fabric with no
  host round trip (`f_spmv_iter`). Built to compare a device-only iterative
  kernel directly against `sdk-hypersparse-spmv`'s host-driven baseline. WSE-2
  only (no WSE-3 SDK example exists to compare against).
- **`bfs/bool_diag_spmv/`** — boolean-semiring redesign targeting BFS: square
  matrix, diagonal-targeted broadcast/reduce via the SDK's `<collectives_2d>`
  stdlib, concentrated (not fully distributed) vectors. Both a single SpMV
  round (`f_spmv`) and a device-only iterative BFS loop (`f_spmv_iter`,
  visited-masked, 4-phase termination relay) are exported from the same
  kernel. See `bfs/bool_diag_spmv/README.md`.
- **`bfs/sdk-hypersparse-spmv-bfs/`** — host-orchestrated BFS built on the
  unmodified `sdk-hypersparse-spmv` kernel: one `f_spmv` launch per hop, with
  the visited set, parent pointers, frontier masking, and termination all
  kept on the host. Directed and undirected graphs both work (via a free
  CSR/CSC transpose swap). Built deliberately naive, to make the per-hop host
  round-trip overhead visible before a fabric-resident version is attempted.
  Supports both WSE-2 and WSE-3, same as `sdk-hypersparse-spmv/`. See
  `bfs/sdk-hypersparse-spmv-bfs/README.md`.
- **`data/`** — input matrices (Matrix Market format, 1-based), shared across
  versions and benchmarks: the repo's original small test matrix
  (`rmat4.4x4.lb.mtx`) plus synthetic RMAT/GRAPH500-style and uniform-random
  matrices generated for benchmarking (regenerable — see
  `benchmarks/gen_rmat.py`).
- **`util/`** — `analyze.cpp`, a load-balancing tool: given a matrix and a
  PE grid shape, searches for a row/column permutation minimizing the
  variance of nonzeros per PE block. Version-agnostic (operates purely on
  the matrix file); build with the commands in its header comment.
- **`benchmarks/`** — timing comparison scripts across versions
  (`bench_orig_timing.py`, `bench_bool_timing.py`, shared helpers in
  `bench_common.py`/`bench_log.py`), raw results (`bench_results.jsonl`),
  and the write-up of everything measured so far (`bench_notes.md` —
  load-balancing impact, sparsity sensitivity, grid-size sensitivity,
  GRAPH500 rules-compliance research). Read `bench_notes.md` before assuming
  either version's relative performance in an untested regime.

## Quick start

**Important**: `cslc`/`cs_python` run inside a container that only binds the
*invocation's current directory* into its sandbox — a path that reaches
outside it (e.g. `../data/...`) is invisible even though it exists on disk.
`commands_wse2.sh` in each version folder relocates itself to this repo root
before doing anything, so it's safe to run from anywhere:

```
cd spmv/sdk-hypersparse-spmv && ./commands_wse2.sh    # or: cd bfs/bool_diag_spmv && ./commands_wse2.sh
```

`sdk-hypersparse-spmv/` and `sdk-hypersparse-spmv-bfs/` also have a `commands_wse3.sh` for WSE-3
(`bool_diag_spmv/` is WSE-2 only, unchanged by the WSE-3 work).

The benchmark scripts do **not** self-relocate — invoke them from *this*
repo root (not from inside `benchmarks/`), with matrix paths relative to
this root too:

```
cs_python benchmarks/bench_orig_timing.py --infile_mtx=data/<matrix>.mtx --num_pe_cols=N --num_pe_rows=N --driver=cslc --arch=wse2
cs_python benchmarks/bench_bool_timing.py --infile_mtx=data/<matrix>.mtx --num_pe_cols=N --num_pe_rows=N --driver=cslc --arch=wse2
```

(Run separately, not in the same process — the simulator can't be
instantiated twice in one process.) Same rule for `benchmarks/gen_rmat.py`:
run it as `cs_python benchmarks/gen_rmat.py <scale> <edgefactor> <seed>
data/<out>.mtx` from this repo root, not from inside `benchmarks/`.

`util/analyze` is a plain native binary (not containerized), so it has no
such restriction — see `benchmarks/bench_notes.md` for example invocations.

## Adding a future version

If/when a further-optimized or BFS-capable iteration is built, give it its
own directory (`<name>/` with its own `src/`, host driver, and README,
following the pattern above) under `spmv/` (SpMV-only) or `bfs/`
(BFS-capable) as appropriate, rather than branching inside an existing
version — that's what keeps the benchmark comparison in `benchmarks/`
meaningful across versions.
