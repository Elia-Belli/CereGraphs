# original_spmv — hypersparse, real-valued SpMV (f32)

This is the original Cerebras SDK `spmv-hypersparse` example, computing
`y = A*x` for a general (possibly rectangular, weighted) hypersparse matrix
`A`, distributed over a `prows x pcols` grid of PEs.

## Files

- `src/layout.csl`, `src/kernel.csl` — top-level glue: sets up `memcpy`,
  imports `hypersparse_spmv` and `allreduce2R1E`, exports host-callable
  functions.
- `src/hypersparse_spmv/{layout,pe}.csl` — the actual SpMV kernel.
- `src/allreduce2R1E/{layout,pe}.csl` — cross-PE clock synchronization, used
  only for accurate timing measurement (`f_sync`), not part of the SpMV math.
- `preprocess.py` — partitions `A` (given as CSR+CSC) into the per-PE
  hypersparse compressed-column format (`mat_col_idx/loc/len_buf`,
  `mat_rows_buf`, `y_rows_init_buf`, plus `mat_vals_buf` for the real values).
- `run.py` — host driver: compiles, distributes `x`/`A`, launches, times, and
  verifies against a dense scipy reference.
- `memory_usage.py` — per-PE memory footprint estimate, used to assert the
  chosen grid fits in 48KB SRAM before compiling.
- `commands_wse2.sh` — one-shot compile+run smoke test on
  `../data/rmat4.4x4.lb.mtx` at a 4x4 grid.

## How it works

Two communication phases per `spmv()` call:

1. **North-south "all-gather"**: `x` starts genuinely scattered — every
   single PE holds a distinct, tiny fragment (`local_vec_sz ~= n/(pcols*prows)`,
   i.e. the vector is chopped by column *and* by row, see the worked example
   in `dist_x_to_hwl`'s docstring in `run.py`). Every PE then broadcasts its
   own fragment vertically within its column (via a dynamic, switch-advanced
   hardware-multicast relay), so all `prows` PEs in a column end up with the
   full column's worth of `x`. This takes `O(prows)` sequential rounds
   because multiple PEs share the same physical color/route, taking turns.
2. **West-east "reduce-scatter"**: after local compute, each PE's partial
   result is routed bidirectionally across its row, split by which PE's
   output row-range it belongs to (via `y_local_low`/`y_local_high`), so `y`
   ends up distributed in the *same* fully-scattered, one-fragment-per-PE
   shape that `x` started in.

That specific choice — `y` coming out in the same shape `x` needs going in —
is what the `spmv(x,y); spmv(y,x)` ping-pong comment in `hypersparse_spmv/pe.csl`
is about: in principle it lets you re-apply `A` with zero data movement
between calls. **This is never actually exercised anywhere in this repo** —
`kernel.csl`'s `f_spmv()` always calls `spmv_mod.spmv(&x_tx_buf, &y_local_buf)`
in that fixed order, and no code calls it with the buffers swapped. It would
also need an explicit transpose step that doesn't exist here: `x`'s per-PE
index is a function of `(px*prows+py)`, `y`'s is `(py*pcols+px)` — different
functions of a PE's coordinates — so naively swapping buffers without moving
data across PEs would compute the wrong vector for any grid bigger than 1x1.

## Why it's built this way (composability, not raw SpMV speed)

The full `P^2`-way distribution (a unique, non-redundant fragment of `x`/`y`
at *every* PE, not just some of them) exists so this kernel can be a
subroutine inside a larger iterative solver — see the SDK's
`conjugate-gradient`/`bicgstab` examples, which layer `dot()` (needs every PE
holding a genuine partial sum) and AXPY-style vector updates on top of a
similar SpMV. Redundant/concentrated vector storage (like `bool_diag_spmv`'s
design, see `../bool_diag_spmv/README.md`) would leave most of the grid idle
for those additional operations. The cost of this generality: `local_vec_sz
~= n/(pcols*prows)` per-PE vector memory (`P` times less than a
concentrated design), and `O(P)` communication rounds per phase instead of
`O(1)`.

## Timing methodology (this version's own, hardware-calibrated numbers)

`f_sync()` (via `allreduce2R1E`) synchronizes all PEs and samples a reference
clock first, since PE tsc counters aren't synchronized at boot. Then:
`cycles_send = max(time_end) - min(time_start)` (each adjusted by the
reference clock), `time_send_us = (cycles_send / 0.85) * 1e-3` (850MHz
clock), and `bandwidth = (2*nnz+m)*4 / time_send_us` MB/s. Note
`../benchmarks/bench_orig_timing.py` deliberately does *not* use this
methodology (it skips `f_sync` for a simplified, uncalibrated comparison
against `bool_diag_spmv` — see `../benchmarks/bench_notes.md`); use `run.py`
directly (or `commands_wse2.sh`) for this version's own calibrated numbers.

## Known limitations (see `../benchmarks/bench_notes.md` for measurements)

- Compile-time buffer sizes (`max_local_nnz` etc.) are the *max* over all PEs
  — a poorly load-balanced matrix (e.g. a GRAPH500/RMAT-style graph with hub
  vertices, partitioned naively) can make every PE's buffers many times
  larger than the average PE needs, up to the point of a **linker failure**
  ("ran out of PE memory") for skewed-enough inputs. `../util/analyze.cpp`
  (a separate tool in this repo) computes a row/column permutation that
  minimizes this skew — see the load-balancing section of
  `../benchmarks/bench_notes.md` for how much it helps.
- Measured against `bool_diag_spmv` (a boolean-semiring, diagonal-reduce
  redesign) on the same matrices, this kernel is consistently ~4-8x slower
  for the workload `bool_diag_spmv` targets — expected, since that redesign
  gives up the `P^2` distribution and ping-pong-compatible layout described
  above in exchange for a cheaper single-source-broadcast /
  single-target-reduce communication pattern. See
  `../benchmarks/bench_notes.md` for the full comparison and caveats.
