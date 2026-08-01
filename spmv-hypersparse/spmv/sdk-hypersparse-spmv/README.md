# sdk-hypersparse-spmv — hypersparse, real-valued SpMV (f32)

This is the original Cerebras SDK `spmv-hypersparse` example, computing
`y = A*x` for a general (possibly rectangular, weighted) hypersparse matrix
`A`, distributed over a `prows x pcols` grid of PEs.

## Files

- `src/layout.csl`, `src/kernel.csl` — top-level glue: sets up `memcpy`,
  imports `hypersparse_spmv` and `allreduce2R1E`, exports host-callable
  functions. **WSE-2 only** — see "WSE-3 support" below.
- `src/hypersparse_spmv/{layout,pe}.csl` — the actual SpMV kernel (WSE-2).
- `src/allreduce2R1E/{layout,pe}.csl` — cross-PE clock synchronization, used
  only for accurate timing measurement (`f_sync`), not part of the SpMV math.
- `src_wse3/` — WSE-3 port of the three files above (same filenames, same
  `hypersparse_spmv/layout.csl`, no `allreduce2R1E/`) — see "WSE-3 support".
- `preprocess.py` — partitions `A` (given as CSR+CSC) into the per-PE
  hypersparse compressed-column format (`mat_col_idx/loc/len_buf`,
  `mat_rows_buf`, `y_rows_init_buf`, plus `mat_vals_buf` for the real values).
  Shared by both architectures — matrix partitioning doesn't depend on arch.
- `run.py` — host driver: compiles, distributes `x`/`A`, launches, times, and
  verifies against a dense scipy reference. Picks `src/` or `src_wse3/` based
  on `--arch`.
- `memory_usage.py` — per-PE memory footprint estimate, used to assert the
  chosen grid fits in 48KB SRAM before compiling.
- `commands_wse2.sh` / `commands_wse3.sh` — one-shot compile+run smoke test
  on `../../data/rmat4.4x4.lb.mtx` at a 4x4 grid, one per architecture.

## WSE-3 support

`src_wse3/` ports the WSE-2 kernel to WSE-3 by applying
[Cerebras/sdk-examples#23](https://github.com/Cerebras/sdk-examples/pull/23)
(originally written against the SDK's own bundled `spmv-hypersparse`
example, which this directory forked from) to this repo's copy. It is **not**
a drop-in replacement for `src/` — the changes are architecturally
WSE-3-specific and don't compile as WSE-2:

- **Queue remapping.** WSE-3's `memcpy` module reserves input queue 1 for
  its own command stream, which the WSE-2 kernel's `input_queues={4,1,6,7}`
  collides with (confirmed locally: compiling `src/` unmodified with
  `--arch wse3` fails with "initialization for this queue has already been
  set" at exactly that queue). `src_wse3/kernel.csl` remaps to
  `input_queues={2,3,4,5}`.
- **4 distinct output queues instead of 2 reused ones.** WSE-2's kernel
  reuses the same 2 output queues for both the north-south phase and the
  west-east phase (safe because the two phases never run concurrently, and
  WSE-2 lets each DSD carry its own `.fabric_color` regardless of which
  queue it's on). WSE-3 binds a fixed color to a queue at
  `@initialize_queue` time instead of per-DSD, so reusing one queue for two
  differently-colored trains is no longer possible — `output_queues` grows
  from `[2]u16` to `[4]u16` and `.fabric_color` is dropped from every
  `fabout_dsd` in `hypersparse_spmv/pe.csl` in favor of a
  `@get_output_queue(...)` + `@initialize_queue(..., .{.color = ...})` pair,
  gated behind `if (@is_arch("wse3"))`.
- **No `allreduce2R1E`-based `f_sync`.** `kernel.csl` drops the
  `allreduce2R1E` import and cross-PE reduction entirely; `f_sync` instead
  busy-waits on each PE's own tsc until a fixed threshold, then records that
  as the reference clock. Timing numbers from the two architectures aren't
  necessarily calibrated the same way as a result — see "Timing methodology"
  below, which still describes `src/`'s (WSE-2's) approach.

Verified end-to-end against the same scipy dense reference `run.py` already
checks WSE-2 against: `./commands_wse3.sh` reports `PASS` and
~141.8 MB/s on the 16x16/4x4 smoke test (vs. `commands_wse2.sh`'s
~118.6 MB/s) — both figures match the PR's own reported before/after numbers
almost exactly, which is a good sign this port is faithful to the original.

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
design, see `../../bfs/bool_diag_spmv/README.md`) would leave most of the grid idle
for those additional operations. The cost of this generality: `local_vec_sz
~= n/(pcols*prows)` per-PE vector memory (`P` times less than a
concentrated design), and `O(P)` communication rounds per phase instead of
`O(1)`.

## Timing methodology (this version's own, hardware-calibrated numbers)

`f_sync()` (via `allreduce2R1E`) synchronizes all PEs and samples a reference
clock first, since PE tsc counters aren't synchronized at boot. Then:
`cycles_send = max(time_end) - min(time_start)` (each adjusted by the
reference clock), `time_send_us = (cycles_send / 0.85) * 1e-3` (850MHz
clock), and `bandwidth = (2*nnz+m)*4 / time_send_us` MB/s. Use `run.py`
directly (or `commands_wse2.sh`) for this version's own calibrated numbers.

## Known limitations

- Compile-time buffer sizes (`max_local_nnz` etc.) are the *max* over all PEs
  — a poorly load-balanced matrix (e.g. a GRAPH500/RMAT-style graph with hub
  vertices, partitioned naively) can make every PE's buffers many times
  larger than the average PE needs, up to the point of a **linker failure**
  ("ran out of PE memory") for skewed-enough inputs. `../../util/analyze.cpp`
  (a separate tool in this repo) computes a row/column permutation that
  minimizes this skew.
- Measured against `bool_diag_spmv` (a boolean-semiring, diagonal-reduce
  redesign) on the same matrices, this kernel is consistently ~4-8x slower
  for the workload `bool_diag_spmv` targets — expected, since that redesign
  gives up the `P^2` distribution and ping-pong-compatible layout described
  above in exchange for a cheaper single-source-broadcast /
  single-target-reduce communication pattern.
