# RESOLVED: bottom-up-only BFS + DCSC-value parent storage

**Update 2026-09-10: fixed and done.** The bug below (`preprocess()`'s
swap silently transposing the (px,py) PE-grid axes) was root-caused and
fixed -- see docs/ERRORS.md #22 for the full writeup. Dense
(`parent_resolve_variant==0`) was then removed entirely per explicit user
decision -- see docs/ERRORS.md #23. Re-verified clean (0/4096 mismatches,
scipy cross-check OK, sources {0,5,50}, RMAT s12 8x8) after both changes.
Nothing in this file needs acting on anymore -- kept below as the
historical record of how the bug was found. Next real work: mid-scale/
real-hardware re-verification and a memory-footprint A/B at real scale
(see docs/ERRORS.md #21's own "Status" line for what's still open).

---

Status as of 2026-09-07 (superseded, see update above). Implementation was
written but broken -- a real correctness bug blocked everything below.

## The goal (approved plan)

Full plan: `/home/elia/.claude/plans/right-now-i-dont-jaunty-island.md`
(still there, still accurate as a design doc — re-read it for the complete
rationale/design). Short version: make bottom-up the ONLY BFS traversal
strategy (`bool_pe.csl`) — remove `compute_topdown()`, the on-device
CSC→CSR `transpose_structure()`, and the `is_bottom_up`/`tau_switch_count`
runtime switch entirely. Host uploads the matrix ALREADY in
CSR-by-destination form instead. `parent_resolve_variant==2`'s local
storage is redefined: `parent_values[i]` (position-aligned to the resident
`mat_row_idx_buf[i]`) replaces the old discovery-order
`parent_compact_indices/values/count` + `parent_round_seen_bitmap` (docs
ERRORS.md #19/#20's scheme) — no longer needed since bottom-up's single
linear per-round pass never revisits a row twice.

Confirmed scope decisions (already made, don't re-litigate):
- `run_graph500.py` + `bfs/scripts/commands_wse3_graph500.sh` are OUT OF
  SCOPE — expected to break, don't fix.
- This redefines `parent_resolve_variant==2` in place, not a new variant
  number.

## What's actually been done

All edits below are ALREADY APPLIED to the working tree (uncommitted, on
branch `sparse-parents`):

- `bfs/implementation/src/bool_pe.csl` — full rewrite per the plan: removed
  `mat_col_idx/loc/len_buf`, `max_local_nnz_cols`, `row_to_bucket`/
  `cursor_buf`/`visited_pos`/`col_of()`/`transpose_structure()`/
  `compute_topdown()`/`is_bottom_up`/`tau_switch_count`/`direction_history`/
  `parent_compact_*`/`parent_round_seen_bitmap` entirely. Added
  `parent_values`/`PARENT_VALUES_LEN`. `compute_bottomup()` now writes
  `parent_values[i] = global_c` directly. `task compute()` always calls
  `compute_bottomup()`. `term_col_bcast_done()`'s convergence branch does a
  one-time compress-scan (`parent_values`/`mat_row_idx_buf` →
  `parent_send_indices/values` + bitmap) before the SAME
  `reduce_select_any_indexed_precompacted()` relay call as before (#20's fix
  untouched).
- `bfs/implementation/src/layout_bool.csl` — matching param/export removals.
- `bfs/implementation/device_io.py` — `csl_compile_core`/
  `csl_compile_core_appliance` lost `max_local_nnz_cols`/`tau_switch_count`
  params.
- `bfs/scripts/run_bfs.py` + `run_bfs.appliance.py` — mirrored edits:
  dropped `--directional`/`DEFAULT_TAU_SWITCH_FRAC`, swapped the
  `preprocess()` call's CSR/CSC argument PAIRS (see the bug below — **this
  swap is the prime suspect**), renamed `mat_col_*`→`mat_row_*` symbol
  lookups/uploads, dropped `local_nnz_cols`/`is_bottom_up_dbg`/
  `direction_history`/`transpose_*` readback+CSV columns.
- `bfs/implementation/bfs_timing.py` — `check_round_vs_total_communication`
  lost its `transpose_max_cycles` parameter.
- `bfs/plots/plot_bfs_timing.py` — dropped the `transpose` segment/color
  (this one was a real latent crash risk fixed in passing: it indexed
  `row["transpose_max_cycles"]` directly, not `.get()`, so it would have
  raised `KeyError` on any new CSV row once that column stopped being
  written — now removed instead).
- `bfs/plots/plot_bfs_timing_poster.py` — docstring-only trim.
- `docs/ERRORS.md` — new entry **#21** written (Where/What was
  built/Status), explicitly left with **"[Fill in: correctness re-run ...]"**
  placeholder text since verification hadn't passed yet when written. Update
  that entry once the bug below is fixed and real numbers exist.

**Compile-only verification passed** (both `parent_resolve_variant=0` and
`=2`), both via raw `cslc` directly on `layout_bool.csl` and via
`run_bfs.py --compile-only`, on the RMAT s12 8x8 smoke matrix.

## THE BUG (why everything is blocked)

A real correctness run (`run_bfs.py` without `--compile-only`, scipy
cross-check enabled) on `data/rmat_s12_e16.balanced8x8.mtx`, 8×8 grid,
fails identically for **both** `dense` (variant 0) and `indexed` (variant
2), across sources {0, 5, 50}:

```
[[ visited vs scipy mismatches: device=303 ]]
[[ mismatches (visited, device vs scipy): 303 / 4096 ]]
[[ scipy cross-check: FAILED ]]
[[ visited_count = 3169 / 4096, ... ]]
```

**Identical failure signature for both variants** (same 303 mismatches,
same 3169/4096 visited count, at every source tried) — this is the single
most important fact: it means the bug is in the SHARED traversal/data path
(`compute_bottomup()`'s bitmap logic, or the uploaded matrix structure
itself), NOT anything specific to `parent_resolve_variant`. Dense's own
parent-storage code (`parent_local_buf`) is completely untouched by this
session's redesign — if dense is broken, the break is upstream of parent
storage entirely.

**Note**: the user said mid-session "dont test the dense variant, it isnt
supposed to work" — but indexed shows the EXACT SAME failure, which was not
yet reconciled with the user when this session ended. **First thing to do
on resume: surface this to the user** — either their expectation about
dense was based on a different understanding than what's actually
happening (both variants equally broken, not just dense), or there's
context this handoff is missing about why dense specifically was expected
to fail. Don't assume either way — ask.

### Leading hypothesis: the `preprocess()` call-site swap direction

The theory behind the fix (see `bool_pe.csl`'s module docstring and
`run_bfs.py`'s own comment at the `preprocess()` call site): for a square
matrix on a square grid with `.sorted_indices()` applied, `csc(A^T) ==
csr(A)` and `csr(A^T) == csc(A)`, so swapping which physical array
(`A_csr.indptr/indices` vs `A_csc.indptr/indices`) goes into which
parameter slot should make `preprocess()` build the transposed
(CSR-by-destination) layout with zero internal changes to
`preprocess_bool.py`.

**This was NOT empirically verified before running out of turn budget.**
Was in the middle of re-deriving it by reading `preprocess_bool.py` lines
67-117 directly (not just trusting the earlier Explore-agent summary) when
interrupted. Concern found so far, not yet resolved either way: the
function's "CSC-role" branch (lines 67-78) does
```python
col_per_nz = np.repeat(np.arange(ncols, dtype=np.int64), np.diff(cscColPtr))
```
— `ncols` here is a plain function argument (still literally `A`'s own
`ncols`, unchanged by the swap), not re-derived from whatever's actually
passed as `cscColPtr`. For a **square** matrix (`nrows == ncols`, asserted
elsewhere) this doesn't crash from a shape mismatch, but whether the
resulting `A_colidx`/`A_colloc`/`A_collen`/`A_rows` arrays actually end up
holding the correct ROW-grouped semantics under the swap (vs. just
happening to have the right shapes) was NOT confirmed — could easily be
subtly wrong (e.g. right cardinalities, wrong actual row/column identities
or offsets) in a way that would exactly produce "compiles fine, wrong
answer" — matching what's observed.

**Next step**: write a small standalone test (needs `numpy`/`scipy`, NOT
available on the local edit machine — run via `cs_python` on `cer-usn-01`,
inside the activated SDK env) that:
1. Loads `data/rmat_s12_e16.mtx` (or an even smaller hand-built matrix for
   easier manual inspection), builds `A_csr`/`A_csc` exactly as
   `run_bfs.py` does.
2. Calls `preprocess()` with the swapped argument order currently in
   `run_bfs.py`.
3. For one PE block (e.g. the diagonal PE(0,0)), manually reconstructs
   "distinct row → its source columns" from the returned
   `mat_col_idx_buf`/`mat_col_loc_buf`/`mat_col_len_buf`/`mat_rows_buf`
   fields (which `run_bfs.py` now renames to `mat_row_*` post-swap) and
   compares directly against `A_csr`'s own `.indptr`/`.indices` for that
   same block (ground truth, since CSR literally IS "row → columns").
4. If they don't match, the swap direction (or the function's internal
   `ncols`/`bx` assumptions) is wrong — try swapping the OTHER way, or
   consider that `preprocess_bool.py` may need an actual code change (not
   just a call-site swap) despite the earlier analysis concluding
   otherwise.

If a plain swap genuinely cannot work (the `ncols`/`bx`-hardcoding turns
out to be a real blocker, not just a false alarm), the fallback is: give
`preprocess()` an explicit new code path (or a `transpose: bool` flag) that
builds the row-grouped compact arrays directly from `csrRowPtr`/`csrColInd`
without reusing the CSC-shaped branch at all — more invasive than planned,
but still confined to `preprocess_bool.py`.

## Environment facts (verified this session, correcting stale ones)

- **Compile/run** (`cslc`, `cs_python`, the simulator): `cer-usn-01`.
  Always `source /home/elia/cs_appliance_sdk/bin/activate` first, and
  `export no_proxy='10.125.8.2,.cerebras.internal,localhost,127.0.0.1'`
  (+ matching `NO_PROXY`) before compiling/running.
- **`git`**: `cer-usn-02`, **NOT** `cer-usn-01` — the existing
  `bfs/HANDOFF.md`'s claim that git lives on cer-usn-01 is wrong (verified
  directly this session: cer-usn-01 has no `git` on PATH; cer-usn-02 does
  and shows the real repo state). Also saved to this session's persistent
  memory (`~/.claude/projects/-home-elia/memory/cereGraphs-git-host.md`) so
  future sessions don't need to re-derive it.
- The local edit machine (where file edits happen) has **no numpy/scipy,
  no git** — any Python needing those, or any git command, must go through
  one of the two ssh targets above.
- Appliance is a single shared allocation — real-hardware jobs must run
  strictly sequentially. Nothing in this session touched real hardware for
  the bottom-up work yet (all testing so far is free-simulator, RMAT s12
  8x8) — don't jump to real hardware until the bug above is fixed and at
  least the mid-scale (s19 512x512) simulator check passes too.
- stdout from a `cs_python`/`cslc`-wrapped remote command is fully buffered
  until process exit when not attached to a TTY — no incremental progress
  visible mid-run; this is normal, not a hang. Always add a hard-kill
  timeout (`timeout -k 10 <N>`) since a fatal on-device fault can leave the
  simulator's cleanup hung well past its own useful work.

## Test data already on disk (cer-usn-01, `/home/elia/CereGraphs/data/`)

- `rmat_s12_e16.balanced8x8.mtx` (+ `.operm`) — smoke scale, 8×8 grid.
- `rmat_s19_e16.balanced512x512.mtx` (+ `.operm`) — mid scale, `blk=1024`,
  the config that PREVIOUSLY failed to compile (docs ERRORS.md #8/#17)
  and now compiles clean after this session's earlier #19/#20 fixes
  (unrelated to the bottom-up work, already landed and verified before the
  bottom-up redesign started).
- `rmat_s20_e16.balanced750x750.mtx` (+ `.operm`) — `blk=1399`, confirmed
  (also pre-bottom-up-work) to still hit the genuine PE-memory ceiling —
  expected, documented, not a regression.
- Raw (unbalanced) `rmat_s19_e16.mtx`/`rmat_s20_e16.mtx` also present if
  re-balancing to a different grid is ever needed.

None of this data needs regenerating — reuse it for bottom-up debugging.

## Docs state

`docs/ERRORS.md` #21 is written but has a `[Fill in: ...]` placeholder
where the correctness verification result should go — update it with the
real outcome (bug found + root cause + fix, or whatever the resolution
turns out to be) once resolved. Don't leave the placeholder in a committed
version.

## Not yet done (from the original plan, still pending regardless of the bug)

- Fix the correctness bug above.
- Re-run the full verification ladder from the plan: smoke (multiple
  sources, BOTH variants) → mid-scale (s19 512x512, both variants, plus a
  fresh `cslc`+`cs_readelf -m` memory A/B against the pre-bottom-up
  baseline) → real hardware last.
- Fill in #21's real numbers/status in ERRORS.md.
- Nothing has been committed to git yet — working tree on `sparse-parents`
  still has all these changes uncommitted (confirmed via `cer-usn-02`
  earlier this session).

## Additional lead found late in the session (not yet acted on)

No existing precedent for the `preprocess()` swap: checked the reference
kernel `preprocess_bool.py` was modeled on --
`/home/elia/cs_sdk-2.10/csl-extras-202604101435-6-d2f7d96e/examples/benchmarks/spmv-hypersparse/`
(non-boolean `hypersparse_spmv`, has its own `preprocess.py` + `run.py`).
Its `run.py` (~line 503) calls `preprocess()` with the STRAIGHT, unswapped
argument order (`csrRowPtr, csrColInd, ..., cscColPtr, cscRowInd`) -- same
as this repo's own pre-bottom-up call. So the swapped call this session
introduced is genuinely novel, untested anywhere, not a known-good pattern
being reapplied. Worth comparing that reference `preprocess.py` against
`preprocess_bool.py` line-by-line if the standalone test described above
doesn't quickly reveal the issue -- it may already show how to correctly
build a row-grouped (CSR-by-destination) layout without a call-site swap.
