# Handoff: remove the single-iteration (`f_spmv`) BFS path, make iterative the default

## Goal

`bfs/` exports two entrypoints from the same compiled kernel
(`src/layout_bool.csl`):

- **`f_spmv`** — "single iteration": one broadcast → local boolean multiply →
  reduce-to-diagonal, then the kernel exits. No masking, no termination
  relay, no direction-optimizing switch, no parent tracking.
- **`f_spmv_iter`** — "iterative": repeats that same core round on-device
  (cumulative visited-set masking, a 4-phase termination relay,
  top-down/bottom-up switching, parent tracking) until BFS convergence, with
  no host round-trip.

Make `f_spmv_iter` the sole/default path: delete `f_spmv` and everything
whose only job is exercising it.

There is **no CLI flag or compile param** toggling single-vs-iterative — the
split is purely "which exported entrypoint does the host call", and both are
always compiled together. So this is a code-deletion/edit exercise, not a
flag flip.

This document was produced by a read-only identification pass (no code was
changed). It's meant to be executed by an agent in an environment that can
actually compile and run the kernel, so it can verify the program still
works after each step — see "Environment notes" and "Verification" below.

## A. Delete outright (dedicated only to single-iteration)

- `bfs/run_single_spmv.py` — host driver that only ever calls
  `runner.launch("f_spmv", ...)` (`run_single_spmv.py:278`).
- `bfs/commands_wse3.sh` — smoke-test script whose only job is
  compile + `run_single_spmv.py --run-only`.
- `bfs/run_host_driven_bfs.py` — its regression test's whole premise is
  comparing one `f_spmv_iter` launch against a host-driven loop of
  sequential `f_spmv` calls as the correctness baseline (module docstring
  at `run_host_driven_bfs.py:1-65`; the `f_spmv` launch itself at
  `run_host_driven_bfs.py:425`). Once `f_spmv` is gone, this test has no
  baseline left to run against, so it goes too (confirmed with the repo
  owner — not just inferred).

Also sanity-check: `tests/commands_wse2_bfs.sh` and `README.md` both
reference a `commands_wse2.sh` that does not currently exist in the tree
(stale, likely left over from an earlier rename) — not part of this task,
but don't let its absence be mistaken for something this pass deleted.

## B. Edit (shared between both modes — strip the single-iteration branches, keep the rest)

- **`bfs/src/bool_pe.csl`** — pervasively `if (is_iterative) {…} else {…}`.
  Remove the `is_iterative` var (`bool_pe.csl:630`), `fn f_spmv()` itself
  (`bool_pe.csl:754-757`), and every `is_iterative`/`!is_iterative` guard,
  keeping only the surviving (iterative) branch. Key sites:
  - `bool_pe.csl:699-722` — `start_spmv()`'s visited/parent-buf reset
  - `bool_pe.csl:726-745` — first-round dispatch (the `else` one-shot
    broadcast goes away entirely)
  - `bool_pe.csl:900-909`, `939-944` — parent-candidate tracking in
    `compute_topdown()`/`compute_bottomup()`, becomes unconditional
  - `bool_pe.csl:951-971` — `compute()`'s timestamp recording, becomes
    unconditional
  - `bool_pe.csl:979-984` — **the crux of the change**: `reduce_done()`'s
    `if (!is_iterative) { sys_mod.unblock_cmd_stream(); return; }` early
    exit is exactly what makes `f_spmv` stop after one round; deleting it
    is what makes every call run the full iterative loop
  - `bool_pe.csl:1158-1196` — `term_col_bcast_done()`'s end-of-run parent
    resolve, becomes unconditional
  - Also update doc comments describing the dual design, e.g.
    `bool_pe.csl:1` (module docstring) and `bool_pe.csl:447`.
- **`bfs/src/layout_bool.csl`** — drop the `f_spmv` export
  (`layout_bool.csl:120`); update the top doc-comment, which currently
  frames the file as "(one iteration)" (`layout_bool.csl:1`); the
  `max_rounds`/`tau_switch_count`/`parent_resolve_variant` param defaults
  at `layout_bool.csl:25-40` currently exist so non-iterative-aware callers
  don't need to pass them — that reasoning goes away once those callers are
  gone, though the params/defaults themselves are still useful and can stay.
- **`bfs/device_io.py`** — doc-only edits: module docstring lists
  `run_single_spmv.py` as a caller (`device_io.py:2`);
  `memcpy_h2d_chunked`'s comment says it's "kept for
  run_single_spmv.py/run_host_driven_bfs.py" (`device_io.py:117`, `:373`); a
  comment contrasting `f_spmv_iter`'s self-zeroing invariant against
  "run_single_spmv.py's one-shot f_spmv" (`device_io.py:169-171`). The
  functions themselves (`dist_x_to_diag_hwl`, `pack_dense_to_bitmap`,
  `unpack_bitmap_to_dense`, `extract_diag_result`, `csl_compile_core`)
  remain needed by the iterative path — don't remove them.
- **`bfs/graph_loader.py`** — module docstring lists `run_single_spmv.py`
  among its callers (`graph_loader.py:2`) — trivial doc edit, no functional
  branching.
- **`bfs/cmd_parser.py`** — currently shared by `run_single_spmv.py` and
  `run_host_driven_bfs.py` (per `README.md:355-356`). Once both callers are
  removed, check whether any remaining driver (`run_bfs.py`,
  `run_bfs.appliance.py`, `run_graph500.py`) also needs it, or whether it
  becomes dead code to fold away.
- **`bfs/README.md`** — needs a real rewrite, not just spot edits. Sections
  that currently document both entrypoints side by side: the intro
  (`README.md:20`), the `src/layout_bool.csl`/`run_single_spmv.py`/
  `run_host_driven_bfs.py` file-list entries (`README.md:27-59`), the
  `commands_wse2.sh`/`commands_wse3.sh` smoke-test section
  (`README.md:131-151`), the square-matrix/build-flow section referencing
  `run_single_spmv.py` (`README.md:328-356`), and the explicit
  "`f_spmv`/`f_spmv_iter`, both exported from the same compiled kernel"
  section (`README.md:388-404`).

## C. Leave alone (iterative-only — confirms iterative is the mature, default path)

`run_bfs.py`, `run_bfs.appliance.py`, `run_graph500.py`, `bfs_timing.py`,
`plots/bfs_tree_plot.py`, `plots/plot_bfs_timing.py`,
`plots/plot_bfs_timing_poster.py`, `plots/plot_pe_heatmap.py`,
`commands_wse3_iterative.sh`, `commands_wse3_graph500.sh`,
`tests/commands_wse2_bfs.sh`, `rmat_grid_sweep.sh`, `run_snap_sweep.sh`,
`run_snap_sweep_screen.sh`, `sweep_bfs.sh`, `slurm_bfs.sh`,
`slurm_graph500.sh` — all of these exclusively drive `f_spmv_iter` already;
no changes expected here beyond what falls out of B's edits.

## D. Unrelated to this split — do not touch for this refactor

`preprocess_bool.py` (no `is_iterative` awareness at all),
`src/collectives_2d/*.csl`, `src/*_reduce_or_test.csl`,
`src/*_reduce_select*_test.csl`, `tests/run_reduce_or_test.py`,
`tests/run_reduce_select_test.py`, `tests/run_transpose_test.py`,
`tests/run_transpose_device_test.py` (tests a different,
direction-optimizing-BFS/transpose feature; happens to also call
`f_spmv_iter`, but isn't part of the single/iterative split).

Two other params in `bool_pe.csl`/`layout_bool.csl` are orthogonal and must
not be confused with this distinction: `parent_resolve_variant` (A/B
parent-resolve collective choice) and `tau_switch_count`/`--directional`
(top-down/bottom-up direction-optimizing switch, meaningful only inside the
iterative loop). Leave both as-is.

## Suggested execution order

1. Delete the three files in section A.
2. Edit `src/bool_pe.csl` and `src/layout_bool.csl` per section B, removing
   `is_iterative`/`f_spmv` entirely.
3. Recompile (`cslc`) and run the remaining iterative smoke tests (see
   Verification below) to confirm nothing broke.
4. Do the doc-only edits in `device_io.py`, `graph_loader.py`,
   `cmd_parser.py`, `README.md`.
5. Final sweep: `grep -rn "f_spmv\b\|run_single_spmv\|is_iterative" bfs/`
   should return nothing.

## Environment notes (needed to actually compile/run and verify)

- Real hardware runs happen on `cer-usn-01` via ssh; the repo is
  NFS-mirrored to the local editing machine, but `git`/`cslc`/the appliance
  SDK only exist on `cer-usn-01`.
- Always `source /home/elia/cs_appliance_sdk/bin/activate` before any
  appliance-mode Python invocation — a bare `python` fails instantly
  otherwise.
- Always `export no_proxy='10.125.8.2,.cerebras.internal,localhost,127.0.0.1'`
  (and matching `NO_PROXY`) before compiling/running — otherwise compile
  submission fails with a generic-looking `grpc UNAVAILABLE: Socket closed`
  that has nothing to do with the compile farm itself and everything to do
  with a missing proxy bypass.
- The appliance is a **single shared allocation** — real-hardware jobs must
  run strictly sequentially, never concurrently, including with other
  users/sessions who may also be active on the same node. Check
  `git status`/`git log`/who else is active before assuming exclusive use.
- When running `ssh cer-usn-01 "..."`, each invocation starts a fresh shell
  in `$HOME` — prefix commands with the right `cd`, or use `git -C <path>`/
  absolute paths, not a bare `cd` you assume persists across calls.

## Verification (how to confirm the program still runs correctly)

1. **Compile-only sanity check** first (cheap, catches CSL errors before
   spending appliance time):
   `cslc` the kernel with `--run-only` disabled — same invocation
   `commands_wse3_iterative.sh` already uses, minus the run step — to make
   sure `layout_bool.csl`/`bool_pe.csl` still compile after `f_spmv` is
   removed.
2. **Run the existing iterative smoke tests end-to-end** and confirm they
   still pass exactly as before the change:
   - `bfs/commands_wse3_iterative.sh`
   - `bfs/commands_wse3_graph500.sh`
   - `bfs/tests/commands_wse2_bfs.sh`
3. **Run `run_bfs.py` on at least one known-good matrix** (whatever small
   test `.mtx` the repo already uses elsewhere in these scripts) and confirm
   its BFS-tree/timing/parent-tracking output is unchanged from a pre-change
   run — diff against `results/sim/bfs_timing.csv` or
   `results/hw/bfs_timing.csv` if a prior run's numbers are available there.
4. **Final grep sweep**: `grep -rn "f_spmv\b\|run_single_spmv\|is_iterative" bfs/`
   should return no hits once done.
5. Report back explicitly if any step fails, rather than silently skipping
   it — in particular, don't assume the appliance is free; if another
   session is actively using it, wait rather than force a concurrent run.
