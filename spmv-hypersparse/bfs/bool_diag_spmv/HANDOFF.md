# Handoff — bool_diag_spmv real-hardware BFS validation

Status as of 2026-07-29. Written for whoever picks this up next (including
future-me). Read this before re-running anything — several results that
*look* published/valid are known-stale as of this writing (see "Data you
should NOT trust" below).

## TL;DR

This session found and fixed two real, independent hardware/correctness
bugs (h2d gRPC ceiling; directed-BFS computing the wrong reachability
direction), built the RMAT scale×grid sweep, and got partway through
re-validating SNAP graphs under the corrected methodology before running
out of runway. **The SNAP side is the unfinished part** — pokec, topcats,
and livejournal all need to be regenerated + rebalanced + rerun under the
direction fix; only berkstan got that treatment, and it's now blocked by a
different, real ceiling once correctly balanced.

## What's actually fixed and verified

1. **h2d gRPC ~2GiB message-size ceiling** (matrix-structure upload) —
   `device_io.py::memcpy_h2d_chunked`, commit `70b5abc`. Root cause: the
   vendor SDK's own internal chunker has an off-by-protobuf-envelope bug
   (chunks to exactly `MAX_MESSAGE_LENGTH` with zero margin for the
   enclosing message). Fix chunks our own calls along the PE-row axis with
   real margin (1.5GiB default) so the vendor's buggy chunker never
   triggers. Verified: regression-safe for small cases (byte-identical
   single-call path below the threshold), and worked end-to-end on
   berkstan *before* the direction fix was discovered (see below for why
   that specific success is now stale). Full root-cause math in
   `ERRORS.md` #4.

2. **Mandatory identity-preserving balancing** — `util/analyze.cpp`,
   commit `c08c84c`. Removed the independent-row/column-permutation code
   path entirely; `--symmetric` or `--shared-perm` is now required on
   every invocation, so a balanced matrix's provenance (does it preserve
   vertex identity for a canonical source?) is never ambiguous again.

3. **Directed BFS was computing ancestor-, not descendant-reachability**
   — `graph_loader.py`, commit `e1dc2fa`, `ERRORS.md` #15. This is the
   big one. `bool_pe.csl`'s kernel natively computes "who points at the
   frontier" given a row=src matrix convention; the SNAP edge-list loader
   fed exactly that convention, so every directed-graph BFS "from source
   X" was actually finding X's ancestors, not its descendants. Invisible
   until directly checked against independent ground truth (a published
   reference table's expected explored-edge count for berkstan's
   canonical root, vertex 546279). Fixed by feeding the transpose instead.
   Verified mechanistically (traced the kernel's actual SpMV walk
   direction, confirmed balancing/matrix-writing were never wrong, matched
   the corrected number to true forward reachability exactly: 459,847).
   **RMAT and orkut are unaffected** (both structurally symmetric,
   direction-independent).

4. **RMAT scale×grid 2D sweep** — `rmat_grid_sweep.sh`, complete for
   s10–s20 (s21 abandoned, see below). Every (scale, grid) combo that
   compiles is in `results/hw/bfs_timing.csv` and the published artifact's
   heatmap.

5. Assorted smaller fixes/cleanups: deduped `bfs_timing.csv` (kept most
   recent per input), removed stale CSV rows/plots that carried a garbage
   `h2d_matrix` timing stat (see open item below), before/after balance
   poster figures (`plots/plot_balance_before_after.py`).

## Data you should NOT trust right now

- **Every currently-published SNAP result except the RMAT heatmap is
  suspect or stale.** Specifically:
  - **berkstan**: no valid real-hardware row exists at all right now. The
    two rows that existed (source=0, and canonical-root
    546279→353938) were both computed against the *pre-direction-fix*
    matrix and were deleted from `bfs_timing.csv` (see commit `e1dc2fa`).
    The corrected, direction-fixed matrix is on disk
    (`data/snap_berkstan.mtx`, `.balanced750x750.mtx/.operm`, regenerated
    2026-07-29) but **cannot currently compile at 750×750** — see next
    section.
  - **pokec**: the one remaining SNAP row in `bfs_timing.csv`
    (`snap_pokec.balanced750x750.mtx`, source=1178437, gteps=0.2338)
    **predates the direction fix** and has not been re-verified. It may
    still be correct (pokec's degree distribution might not expose the
    bug the same way berkstan's did — untested), or it may be silently
    computing ancestor-reachability like berkstan was. Don't cite this
    number without redoing it first.
  - **topcats, livejournal**: still fail to compile (PE static-memory
    ceiling, `ERRORS.md` #8) — this predates and is independent of the
    direction fix, so their status is unchanged, but their *source data*
    (raw `.mtx`) was never regenerated under the fix either.
  - **orkut**: still can't get past compile (linker file-vanished flake,
    `ERRORS.md` #12, now 3/3 failures across two sessions) — never even
    reached the point where direction would matter.
- The published artifact
  (`https://claude.ai/code/artifact/c150d76e-d604-42b0-8b17-a509fe2d29c0`)
  has been updated to reflect all of this honestly (berkstan back to
  "fail-compile", pokec flagged "UNVERIFIED", a visible correctness note)
  — it is not stale, but the underlying SNAP *data* it can draw on is thin
  until the regeneration below happens.

## Immediate next steps, in priority order

1. **Regenerate + rebalance + rerun pokec under the direction fix.**
   `benchmarks/snap_to_mtx.py data/snap/soc-pokec-relationships.txt.gz
   data/snap_pokec.mtx` (now direction-corrected automatically, no flag
   needed — the fix is in the shared loader), then
   `util/analyze --matrix data/snap_pokec.mtx --shared-perm --fabx 750
   --faby 750 --omatrix data/snap_pokec.balanced750x750.mtx --operm
   data/snap_pokec.balanced750x750.operm`, then compile+run same as
   before. Compare `max_local_nnz` printed during balancing against the
   old value to know up front whether it'll even fit (berkstan's grew
   1102→3481 and stopped fitting; pokec might behave differently — its
   degree distribution is different, this needs to actually be measured,
   not assumed).
2. **Same regeneration for topcats/livejournal**, even though they're
   expected to still fail to compile (PE-memory ceiling, independent of
   direction) — do it anyway so their on-disk `.mtx` files aren't
   silently pre-fix stale, in case a future bigger-grid fix makes them
   relevant again.
3. **Decide what to do about berkstan.** It's real and correctly balanced
   now, but doesn't fit at 750×750 (`max_local_nnz=3481`). Options, none
   attempted yet:
   - Try a smaller grid ladder position where the per-PE working set
     might fit despite lower total parallelism (unlikely to help much,
     since a smaller square grid generally makes per-PE skew *worse*, not
     better — but not verified either way).
   - This is fundamentally the same open problem as `ERRORS.md` #8
     (topcats/livejournal) — a coarser or larger grid would help, but the
     kernel is square-grid-only by design (see the "why not rectangular
     grids" discussion earlier this session — WSE-3's real fabric is
     762×1172, and 750×750 already uses nearly the full width while
     leaving ~35% of the height unused; a rectangular-grid-capable
     redesign would need to touch the diagonal-targeted broadcast/reduce
     scheme, not a small change).
4. **orkut**: probably still blocked (3/3 linker-flake failures across two
   sessions, judged infra-side and size-correlated per explicit prior
   user decision — don't retry blindly again without a reason to think
   something changed compile-farm-side).
5. **Cosmetic, low-priority**: `ERRORS.md` #4b, the `h2d_matrix` cycle-
   count timing stat is garbage (likely 32-bit cycle-counter wraparound)
   independent of everything else here — doesn't affect GTEPS/correctness,
   just that one diagnostic print. Not investigated further this session.

## Key files

- `bfs/bool_diag_spmv/ERRORS.md` — the full compendium (15 numbered
  issues + a summary table). Read this for the complete history/root
  causes; this handoff only summarizes what's actionable right now.
- `bfs/bool_diag_spmv/device_io.py` — `memcpy_h2d_chunked` (h2d fix)
- `bfs/bool_diag_spmv/graph_loader.py` — `_load_edgelist` (direction fix)
- `util/analyze.cpp` — mandatory `--symmetric`/`--shared-perm` balancing
- `bfs/bool_diag_spmv/rmat_grid_sweep.sh` — the RMAT 2D sweep (done)
- `bfs/bool_diag_spmv/results/hw/bfs_timing.csv` — real-hardware results,
  currently 66 RMAT rows + 1 (unverified) pokec row
- `bfs/bool_diag_spmv/plots/plot_grid_scale_heatmap.py`,
  `plot_balance_before_after.py` — poster/report figure generators
- Published artifact:
  `https://claude.ai/code/artifact/c150d76e-d604-42b0-8b17-a509fe2d29c0`
  (redeploy the same file path to update — see any recent turn in this
  session's transcript for the exact HTML structure/DATA array format)

## Environment notes (easy to get wrong)

- Real hardware runs happen on `cer-usn-01` via ssh; the repo is NFS-
  mirrored to the local machine used for editing, but `git`/`cslc`/the
  appliance SDK only exist on `cer-usn-01`.
- Always `source /home/elia/cs_appliance_sdk/bin/activate` before any
  appliance-mode Python invocation — bare `python` fails instantly
  otherwise.
- Always `export no_proxy='10.125.8.2,.cerebras.internal,localhost,127.0.0.1'`
  (and `NO_PROXY` to match) before compiling/running — otherwise compile
  submission fails with a generic-looking `grpc UNAVAILABLE: Socket
  closed` that has nothing to do with the actual compile-farm and
  everything to do with a missing proxy bypass (cost real time
  misdiagnosing this earlier in the session — don't repeat that).
- The appliance is a **single shared allocation** — real-hardware jobs
  must run strictly sequentially, never concurrently, including with
  other users/sessions who may also be active on the same node.
- **Another session was concurrently modifying SNAP prep scripts**
  (`benchmarks/download_snap_graphs.sh`, `benchmarks/prep_snap_v3.sh`,
  `bfs/bool_diag_spmv/run_snap_sweep.sh`) as of this writing, uncommitted.
  Check `git status`/`git log` before assuming you have the field to
  yourself, and don't blindly overwrite their in-progress work.
- For long/multi-paragraph git commit messages, use `git commit -F
  <file>` (write the message to a file first) rather than a bash heredoc
  piped through `-m "$(cat <<'EOF' ... EOF)"` — the heredoc form broke at
  least once this session on a long message for unclear quoting reasons.
- When running `ssh cer-usn-01 "..."`, remember each invocation starts a
  fresh shell in `$HOME` — either prefix commands with `cd
  /home/elia/CereGraphs/spmv-hypersparse &&`, or use `git -C <path>`
  /absolute paths, not a bare `cd` you assume persists across calls.

## Addendum from the concurrent session (2026-07-29, later same day)

This is the "another session ... concurrently modifying SNAP prep
scripts" referred to above — reporting back so neither session loses the
other's context.

- Added **as-Skitter** and **cit-Patents** to the SNAP pipeline
  (`benchmarks/download_snap_graphs.sh`, `benchmarks/prep_snap_v3.sh`,
  `bfs/bool_diag_spmv/run_snap_sweep.sh` — all three now also accept
  optional name args, e.g. `run_snap_sweep.sh skitter patents`, matching
  the existing convention). New `run_snap_sweep_screen.sh` wrapper launches
  a detached `screen` session so long compile+run jobs survive
  independently of whatever ssh/tool session started them.
- **Both as-Skitter and cit-Patents fail to compile at 750x750** — but
  it's issue **#8** (PE static-memory ceiling), not #12. See #8's own
  updated entry for the full diagnosis and a new "don't misdiagnose as
  #12" note — the "cannot open ...cslc-<hash>.o" message looks like #12
  at a glance but isn't; check for `.bss`/task-table overflow lines
  earlier in the same log first.
- Per this session's read of your pokec warning ("don't cite without
  redoing it"): regenerated `data/snap_pokec.mtx` from raw (picked up your
  e1dc2fa direction fix automatically, no flag needed), rebalanced
  (`--shared-perm`), recompiled (succeeded, 380s, no PE-memory errors) —
  in progress running as of this writing, check `bfs_timing.csv` for a
  fresh pokec row with today's date to confirm it landed.
- **Did not touch berkstan** — left the "what to do about it not fitting
  post-fix" decision to you, per your own next-steps list.
