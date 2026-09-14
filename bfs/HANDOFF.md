# Knowledge

## Next session (start here)

RMAT s23 and s24 @ 750x750 are now **fully verified real-hardware
high-water marks** (docs/ERRORS.md #27/#28) — s23 with a full scipy
cross-check (`0/8,388,750` mismatches), s24 with compile+execution
verified (scipy cross-check auto-skipped above RMAT-s20 scale by
design, see below). Current state, most-to-least reachable:

| scale | peak/PE | vs ceiling | real hardware | scipy-verified |
|---|---|---|---|---|
| s22 | 20,848B | 58% margin | yes | yes |
| s23 | 26,320B | 46% margin | yes | yes |
| s24 | 36,768B | 25% margin | yes | no (auto-skipped, see #28) |
| s25 | ~57,000B (modeled) | **+16% over** | no — `cslc` fails for real | n/a |

Two things remain open, both flagged rather than fixed:

1. **RMAT s25 @ 750x750 itself, still structurally blocked.** A fitted
   memory model (bytes ≈ 14,573 + 1.79·`max_local_nnz` + 8.21·
   `max_local_nnz_rows` + 20.08·⌈`blk`/32⌉, fit on 4 real
   `cs_readelf -m` points, cross-validated against a 5th to <0.05%
   error) puts s25 at **~57,000B — ~16% over the 49,152B ceiling**, and
   identifies exactly why: the `x_bitmap`/`visited_bitmap`/`y_bitmap`/
   `y_bitmap_reduced`/`popcnt_scratch` buffers (all scale with `blk`
   alone, not edge count) account for **~49% of total memory at s25's
   scale** — the single biggest term, and the one #24/#26/#27's fixes
   never touched (they all targeted `max_local_nnz`/`max_local_nnz_rows`-
   scaled buffers, or host-side cost). A real fix would need to shrink
   the bitmap representation itself (e.g. a sparser frontier encoding),
   not just clean up existing buffers. Behind that: the `parent_values`
   d2h payload at s25 scale (~2.71GiB) still exceeds the 2GiB gRPC
   ceiling (#1/#4) with no d2h-chunking fix in place. See the **Sparse
   Parents Ledger** artifact (published this session — ask the user for
   the link, or check `/artifacts`, if picking this back up) for the
   full per-scale/per-buffer breakdown, live and current as of s19-s25.
2. **The scipy correctness-check's own memory cost, not fixed, only
   worked around.** #28 found a THIRD host-memory wall, independent of
   `preprocess()` (#26) and the WSE-3 ceiling (#27): `breadth_first_order()`
   + rebuilding a transposed CSR copy of `A` OOMs on its own at RMAT s24
   scale (confirmed via `dmesg`, ~99.8GiB anon-rss). Fixed pragmatically
   by auto-disabling the check above RMAT-s20 scale (`n > 1,100,000`,
   `--force-scipy` to override) — s24 itself has never actually been
   scipy-verified as a result. If real correctness confirmation at s24+
   scale is wanted, the scipy-check code itself needs the same kind of
   treatment `preprocess_bool.py` got in #26/#27 (free `A_csr` before/
   while building `A_fwd`, or a cheaper validation approach entirely) —
   not yet scoped.

If s25 is revisited: start from the Ledger's own buffer breakdown, not
from scratch — it already identifies which buffers to target. If a
scipy-verified s24 (or a scipy-verified s25, should the memory-overflow
problem ever get solved) is wanted, that's a separate, smaller task:
shrink the cross-check's own footprint.

## Recent history

Bottom-up-only BFS redesign (docs/ERRORS.md #21) landed 2026-09-07 through
2026-09-11: a real `preprocess()` axis-transpose bug was found and fixed
(#22), then dense (`parent_resolve_variant==0`) was removed entirely (#23)
-- the kernel now has a single parent-resolution path. #24 (still
2026-09-11) then removed the on-device parent-aggregation relay
entirely, moving the per-row combine host-side -- real memory win once
measured (40,736B->16,384B at s19 512x512, 47,120B->16,672B at s20
750x750), at the accepted cost of a much larger d2h transfer.

2026-09-11 (later): a user observation caught that the host-side combine
had NO timer at all (#25, fixed: `host_parent_combine_seconds`, new plot
bars in both `plot_bfs_timing.py` and `plot_bfs_timing_poster.py`). Then,
in pursuit of RMAT s25 @ 750x750 (2x s22's scale): `parent_values` was
shrunk from a global u32 vertex id to a local u16 column offset (#26,
on-device + `device_io.py` + both `run_bfs*.py` scripts), landing real,
verified new peak-memory numbers -- s19: 16,384B->16,192B; s20:
16,672B->16,384B; s22: 21,488B->20,848B (real hardware confirmed clean at
s22: 0/4,194,750 mismatches, 164 GTEPS excl. transfer). Chasing s25 then
surfaced a host-side ceiling: a 100GiB per-user cgroup memory limit on
`cer-usn-01`/`02`/`03` (unrelated to the WSE-3 per-PE ceiling). Several
real `preprocess_bool.py`/`graph_loader.py` memory fixes landed that
session but didn't fully close the gap -- left as a "next session" lead:
a whole second full-`nnz` sort over a separately-built CSC representation
was computing `local_nzrows`, and this looked derivable from data the
CSC-ordered pass already had.

2026-09-12: that lead was implemented -- and turned out better than
expected. Tracing every LIVE caller showed `local_nzrows`/
`max_local_nnz_rows` was read by NEITHER `run_bfs.py` nor
`run_bfs.appliance.py` at all -- so instead of deriving it more cheaply,
it was deleted outright (#27): `preprocess()`'s `csrRowPtr`/`csrColInd`
parameter pair is gone, and callers no longer build a second scipy
sparse representation (`A_csc`) at all. Real win at every scale (RMAT
s22's own `preprocess()` cost dropped 13.07GB->9.67GB, ~26% cheaper). At
RMAT s25 scale, `preprocess()` completed for the first time (86.29GB
peak RSS, under the 100GiB ceiling) -- but the actual `cslc` compile then
failed with a genuine `ran out of PE memory` error: a DIFFERENT, harder,
on-device wall.

2026-09-14 (this session): pushed the fallback plan through. RMAT s23 and
s24 both compiled and ran clean on real WSE-3 hardware (s23: 26,320B,
0/8,388,750 mismatches, 224.5 GTEPS; s24: 36,768B, 273.99 GTEPS) --
genuine new high-water marks. s24's own real-hardware run then surfaced a
THIRD host-memory wall (#28): the scipy correctness-check itself OOMs at
this scale (dmesg-confirmed, ~99.8GiB anon-rss), independent of both
#26's `preprocess()` wall and #27's on-device wall. Fixed by auto-
disabling the scipy check above RMAT-s20 scale by default (`--force-scipy`
to override) -- verified both at smoke scale (unchanged) and via a clean
RMAT s24 real-hardware run with zero flags. A linear memory model (fit
on 4 real `cs_readelf -m` totals, cross-validated to <0.05% error against
a 5th) was derived and used to explain RMAT s25's failure precisely:
~57,000B projected, ~16% over ceiling, with the `blk`-scaled bitmap
buffers (not the `max_local_nnz`/`max_local_nnz_rows`-scaled ones every
prior fix targeted) identified as the single largest contributor
(~49% of total at s25 scale) -- published as the **Sparse Parents
Ledger** artifact (interactive stacked-bar breakdown + full history
timeline; ask the user for the link if picking this up fresh).

See `bfs/HANDOFF_BOTTOMUP_WIP.md` for how the original #22 bug was found
(historical record) and docs/ERRORS.md #21-28 for the full story,
including every current number.

## Key files

- `docs/ERRORS.md` — the full compendium (28 numbered issues + a summary
  table). Read this for the complete history/root causes; this handoff
  only summarizes what's actionable right now.
- `bfs/implementation/preprocess_bool.py` — the RMAT-s25-motivated memory
  fixes (#26) and the CSR/CSC-redundancy elimination (#27, done)
- `bfs/implementation/graph_loader.py` — `_load_edgelist` (direction fix);
  `load_graph()`'s `.data` dtype shrink (#26)
- `bfs/implementation/device_io.py` — `memcpy_h2d_chunked`/
  `prepare_h2d_chunked`/`send_h2d_chunked` (h2d chunking fix, #4 -- the
  pattern a future d2h chunking fix would mirror, see "Next session");
  `extract_parent_result()` (host-side parent combine, #24/#26)
- `bfs/scripts/run_bfs.py`/`run_bfs.appliance.py` —
  `_SCIPY_CHECK_AUTO_DISABLE_N`/`--force-scipy` (#28, the scipy-check
  memory-cost workaround)
- `.claude_scratch/probe_preprocess.py`,
  `.claude_scratch/probe_preprocess_instrumented.py` — standalone
  `preprocess()` regression-test/probe scripts built this session
  (outside git, on the shared NFS home — reusable directly; the
  instrumented one prints a peak-RSS checkpoint after every pipeline
  stage, useful for isolating where a future OOM happens)
- `util/analyze.cpp` — mandatory `--symmetric`/`--shared-perm` balancing
- `bfs/scripts/rmat_grid_sweep.sh` — the RMAT 2D sweep (done)
- `bfs/results/hw/bfs_timing.csv` — real-hardware results (see the file
  for current row count)
- `bfs/plots/plot_grid_scale_heatmap.py`, `plot_balance_before_after.py`
  — poster/report figure generators
- `bfs/plots/plot_bfs_timing.py`, `plot_bfs_timing_poster.py` — per-run
  timing charts, now with a "combine" bar (#25)
- **Sparse Parents Ledger** (published Artifact, 2026-09-14) — interactive
  per-scale memory breakdown (bitmaps/row bookkeeping/column data/fixed
  overhead) + the s19 project-history timeline; the fitted memory model
  lives in its own `<script>`, reusable for future scale estimates

## Environment notes (easy to get wrong)

- Real hardware runs, `cslc`, and the appliance SDK live on `cer-usn-01`
  via ssh; the repo is NFS-mirrored to the local machine used for editing.
  `git` is on **`cer-usn-02`** instead (verified directly — cer-usn-01 has
  no `git` on PATH), correcting this file's own previous claim that it was
  on cer-usn-01 too.
- **100GiB per-user cgroup memory limit** on `cer-usn-01`/`02`/`03` (`cat
  /sys/fs/cgroup/memory/user.slice/user-<uid>.slice/memory.limit_in_bytes`
  == `107374182400` — confirmed identical on all three, a cluster-wide
  policy). This is a HOST-SIDE ceiling, independent of the WSE-3 per-PE
  static-memory ceiling everything else in docs/ERRORS.md targets — a
  large host-side Python step can get OOM-killed here long before ever
  reaching the compiler OR the appliance. Watch for a plain `Killed` with
  no Python traceback (that's the tell — the process didn't raise, the
  kernel killed it) and check `dmesg | grep -i oom` to confirm. **It's
  per-USER, not per-process** — concurrent host-side jobs (e.g. a
  `util/analyze` balance running alongside a `run_bfs.appliance.py`
  correctness run) share the same 100GiB budget and can combine to OOM
  even when neither alone would (confirmed the hard way this session --
  don't run two memory-heavy host jobs at once, even unrelated ones).
  Three independent things have now hit this same ceiling: `preprocess()`
  at RMAT-s25 scale (#26, fixed), nothing on the WSE-3 side itself (#27
  is a separate, on-device ceiling), and the scipy correctness-check at
  RMAT-s24 scale (#28, worked around by auto-disabling above s20 scale).
- **The SDK's real wire payload for memcpy_h2d/memcpy_d2h is always 4
  bytes/element, regardless of the `data_type` kwarg** (documented in
  `device_io.py`, confirmed against the installed SDK client) — a u16
  device-side transfer still counts as 4 bytes/element for gRPC
  message-size purposes. Relevant any time a per-tile array is large
  (`max_local_nnz_rows`-sized or bigger) at a big grid — the 2,147,482,624-
  byte ceiling (#1/#4) is closer than the raw device-side byte count
  would suggest.
- `MemcpyDataType.MEMCPY_16BIT`'s host-buffer quirk: the SDK requires the
  HOST-side numpy buffer to stay `uint32` regardless of the DEVICE
  symbol's actual declared width (`u16`, say) — only `data_type=...`
  controls the wire/transfer width. Passing a narrower host buffer throws
  `RuntimeError: Internal data type of any memcpy_d2h()/memcpy_h2d()
  operation should be 32 bit` at runtime, with no hint that the fix is
  about the HOST buffer's dtype, not the device symbol's. This
  codebase's own `rounds_completed`/`mat_row_idx_buf`/`ts_buf`/
  `nf_history` transfers already follow this convention (check any of
  those for the pattern) — easy to miss when adding a new u16 device
  buffer.
- Always `source /home/elia/cs_appliance_sdk/bin/activate` before any
  appliance-mode Python invocation — bare `python` fails instantly
  otherwise. For a LOCAL/free compile-only (no appliance job, just
  `cslc` + `cs_readelf -m`), use `cs_python` from `/home/elia/cs_sdk-2.10`
  instead (needs `singularity`/`apptainer` on PATH — present on
  `cer-usn-01` — the `cs_appliance_sdk` venv alone lacks
  `cerebras.sdk.runtime` and fails with `ModuleNotFoundError`).
- Always `export no_proxy='10.125.8.2,.cerebras.internal,localhost,127.0.0.1'`
  (and `NO_PROXY` to match) before compiling/running — otherwise compile
  submission fails with a generic-looking `grpc UNAVAILABLE: Socket
  closed` that has nothing to do with the actual compile-farm and
  everything to do with a missing proxy bypass (cost real time
  misdiagnosing this earlier in the session — don't repeat that).
- The appliance is a **single shared allocation** — real-hardware jobs
  must run strictly sequentially, never concurrently, including with
  other users/sessions who may also be active on the same node. Check
  `csctl get job` immediately before submitting, every time — another
  user's job can appear between an earlier check and the actual submit.
- A background shell job (compile, balance, appliance run) launched via
  `ssh cer-usn-01 "..."` **survives a Claude Code session ending** — the
  remote process keeps running on its own, but the local log file you
  were redirecting its output to (under this session's own scratchpad)
  does not persist across sessions. If a job seems to have vanished after
  a session boundary, check `ps aux` on the remote host directly before
  assuming it was lost — it may just need a fresh log-capturing wrapper
  (or, for a real-hardware run, a plain re-run against the still-valid
  `artifact_path.json`, which recompiles for free from cache).
- For long/multi-paragraph git commit messages, use `git commit -F
  <file>` (write the message to a file first) rather than a bash heredoc
  piped through `-m "$(cat <<'EOF' ... EOF)"` — the heredoc form broke at
  least once this session on a long message for unclear quoting reasons.
- When running `ssh cer-usn-01 "..."`, remember each invocation starts a
  fresh shell in `$HOME` — either prefix commands with `cd
  /home/elia/CereGraphs &&`, or use `git -C <path>`/absolute paths, not a
  bare `cd` you assume persists across calls.
