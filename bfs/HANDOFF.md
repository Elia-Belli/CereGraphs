# Knowledge

## Recent history

Bottom-up-only BFS redesign (docs/ERRORS.md #21) landed 2026-09-07 through
2026-09-11: a real `preprocess()` axis-transpose bug was found and fixed
(#22), then dense (`parent_resolve_variant==0`) was removed entirely (#23)
-- the kernel now has a single parent-resolution path. Mid-scale + real
hardware re-verification (RMAT s19 512x512, blk=1024 -- the config #8/#17
originally documented as failing to compile) confirmed clean on 2026-09-11:
compiles and runs correctly on real WSE-3 hardware, 0/524288 mismatches,
peak per-tile memory down to 40,736B (from 46,032B pre-#21). See
`bfs/HANDOFF_BOTTOMUP_WIP.md` for how the original bug was found
(historical record) and docs/ERRORS.md #21-23 for the full story,
including current numbers.

## Key files

- `docs/ERRORS.md` — the full compendium (23 numbered issues + a summary
  table). Read this for the complete history/root causes; this handoff
  only summarizes what's actionable right now.
- `bfs/implementation/device_io.py` — `memcpy_h2d_chunked` (h2d fix)
- `bfs/implementation/graph_loader.py` — `_load_edgelist` (direction fix)
- `util/analyze.cpp` — mandatory `--symmetric`/`--shared-perm` balancing
- `bfs/scripts/rmat_grid_sweep.sh` — the RMAT 2D sweep (done)
- `bfs/results/hw/bfs_timing.csv` — real-hardware results (see the file
  for current row count)
- `bfs/plots/plot_grid_scale_heatmap.py`, `plot_balance_before_after.py`
  — poster/report figure generators

## Environment notes (easy to get wrong)

- Real hardware runs, `cslc`, and the appliance SDK live on `cer-usn-01`
  via ssh; the repo is NFS-mirrored to the local machine used for editing.
  `git` is on **`cer-usn-02`** instead (verified directly — cer-usn-01 has
  no `git` on PATH), correcting this file's own previous claim that it was
  on cer-usn-01 too.
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
- For long/multi-paragraph git commit messages, use `git commit -F
  <file>` (write the message to a file first) rather than a bash heredoc
  piped through `-m "$(cat <<'EOF' ... EOF)"` — the heredoc form broke at
  least once this session on a long message for unclear quoting reasons.
- When running `ssh cer-usn-01 "..."`, remember each invocation starts a
  fresh shell in `$HOME` — either prefix commands with `cd
  /home/elia/CereGraphs &&`, or use `git -C <path>`/absolute paths, not a
  bare `cd` you assume persists across calls.


