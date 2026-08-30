# Knowledge

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


