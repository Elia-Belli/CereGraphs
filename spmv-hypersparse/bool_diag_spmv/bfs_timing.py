""" Shared per-round timing constants/decoders for bool_diag_spmv's
  f_spmv_iter profiling (see src/bool_pe.csl's ts_buf/record_ts()/TS_*
  comment for the on-device side of this). Single source of truth for
  run_bfs.py (which writes bfs_timing.csv) and plot_bfs_timing.py (which
  reads it back) -- previously these lived only in bench_timing.py, with
  plot_bfs_timing.py keeping its own separate, hand-copied phase list that
  could silently drift out of sync.
"""

import os

import numpy as np

# must match bool_pe.csl's TS_* constants / NUM_TS_SLOTS exactly -- these are
# indices into the 10-slot-per-round raw timestamp capture, not phases
# themselves (phases are consecutive-slot differences, see PHASES below).
TS_VBCAST_ISSUE = 0
TS_VBCAST_DONE = 1
TS_COMPUTE_ENTRY = 2
TS_REDUCE_ISSUE = 3
TS_REDUCE_DONE = 4
TS_RELAY_ISSUE = 5
TS_TERM_COL_DONE = 6
TS_TERM_ROW_DONE = 7
TS_TERM_ROW_BCAST_DONE = 8
TS_TERM_COL_BCAST_DONE = 9
NUM_TS_SLOTS = 10

# (name, start slot, end slot) -- each phase is literally end-minus-start of
# two of the raw captures above; relay_total spans all four relay phases at
# once rather than being their sum, as a direct (not accumulated) check.
PHASES = [
    ("visited_bcast", TS_VBCAST_ISSUE, TS_VBCAST_DONE),
    ("vertical_bcast", TS_VBCAST_DONE, TS_COMPUTE_ENTRY),
    ("local_compute", TS_COMPUTE_ENTRY, TS_REDUCE_ISSUE),
    ("reduce", TS_REDUCE_ISSUE, TS_REDUCE_DONE),
    ("local_term_cond", TS_REDUCE_DONE, TS_RELAY_ISSUE),
    ("relay_col_reduce", TS_RELAY_ISSUE, TS_TERM_COL_DONE),
    ("relay_row_reduce", TS_TERM_COL_DONE, TS_TERM_ROW_DONE),
    ("relay_row_bcast", TS_TERM_ROW_DONE, TS_TERM_ROW_BCAST_DONE),
    ("relay_col_bcast", TS_TERM_ROW_BCAST_DONE, TS_TERM_COL_BCAST_DONE),
    ("relay_total", TS_RELAY_ISSUE, TS_TERM_COL_BCAST_DONE),
]

# chronological order == stack order for plot_bfs_timing.py's bars, bottom
# to top -- PHASES minus relay_total (which would double-count its own 4
# relay_* sub-phases if also drawn as its own segment).
LEAF_PHASES = [name for name, _, _ in PHASES if name != "relay_total"]

# the 4-phase termination relay's own sub-phases (see bool_pe.csl's module
# docstring) -- broken out so callers (plot_pe_heatmap.py's --relay) can
# compare just these four on a shared cycle scale, distinct from
# LEAF_PHASES' full-algorithm overview (which intentionally keeps each
# phase's own independent scale, since e.g. local_compute and
# local_term_cond differ by an order of magnitude).
RELAY_PHASES = ["relay_col_reduce", "relay_row_reduce", "relay_row_bcast", "relay_col_bcast"]

# on-device phases that make up the Graph500-style "search time" (see
# GRAPH500_BENCHMARK.md section 3) -- relay_total is used directly instead
# of its own 4 relay_col/row_* sub-phases, to avoid double-counting.
SEARCH_TIME_PHASES = [
    "visited_bcast", "vertical_bcast", "local_compute", "reduce", "local_term_cond", "relay_total",
]

# the two separately-timed h2d brackets (see GRAPH500_BENCHMARK.md section
# 2/3) -- matrix structure (construction-like, one-time) vs seed x
# (per-search).
H2D_PARTS = ["h2d_matrix", "h2d_seed"]

# timestamp.tsc_size_words in bool_pe.csl -- the <time> library's fixed
# [3]u16 timestamp width (see SKILL-LIBRARIES.md's <time> entry).
TSC_WORDS = 3

# WSE clock frequency, for converting search_time_cycles -> seconds for
# GTEPS (GRAPH500_BENCHMARK.md section 5/6).
CLOCK_FREQ_HZ = 875e6


def decode_round_timestamps(ts_hwl_u32, height, width, max_rounds):
  """ts_hwl_u32: (height, width, max_rounds*NUM_TS_SLOTS*3) uint32 array
  fresh off memcpy_d2h (u16 values zero-extended into u32 words, COL_MAJOR
  wire order already resolved into this hwl shape -- same convention as
  device_io.py's other u16 buffers, e.g. extract_parent_result's input).
  The last axis is C-contiguous per-PE in exactly the order bool_pe.csl's
  record_ts() wrote it (round slowest, slot, then word fastest), so a plain
  reshape (no extra transpose) recovers (height, width, max_rounds,
  NUM_TS_SLOTS, 3).

  Returns a (height, width, max_rounds, NUM_TS_SLOTS) int64 array of
  absolute 48-bit timestamps, one per PE per round per capture slot --
  packed low-to-high exactly like cerebras.sdk.sdk_utils.make_u48
  (word0 | word1<<16 | word2<<32)."""
  ts_hwl = ts_hwl_u32.reshape(height, width, max_rounds, NUM_TS_SLOTS, 3)
  w0 = ts_hwl[..., 0].astype(np.int64)
  w1 = ts_hwl[..., 1].astype(np.int64)
  w2 = ts_hwl[..., 2].astype(np.int64)
  return w0 + (w1 << 16) + (w2 << 32)


def read_tic_toc_delta(runner, sym_tsc_start, sym_tsc_end, height, width):
  """Read back tsc_start_buffer/tsc_end_buffer (each PE's own f_tic()/
  f_toc() capture -- see f_enable_tsc()/f_tic()/f_toc() in bool_pe.csl) and
  return the per-PE elapsed-cycle deltas as a flat (height*width,) int64
  array. Same-PE subtraction (this PE's own toc minus this PE's own tic),
  so no cross-PE clock synchronization is needed -- see run_bfs.py's own
  module docstring for why."""
  # local import: only needed here, avoids importing the whole SDK runtime
  # module for callers that only want the pure-python constants above.
  from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
      MemcpyDataType, MemcpyOrder,
  )

  def _read(sym):
    buf_1d = np.zeros(height * width * TSC_WORDS, np.uint32)
    runner.memcpy_d2h(buf_1d, sym, 0, 0, width, height, TSC_WORDS,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    hwl = np.reshape(buf_1d, (height, width, TSC_WORDS), order="F")
    w0 = hwl[..., 0].astype(np.int64)
    w1 = hwl[..., 1].astype(np.int64)
    w2 = hwl[..., 2].astype(np.int64)
    return (w0 + (w1 << 16) + (w2 << 32)).reshape(-1)

  tic = _read(sym_tsc_start)
  toc = _read(sym_tsc_end)
  return toc - tic


def decode_phase_row(ts_hwl_u32, height, width, max_rounds, rounds_completed, verbose=True):
  """Decode one search's ts_buf into per-phase min/max/avg CSV columns
  (semicolon-joined per-round strings, same shape run_bfs.py has always
  logged) plus each phase's straggler-PE-max summed across rounds for
  SEARCH_TIME_PHASES -- the on-device portion of GRAPH500_BENCHMARK.md
  section 3's search_time_cycles (h2d_seed is added by the caller, since
  it's measured outside this function's tsc bracket). Shared by run_bfs.py
  (one search, verbose=True) and run_graph500.py (64 searches, verbose=False
  to avoid flooding stdout with a full phase breakdown per search).

  Returns (row_cols, device_time_cycles, profiled_rounds)."""
  profiled_rounds = min(rounds_completed, max_rounds)
  if rounds_completed > max_rounds:
    print(f"[[ WARNING: BFS ran {rounds_completed} rounds but max_rounds={max_rounds} -- "
          f"only the first {max_rounds} rounds were timestamped; bump max_rounds to "
          "profile the rest ]]")

  ts = decode_round_timestamps(ts_hwl_u32, height, width, max_rounds)
  ts = ts[:, :, :profiled_rounds, :].reshape(height * width, profiled_rounds, NUM_TS_SLOTS)

  row_cols = {}
  phase_max_by_name = {}
  for name, start_slot, end_slot in PHASES:
    cycles = ts[:, :, end_slot] - ts[:, :, start_slot]
    per_round_min = cycles.min(axis=0)
    per_round_max = cycles.max(axis=0)
    per_round_avg = cycles.mean(axis=0)
    phase_max_by_name[name] = per_round_max
    row_cols[f"{name}_min_cycles"] = ";".join(str(int(v)) for v in per_round_min)
    row_cols[f"{name}_max_cycles"] = ";".join(str(int(v)) for v in per_round_max)
    row_cols[f"{name}_avg_cycles"] = ";".join(f"{v:.1f}" for v in per_round_avg)
    if verbose:
      print(f"  {name:>18s}: min={per_round_min.tolist()} "
            f"max={per_round_max.tolist()} avg={np.round(per_round_avg, 1).tolist()}")

  device_time_cycles = sum(int(phase_max_by_name[name].sum()) for name in SEARCH_TIME_PHASES)
  return row_cols, device_time_cycles, profiled_rounds


def decode_pe_phase_cycles(ts_hwl_u32, height, width, max_rounds, rounds_completed):
  """Like decode_phase_row, but keeps the full (height, width) PE-grid shape
  instead of collapsing it to min/max/avg -- for per-PE diagnostics (e.g.
  a heatmap of which PEs are a phase's stragglers, see plot_pe_heatmap.py),
  not the aggregate CSV log.

  Returns (phase_cycles, profiled_rounds): phase_cycles is a dict of
  {phase_name: (profiled_rounds, height, width) int64 array}, one entry per
  PHASES tuple."""
  profiled_rounds = min(rounds_completed, max_rounds)
  ts = decode_round_timestamps(ts_hwl_u32, height, width, max_rounds)
  ts = ts[:, :, :profiled_rounds, :]  # (height, width, profiled_rounds, NUM_TS_SLOTS)

  phase_cycles = {}
  for name, start_slot, end_slot in PHASES:
    cycles = ts[:, :, :, end_slot] - ts[:, :, :, start_slot]  # (height, width, profiled_rounds)
    phase_cycles[name] = np.transpose(cycles, (2, 0, 1))  # (profiled_rounds, height, width)
  return phase_cycles, profiled_rounds


def save_pe_phase_cycles(path, phase_cycles, metadata, structural_grids=None):
  """Save one run's full per-PE-per-round-per-phase cycle grid to a single
  self-contained .npz file, one file per run -- NOT appended across runs the
  way bfs_timing.csv/graph500_searches.csv are. A heatmap needs the cycle
  cost back as a (height, width) grid per phase/round; a CSV would force
  either a wide format that doesn't scale with PE count or a long/tidy
  format that has to be pivoted back into a grid every time it's read,
  neither of which numpy's own binary round-trip needs. `metadata` is a
  dict of small scalars/strings (infile_mtx, pe_grid, source,
  rounds_completed, ...) saved alongside the arrays in the same file --
  read back via load_pe_phase_cycles() in plot_pe_heatmap.py.

  `structural_grids`: optional dict of additional (height, width) 2D
  arrays that aren't per-round timing at all -- e.g. run_bfs.py passes the
  matrix's own per-PE partition counts (local_nnz/local_nnz_cols/
  local_nnz_rows from preprocess_bool.py) here, so plot_pe_heatmap.py's
  sparsity.png can be checked by eye against the timing heatmaps for
  correlation (e.g. does local_compute's imbalance actually track
  local_nnz's imbalance, or is it something else)."""
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  np.savez_compressed(path, **phase_cycles, **(structural_grids or {}), **metadata)


def compute_m_and_gteps(A_coo, device_visited, is_symmetric, search_time_cycles,
                         clock_freq_hz=CLOCK_FREQ_HZ):
  """m (edges traversed) + GTEPS for one search, per GRAPH500_BENCHMARK.md
  sections 4/5. A_coo is the caller's A_csr.tocoo() -- passed in rather than
  recomputed here since run_graph500.py calls this once per search against
  the SAME static matrix.

  is_symmetric picks the counting convention (see GRAPH500_BENCHMARK.md
  section 4): Graph500's own undirected dedup rule, or -- for a directed
  --infile_mtx -- the "edges out of every visited source" convention, still
  real algorithmic work but not a Graph500-spec-comparable number."""
  if is_symmetric:
    m = int(np.sum(device_visited[A_coo.row] & (A_coo.col <= A_coo.row)))
    m_convention = "undirected_dedup"
  else:
    m = int(np.sum(device_visited[A_coo.col]))
    m_convention = "directed_source_visited"
  search_time_seconds = search_time_cycles / clock_freq_hz
  gteps = m / search_time_seconds / 1e9 if search_time_seconds > 0 else float("nan")
  return m, m_convention, search_time_seconds, gteps
