""" Shared per-round timing constants/decoders for bool_diag_spmv's
  f_spmv_iter profiling (see src/bool_pe.csl's ts_buf/record_ts()/TS_*
  comment for the on-device side of this). Single source of truth for
  run_bfs.py (which writes bfs_timing.csv) and plot_bfs_timing.py (which
  reads it back) -- previously these lived only in bench_timing.py, with
  plot_bfs_timing.py keeping its own separate, hand-copied phase list that
  could silently drift out of sync.

  This file previously also computed a skew-adjustment (compute_skew_adjusted,
  PHASE_GROUP_AXIS/PHASE_ROOT_KIND/PHASE_MID_ONLY) that tried to separate
  cross-PE wait time from real fabric transit for each individual
  communication phase (visited_bcast/vertical_bcast/reduce/relay). That
  machinery was found unreliable and removed entirely -- see
  docs/GRAPH500_BENCHMARK.md. The replacement methodology is simple and robust:
  compute = local_compute + local_term_cond + transpose (each its own tsc
  bracket, max across PEs); device_time = the whole-run round_trip_start_buffer
  -> round_trip_done_buffer span (excl. parent_resolve, max across PEs);
  communication = device_time - compute. Per round, communication = round_time
  - round_compute, where round_time is the same non-decomposed round span
  this file already computed (compute_round_summary, never part of the
  removed skew machinery) -- see check_round_vs_total_communication() for the
  sanity check tying the two together.
"""

import os

import numpy as np

# must match bool_pe.csl's TS_* constants / NUM_TS_SLOTS exactly -- these are
# indices into the 6-slot-per-round raw timestamp capture, not phases
# themselves (phases are consecutive-slot differences, see PHASES below).
TS_VBCAST_ISSUE = 0
TS_COMPUTE_ENTRY = 1
TS_REDUCE_ISSUE = 2
TS_REDUCE_DONE = 3
TS_RELAY_ISSUE = 4
TS_TERM_COL_BCAST_DONE = 5
NUM_TS_SLOTS = 6

# (name, start slot, end slot) -- each phase is literally end-minus-start of
# two of the raw captures above. Only the two REAL local-work phases survive
# here (real per-PE-varying local work, not a collective/wait) -- every
# communication-phase boundary (visited_bcast/vertical_bcast/reduce/relay)
# was removed along with the skew-adjustment machinery that was its only
# consumer.
PHASES = [
    ("local_compute", TS_COMPUTE_ENTRY, TS_REDUCE_ISSUE),
    ("local_term_cond", TS_REDUCE_DONE, TS_RELAY_ISSUE),
]

# the two separately-timed h2d brackets (see docs/GRAPH500_BENCHMARK.md section
# 2/3) -- matrix structure (construction-like, one-time) vs seed x
# (per-search).
H2D_PARTS = ["h2d_matrix", "h2d_seed"]

# timestamp.tsc_size_words in bool_pe.csl -- the <time> library's fixed
# [3]u16 timestamp width (see SKILL-LIBRARIES.md's <time> entry).
TSC_WORDS = 3

# WSE clock frequency, for converting search_time_cycles -> seconds for
# GTEPS (docs/GRAPH500_BENCHMARK.md section 5/6).
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


def compute_round_summary(round_start, round_end):
  """round_time: per round, max over PEs of (that round's
  TS_TERM_COL_BCAST_DONE - TS_VBCAST_ISSUE) -- justified because the
  termination relay forces every PE to agree on nz_total before ANY PE
  can issue the next round's visited_bcast, so this genuinely is a hard
  synchronization boundary. Never part of the removed skew-adjustment
  machinery -- this is the same robust, non-decomposed per-round span this
  file has always computed.

  `round_start`/`round_end`: (profiled_rounds, height, width) int64 arrays,
  the raw TS_VBCAST_ISSUE/TS_TERM_COL_BCAST_DONE grids from
  decode_pe_phase_cycles.

  Returns round_duration_cycles: (profiled_rounds,) int64 array."""
  return (round_end - round_start).max(axis=(1, 2))


def check_round_vs_total_communication(round_duration_cycles, local_compute_max_cycles,
                                        local_term_cond_max_cycles, device_time_cycles,
                                        transpose_max_cycles, verbose=True):
  """Sanity check tying the per-round split to the whole-run split: per
  round, communication = round_time - round_compute (round_compute =
  local_compute_max + local_term_cond_max for that round); summed across
  rounds, this should be close to device_time - total_compute (total_compute
  = sum of local_compute/local_term_cond across rounds, + transpose, a
  one-time cost). The two sides are measured two structurally different
  ways -- a sum of independently-maxed per-round stragglers vs. one
  whole-run per-PE span, then maxed once -- so exact equality isn't
  expected (summed independent maxes tend to run a bit HIGHER, since
  different rounds' stragglers are rarely the same PE), but a large
  mismatch is a real signal something is off, not just normal variance.

  Returns (delta, tolerance) so the caller can decide whether to log-only
  or raise; also prints a warning itself when the tolerance is exceeded."""
  round_compute = local_compute_max_cycles + local_term_cond_max_cycles
  round_communication = np.clip(round_duration_cycles - round_compute, 0, None)
  total_compute = int(round_compute.sum()) + int(transpose_max_cycles)
  implied_communication = device_time_cycles - total_compute
  summed_communication = int(round_communication.sum())
  delta = summed_communication - implied_communication
  tolerance = max(int(0.05 * abs(implied_communication)), 1000)
  if verbose:
    print(f"  consistency check: sum(round_time - round_compute)={summed_communication} vs "
          f"device_time - total_compute={implied_communication} (delta={delta}, "
          f"tolerance=+/-{tolerance})")
  if abs(delta) > tolerance:
    print(f"[[ WARNING: round-vs-total communication consistency check exceeded tolerance -- "
          f"delta={delta}, tolerance=+/-{tolerance}. Could be a genuinely skewed run or a real "
          f"accounting bug -- investigate before trusting this row's numbers blindly. ]]")
  return delta, tolerance


def decode_phase_row(ts_hwl_u32, height, width, max_rounds, rounds_completed,
                      round_trip_cycles, verbose=True):
  """Decode one search's ts_buf into per-round CSV columns for the two REAL
  local-work phases (local_compute, local_term_cond) plus round_duration_cycles
  (round_time, see compute_round_summary). `device_time_cycles` is
  total_runtime_cycles (round_trip_cycles.max()) -- the whole-run
  round_trip_start_buffer -> round_trip_done_buffer span, excluding
  parent_resolve, always correct regardless of max_rounds truncation (ts_buf
  simply has no slot for any round >= max_rounds; round_trip_cycles doesn't
  depend on ts_buf at all).

  `round_trip_cycles`: per-PE (height*width,) int64 array from
  read_tic_toc_delta(round_trip_start_buffer, round_trip_done_buffer) --
  the caller's job to read back (same pattern as h2d/d2h/transpose/
  parent_resolve).

  Shared by run_bfs.py (one search, verbose=True) and run_graph500.py (64
  searches, verbose=False to avoid flooding stdout with a full phase
  breakdown per search).

  Returns (row_cols, device_time_cycles, profiled_rounds, round_duration_cycles,
  local_compute_max_cycles, local_term_cond_max_cycles) -- the last three as
  raw (profiled_rounds,) int64 arrays (also embedded as semicolon-joined
  strings in row_cols), handed back directly so the caller can feed them to
  check_round_vs_total_communication() without re-parsing its own CSV
  strings."""
  truncated = rounds_completed > max_rounds
  if truncated:
    print(f"[[ WARNING: BFS ran {rounds_completed} rounds but max_rounds={max_rounds} -- "
          f"only the first {max_rounds} rounds were timestamped (per-round phase breakdown "
          f"below is INCOMPLETE); bump max_rounds to profile the rest. "
          f"device_time_cycles/GTEPS remain CORRECT regardless -- "
          f"they use round_trip_start_buffer/round_trip_done_buffer, not ts_buf. ]]")

  phase_cycles, round_start, round_end, profiled_rounds = decode_pe_phase_cycles(
      ts_hwl_u32, height, width, max_rounds, rounds_completed)

  round_duration_cycles = compute_round_summary(round_start, round_end)
  device_time_cycles = int(round_trip_cycles.max())

  row_cols = {}
  row_cols["round_duration_cycles"] = ";".join(str(int(v)) for v in round_duration_cycles)

  per_round_max_by_name = {}
  for name, _, _ in PHASES:
    cycles = phase_cycles[name].reshape(profiled_rounds, height * width)
    per_round_min = cycles.min(axis=1)
    per_round_max = cycles.max(axis=1)
    per_round_avg = cycles.mean(axis=1)
    per_round_max_by_name[name] = per_round_max
    row_cols[f"{name}_min_cycles"] = ";".join(str(int(v)) for v in per_round_min)
    row_cols[f"{name}_max_cycles"] = ";".join(str(int(v)) for v in per_round_max)
    row_cols[f"{name}_avg_cycles"] = ";".join(f"{v:.1f}" for v in per_round_avg)
    if verbose:
      print(f"  {name:>16s}: min={per_round_min.tolist()} max={per_round_max.tolist()} "
            f"avg={np.round(per_round_avg, 1).tolist()}")

  if verbose:
    print(f"  round_duration_cycles (round_time, straggler span, VBCAST_ISSUE->TERM_COL_BCAST_DONE, "
          f"first {profiled_rounds} of {rounds_completed} rounds): "
          f"{round_duration_cycles.tolist()}")
    print(f"  device_time_cycles (total_runtime_cycles, round_trip_start_buffer -> "
          f"round_trip_done_buffer, excl. parent_resolve): {device_time_cycles}")

  return (row_cols, device_time_cycles, profiled_rounds, round_duration_cycles,
          per_round_max_by_name["local_compute"], per_round_max_by_name["local_term_cond"])


def decode_pe_phase_cycles(ts_hwl_u32, height, width, max_rounds, rounds_completed):
  """Decode ts_buf into (profiled_rounds, height, width) per-round cycle
  grids for the two REAL local-work phases (local_compute, local_term_cond,
  see PHASES) plus the two raw round-boundary timestamp grids
  (TS_VBCAST_ISSUE/TS_TERM_COL_BCAST_DONE) needed for round_time
  (compute_round_summary) -- see bool_pe.csl's TS_* comment for what each
  slot means.

  Returns (phase_cycles, round_start, round_end, profiled_rounds):
  phase_cycles is {"local_compute": grid, "local_term_cond": grid}, each
  (profiled_rounds, height, width) int64; round_start/round_end are the raw
  TS_VBCAST_ISSUE/TS_TERM_COL_BCAST_DONE grids, same shape."""
  profiled_rounds = min(rounds_completed, max_rounds)
  ts = decode_round_timestamps(ts_hwl_u32, height, width, max_rounds)
  ts = ts[:, :, :profiled_rounds, :]  # (height, width, profiled_rounds, NUM_TS_SLOTS)

  phase_cycles = {}
  for name, start_slot, end_slot in PHASES:
    cycles = ts[:, :, :, end_slot] - ts[:, :, :, start_slot]  # (height, width, profiled_rounds)
    phase_cycles[name] = np.transpose(cycles, (2, 0, 1))  # (profiled_rounds, height, width)

  round_start = np.transpose(ts[:, :, :, TS_VBCAST_ISSUE], (2, 0, 1))
  round_end = np.transpose(ts[:, :, :, TS_TERM_COL_BCAST_DONE], (2, 0, 1))

  return phase_cycles, round_start, round_end, profiled_rounds


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
  """m (edges traversed) + GTEPS for one search, per docs/GRAPH500_BENCHMARK.md
  sections 4/5. A_coo is the caller's A_csr.tocoo() -- passed in rather than
  recomputed here since run_graph500.py calls this once per search against
  the SAME static matrix.

  is_symmetric picks the counting convention (see docs/GRAPH500_BENCHMARK.md
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
