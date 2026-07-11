""" Shared per-round timing constants/decoders for bool_diag_spmv's
  f_spmv_iter profiling (see src/bool_pe.csl's ts_buf/record_ts()/TS_*
  comment for the on-device side of this). Single source of truth for
  run_bfs.py (which writes bfs_timing.csv) and plot_bfs_timing.py (which
  reads it back) -- previously these lived only in bench_timing.py, with
  plot_bfs_timing.py keeping its own separate, hand-copied phase list that
  could silently drift out of sync.
"""

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
