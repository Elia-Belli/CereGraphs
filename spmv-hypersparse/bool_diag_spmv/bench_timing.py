#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args
""" per-round, per-phase cycle-count profiling for bool_diag_spmv's
  f_spmv_iter (see src/bool_pe.csl's ts_buf/record_ts()/TS_* comment for the
  on-device side of this).

  Every PE captures its own hardware timestamp at each of 10 round-boundary
  points (broadcast issue, compute entry, reduce issue, ...); this script
  reads the full ts_buf rectangle back, decodes each PE's raw 3xu16
  timestamps into 48-bit values (same packing
  cerebras.sdk.sdk_utils.make_u48 uses), subtracts consecutive pairs to get
  each phase's elapsed cycles per round PER PE, then reduces across all
  P*P PEs (min/max/mean) -- exactly what "each PE measures its own tsc,
  then take min/max across PEs" asks for. One CSV row per run; each phase
  gets three columns (min/max/avg), each holding a ';'-joined per-round
  list, so a single row still carries the full per-round curve for later
  plotting (split on ';', cast to int/float).

  This measures f_spmv_iter's real per-round costs, not test_iterative.py's
  host-driven baseline (which pays a host round-trip per round and would
  not be representative of on-device timing at all).

  H2D/D2H transfer timing (h2d_*/d2h_* columns) uses the OTHER tsc
  mechanism already in bool_pe.csl -- f_enable_tsc()/f_tic()/f_toc(), which
  bracket an arbitrary span rather than record_ts()'s fixed round-boundary
  slots. launch()/memcpy_h2d/memcpy_d2h all share one serial command
  stream, so issuing f_tic (nonblock=True) right before the relevant calls
  and f_toc (nonblock=False) right after guarantees the transfers have
  actually completed by the time f_toc's timestamp is taken (confirmed
  against the SDK's own bandwidth-test example, which uses this exact
  bracket to compute a real bandwidth number in the simulator -- transfer
  cost is genuinely modeled there, not free). Per-PE deltas (toc_p - tic_p,
  same PE's own clock both times) don't need cross-PE clock
  synchronization the way an absolute cross-PE span would -- only min/max/
  avg of those already-computed per-PE durations is taken, exactly like
  the phase timing above. The measured delta does include the tic/toc
  launch commands' own dispatch/queueing latency (they're separate host
  calls, serialized through the same command stream as the transfer
  itself), not a perfectly isolated "wafer-only" transfer time -- assumed
  small and roughly constant per call, same accepted-simplification spirit
  as this repo's existing "no cross-PE clock-skew correction" note.

  d2h specifically times only visited_buf + parent_local_buf -- the actual
  BFS output a real caller would read back. rounds_completed/ts_buf are
  this SCRIPT's own profiling instrumentation, not part of any real data
  path, and ts_buf in particular can dwarf the real output's transfer size
  (it's sized for max_rounds regardless of how many rounds actually ran)
  -- both are still read back (needed for the phase breakdown above), just
  outside the timed d2h bracket, so they don't inflate d2h_cycles.

  h2d is split into two separately-timed brackets -- h2d_matrix (the
  hypersparse structure: mat_rows_buf, mat_col_idx/loc/len_buf,
  y_rows_init_buf, local_nnz*) and h2d_seed (just x_buf, the BFS source) --
  mirroring the Graph500 BFS benchmark's own timing contract (see
  https://graph500.org/?page_id=12, section 9.1, confirmed against the
  reference implementation's main.c): Kernel 1 (graph construction) is
  timed once and excluded from the per-search Kernel 2 (BFS) time, but the
  BFS timer explicitly starts "immediately prior to visiting the search
  root" and stops "when the output has been written to memory" -- i.e.
  seeding the root and reading the result back ARE part of a search's
  timed cost, only the one-time matrix structure upload is analogous to
  Kernel 1 and should be excluded when estimating a Graph500-style
  per-search time or TEPS. For a workload that reuses one compiled/loaded
  matrix across many search roots (the actual Graph500 protocol -- 64
  roots per graph), h2d_matrix is paid once and amortized; h2d_seed is
  paid every search, same as d2h.

  search_time_cycles / m_edges_traversed / visited_count / gteps implement
  the in-scope half of GRAPH500_BENCHMARK.md's TEPS definition (sections
  3/4/6): search_time_cycles = h2d_seed's worst-case-PE cycles + every
  profiled round's on-device phases (worst-case PE per phase, summed) --
  h2d_matrix/d2h stay excluded from it for now (see the doc's placeholder
  sections). m_edges_traversed uses one of two conventions depending on
  matrix_symmetric (recorded in m_convention): for a symmetric (undirected)
  A_csr, the reference implementation's own dedup rule (self-loops counted
  once, each non-self-loop edge counted once total, not twice) --
  Graph500-spec-comparable. For a directed A_csr (the BFS kernel itself has
  no symmetry requirement -- only Graph500's own edge-counting convention
  does), every directed edge whose source was visited instead (matches what
  the SpMV kernel actually examines: each visited vertex's out-edges,
  exactly once, the round it's in the active frontier) -- a real,
  meaningful count, just not directly comparable to a Graph500-spec number.
  See GRAPH500_BENCHMARK.md section 4 for both formulas and how the
  non-symmetric case was discovered (a real test matrix, rand600.mtx,
  produced an m over its own nnz/2 upper bound under the undirected rule).
  GTEPS = m / (search_time_cycles / clock_freq_hz) / 1e9 -- the
  conventional Graph500-reporting unit (10^9 edges/s), since raw TEPS
  values run into the millions/billions. CLOCK_FREQ_HZ is a plain
  assumed constant (875 MHz), not calibrated against this simulator run in
  any way -- same "not an absolute hardware-calibrated figure" caveat this
  repo's other tsc-based timing already carries.

  How to compile and run
     cs_python bench_timing.py --arch=wse3 --num_pe_cols=8 --num_pe_rows=8
        --channels=1 --driver=<path to cslc> --infile_mtx=<path to mtx file>
        --source=0 --max-rounds=10 --csv=bfs_timing.csv
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
from preprocess_bool import preprocess
from run_bool import (csl_compile_core, dist_x_to_diag_hwl, extract_diag_result,
                       hwl_to_oned_colmajor, oned_to_hwl_colmajor)
from scipy.io import mmread

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)

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

# on-device phases that make up the Graph500-style "search time" (see
# GRAPH500_BENCHMARK.md section 3) -- relay_total is used directly instead
# of its own 4 relay_col/row_* sub-phases, to avoid double-counting.
SEARCH_TIME_PHASES = [
    "visited_bcast", "vertical_bcast", "local_compute", "reduce", "local_term_cond", "relay_total",
]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--infile_mtx", required=True, help="the sparse matrix in MTX format")
  parser.add_argument("--num_pe_cols", type=int, required=True, help="width of the core rectangle")
  parser.add_argument("--num_pe_rows", type=int, required=True, help="height of the core rectangle")
  parser.add_argument("--fabric-dims", help="Fabric dimension, i.e. <W>,<H>")
  parser.add_argument("--compile-only", action="store_true", help="Compile only")
  parser.add_argument("--run-only", action="store_true", help="Run only")
  parser.add_argument("--width-west-buf", default=0, type=int, help="width of west buffer")
  parser.add_argument("--width-east-buf", default=0, type=int, help="width of east buffer")
  parser.add_argument("--channels", default=1, type=int, help="number of I/O channels, 1-16")
  parser.add_argument("-d", "--driver", help="path to the CSL compiler")
  parser.add_argument("--cmaddr", help="CM address and port, i.e. <IP>:<port>")
  parser.add_argument("--arch", help="wse2 or wse3 (default wse2)")
  parser.add_argument("--latestlink", default="latest", help="folder for the compiled ELFs")
  parser.add_argument("--source", type=int, default=0, help="single BFS source vertex")
  parser.add_argument("--max-rounds", type=int, default=10,
                       help="on-device cap on rounds actually profiled (bool_pe.csl's "
                            "ts_buf) -- rounds beyond this still run correctly, just "
                            "aren't timestamped; bump this if a run reports truncation")
  parser.add_argument("--csv", default=None,
                       help="CSV file to append this run's timing row to "
                            "(default: bfs_timing.csv next to this script)")
  return parser.parse_args()


def decode_round_timestamps(ts_hwl_u32, height, width, max_rounds):
  """ts_hwl_u32: (height, width, max_rounds*NUM_TS_SLOTS*3) uint32 array
  fresh off memcpy_d2h (u16 values zero-extended into u32 words, COL_MAJOR
  wire order already resolved into this hwl shape -- same convention as
  run_bool.py's other u16 buffers, e.g. test_iterative.read_parent_local_buf).
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


# timestamp.tsc_size_words in bool_pe.csl -- the <time> library's fixed
# [3]u16 timestamp width (see SKILL-LIBRARIES.md's <time> entry).
TSC_WORDS = 3

# WSE clock frequency, for converting search_time_cycles -> seconds for
# GTEPS (GRAPH500_BENCHMARK.md section 5/6).
CLOCK_FREQ_HZ = 875e6


def read_tic_toc_delta(runner, sym_tsc_start, sym_tsc_end, height, width):
  """Read back tsc_start_buffer/tsc_end_buffer (each PE's own f_tic()/
  f_toc() capture -- see f_enable_tsc()/f_tic()/f_toc() in bool_pe.csl) and
  return the per-PE elapsed-cycle deltas as a flat (height*width,) int64
  array. Same-PE subtraction (this PE's own toc minus this PE's own tic),
  so no cross-PE clock synchronization is needed -- see this module's own
  docstring for why."""

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


def main():
  """Main method to run the example code."""

  args = parse_args()

  cslc = "cslc"
  if args.driver is not None:
    cslc = args.driver

  width_west_buf = args.width_west_buf
  width_east_buf = args.width_east_buf
  channels = args.channels
  assert 1 <= channels <= 16, "number of I/O channels must be between 1 and 16"

  dirname = args.latestlink

  np_cols = args.num_pe_cols
  np_rows = args.num_pe_rows
  assert np_cols == np_rows, "diagonal-reduce design requires a square PE grid"
  P = np_cols
  width = np_cols
  height = np_rows
  max_rounds = args.max_rounds

  infile_mtx = args.infile_mtx
  source = args.source
  print(f"infile_mtx = {infile_mtx}, source = {source}, max_rounds = {max_rounds}")

  A_coo = mmread(infile_mtx)
  A_csr = A_coo.tocsr(copy=True)
  A_csr = A_csr.sorted_indices()
  assert A_csr.has_sorted_indices == 1, "Error: A is not sorted"

  [nrows, ncols] = A_csr.shape
  assert nrows == ncols, "boolean diagonal-reduce SpMV requires a square matrix"
  n = nrows
  nnz = A_csr.nnz
  assert 0 <= source < n, f"--source={source} out of range [0, {n})"

  print(f"Load matrix A, {nrows}-by-{ncols} with {nnz} nonzeros (structural, boolean)")

  # Graph500's own m formula (the undirected dedup rule below) only makes
  # sense for a symmetrized graph -- true for gen_rmat.py's output (it
  # symmetrizes and drops self-loops explicitly) but NOT guaranteed for an
  # arbitrary --infile_mtx. The BFS kernel itself has no such requirement
  # (it computes y = OR_j(A[i,j] AND x[j]) correctly on any square boolean
  # matrix, directed or not) -- only Graph500-style edge counting cares.
  # For a non-symmetric A_csr we use a different, directed-appropriate m
  # instead (see below), not a "meaningless" placeholder -- see
  # GRAPH500_BENCHMARK.md section 4.
  is_symmetric = (A_csr != A_csr.T).nnz == 0
  if not is_symmetric:
    print("[[ NOTE: A_csr is not symmetric (a directed graph, not gen_rmat.py's undirected "
          "style) -- using the directed edges-traversed formula instead of Graph500's own "
          "undirected dedup rule; not directly comparable to a Graph500-spec TEPS number, but "
          "still a real, meaningful edges-traversed count for this graph. See "
          "GRAPH500_BENCHMARK.md section 4. ]]")

  A_csc = A_csr.tocsc(copy=True)
  A_csc = A_csc.sorted_indices()
  assert A_csc.has_sorted_indices == 1, "Error: A is not sorted"

  matrix_info = preprocess(
      nrows, ncols, nnz, np_cols, np_rows,
      A_csr.indptr, A_csr.indices, A_csc.indptr, A_csc.indices,
  )

  max_local_nnz = matrix_info["max_local_nnz"]
  max_local_nnz_cols = matrix_info["max_local_nnz_cols"]
  max_local_nnz_rows = matrix_info["max_local_nnz_rows"]
  mat_rows_buf = matrix_info["mat_rows_buf"]
  mat_col_idx_buf = matrix_info["mat_col_idx_buf"]
  mat_col_loc_buf = matrix_info["mat_col_loc_buf"]
  mat_col_len_buf = matrix_info["mat_col_len_buf"]
  y_rows_init_buf = matrix_info["y_rows_init_buf"]
  local_nnz = matrix_info["local_nnz"]
  local_nnz_cols = matrix_info["local_nnz_cols"]
  local_nnz_rows = matrix_info["local_nnz_rows"]

  blk = math.ceil(n / P)

  # single-source seed -- a real BFS workload, not test_iterative.py's random
  # ~50%-density stress frontier (which would mask most rounds' costs behind
  # one giant first round).
  x_bool0 = np.zeros(n, dtype=bool)
  x_bool0[source] = True
  x_hwl0 = dist_x_to_diag_hwl(n, x_bool0, blk, P)

  fabric_offset_x = 1
  fabric_offset_y = 1
  core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
  core_fabric_offset_y = fabric_offset_y
  min_fabric_width = core_fabric_offset_x + width + 2 + 1 + width_east_buf
  min_fabric_height = core_fabric_offset_y + height + 1

  fabric_width = 0
  fabric_height = 0
  if args.fabric_dims:
    w_str, h_str = args.fabric_dims.split(",")
    fabric_width = int(w_str)
    fabric_height = int(h_str)
  if fabric_width == 0 or fabric_height == 0:
    fabric_width = min_fabric_width
    fabric_height = min_fabric_height
  assert fabric_width >= min_fabric_width
  assert fabric_height >= min_fabric_height

  code_csl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "layout_bool.csl")

  start = time.time()
  csl_compile_core(
      cslc, code_csl, dirname, fabric_width, fabric_height,
      core_fabric_offset_x, core_fabric_offset_y, args.run_only, args.arch,
      np_cols, np_rows, blk, max_local_nnz, max_local_nnz_cols, max_local_nnz_rows,
      channels, width_west_buf, width_east_buf, max_rounds=max_rounds,
  )
  print(f"Compilation done in {time.time()-start}s", flush=True)

  if args.compile_only:
    print("COMPILE ONLY: EXIT")
    return

  runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

  sym_x_buf = runner.get_id("x_buf")
  sym_visited_buf = runner.get_id("visited_buf")
  sym_parent_local_buf = runner.get_id("parent_local_buf")
  sym_rounds_completed = runner.get_id("rounds_completed")
  sym_ts_buf = runner.get_id("ts_buf")
  sym_tsc_start_buffer = runner.get_id("tsc_start_buffer")
  sym_tsc_end_buffer = runner.get_id("tsc_end_buffer")
  sym_mat_rows_buf = runner.get_id("mat_rows_buf")
  sym_mat_col_idx_buf = runner.get_id("mat_col_idx_buf")
  sym_mat_col_loc_buf = runner.get_id("mat_col_loc_buf")
  sym_mat_col_len_buf = runner.get_id("mat_col_len_buf")
  sym_y_rows_init_buf = runner.get_id("y_rows_init_buf")
  sym_local_nnz = runner.get_id("local_nnz")
  sym_local_nnz_cols = runner.get_id("local_nnz_cols")
  sym_local_nnz_rows = runner.get_id("local_nnz_rows")

  runner.load()
  runner.run()

  print("enabling tsc...")
  runner.launch("f_enable_tsc", nonblock=False)

  print("timing h2d: matrix structure upload (Graph500-style 'construction' -- one-time, "
        "amortized over however many searches would reuse this matrix, not per-search)...")
  runner.launch("f_tic", nonblock=True)

  mat_rows_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz, mat_rows_buf, np.uint32)
  runner.memcpy_h2d(sym_mat_rows_buf, mat_rows_buf_1d, 0, 0, width, height, max_local_nnz,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  mat_col_idx_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_cols, mat_col_idx_buf,
                                            np.uint32)
  runner.memcpy_h2d(sym_mat_col_idx_buf, mat_col_idx_buf_1d, 0, 0, width, height,
                     max_local_nnz_cols, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  mat_col_loc_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_cols, mat_col_loc_buf,
                                            np.uint32)
  runner.memcpy_h2d(sym_mat_col_loc_buf, mat_col_loc_buf_1d, 0, 0, width, height,
                     max_local_nnz_cols, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  mat_col_len_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_cols, mat_col_len_buf,
                                            np.uint32)
  runner.memcpy_h2d(sym_mat_col_len_buf, mat_col_len_buf_1d, 0, 0, width, height,
                     max_local_nnz_cols, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  y_rows_init_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_rows, y_rows_init_buf,
                                            np.uint32)
  runner.memcpy_h2d(sym_y_rows_init_buf, y_rows_init_buf_1d, 0, 0, width, height,
                     max_local_nnz_rows, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  local_nnz_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz, np.uint32)
  runner.memcpy_h2d(sym_local_nnz, local_nnz_1d, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  local_nnz_cols_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz_cols, np.uint32)
  runner.memcpy_h2d(sym_local_nnz_cols, local_nnz_cols_1d, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  local_nnz_rows_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz_rows, np.uint32)
  runner.memcpy_h2d(sym_local_nnz_rows, local_nnz_rows_1d, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)

  runner.launch("f_toc", nonblock=False)  # blocks -> every matrix-structure h2d above is done
  h2d_matrix_cycles = read_tic_toc_delta(
      runner, sym_tsc_start_buffer, sym_tsc_end_buffer, height, width)

  print("timing h2d: seed x upload (Graph500-style per-search cost -- this changes for every "
        "one of the 64 search roots, so unlike the matrix structure above it belongs inside "
        "the timed BFS-search interval)...")
  runner.launch("f_tic", nonblock=True)

  x_buf_1d = hwl_to_oned_colmajor(height, width, blk, x_hwl0, np.float32)
  runner.memcpy_h2d(sym_x_buf, x_buf_1d, 0, 0, width, height, blk,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)

  runner.launch("f_toc", nonblock=False)  # blocks -> seed x h2d above is done
  h2d_seed_cycles = read_tic_toc_delta(
      runner, sym_tsc_start_buffer, sym_tsc_end_buffer, height, width)

  print("running f_spmv_iter (timed)...")
  runner.launch("f_spmv_iter", nonblock=False)

  print("timing d2h readback (visited_buf + parent_local_buf -- the real BFS output, "
        "not our own instrumentation)...")
  runner.launch("f_tic", nonblock=True)

  visited_buf_1d = np.zeros(height * width * blk, np.float32)
  runner.memcpy_d2h(visited_buf_1d, sym_visited_buf, 0, 0, width, height, blk,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  parent_local_buf_1d = np.zeros(height * width * blk, np.uint32)
  runner.memcpy_d2h(parent_local_buf_1d, sym_parent_local_buf, 0, 0, width, height, blk,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)

  runner.launch("f_toc", nonblock=False)  # blocks -> both d2h reads above are done
  d2h_cycles = read_tic_toc_delta(runner, sym_tsc_start_buffer, sym_tsc_end_buffer, height, width)

  # rounds_completed/ts_buf are OUR OWN profiling instrumentation, not part
  # of a real BFS's data path -- read them back same as always, but outside
  # the timed bracket above (they'd otherwise dominate d2h_cycles purely
  # from ts_buf's size, see this module's own docstring/history).
  rounds_buf = np.zeros(height * width, np.uint32)
  runner.memcpy_d2h(rounds_buf, sym_rounds_completed, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)
  rounds_completed = int(np.reshape(rounds_buf, (height, width, 1), order="F")[(0, 0, 0)])

  ts_len = max_rounds * NUM_TS_SLOTS * 3
  ts_buf_1d = np.zeros(height * width * ts_len, np.uint32)
  runner.memcpy_d2h(ts_buf_1d, sym_ts_buf, 0, 0, width, height, ts_len,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)
  ts_hwl_u32 = np.reshape(ts_buf_1d, (height, width, ts_len), order="F")

  runner.stop()

  profiled_rounds = min(rounds_completed, max_rounds)
  if rounds_completed > max_rounds:
    print(f"[[ WARNING: BFS ran {rounds_completed} rounds but --max-rounds={max_rounds} -- "
          f"only the first {max_rounds} rounds were timestamped; bump --max-rounds to "
          "profile the rest ]]")

  ts = decode_round_timestamps(ts_hwl_u32, height, width, max_rounds)
  # (height, width, profiled_rounds, NUM_TS_SLOTS) -> (P*P, profiled_rounds, NUM_TS_SLOTS),
  # dropping unprofiled trailing rounds before computing any deltas.
  ts = ts[:, :, :profiled_rounds, :].reshape(height * width, profiled_rounds, NUM_TS_SLOTS)

  row = {
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "infile_mtx": os.path.basename(infile_mtx),
      "n": n,
      "nnz": nnz,
      "pe_grid": f"{np_cols}x{np_rows}",
      "source": source,
      "channels": channels,
      "rounds_completed": rounds_completed,
      "max_rounds": max_rounds,
      "matrix_symmetric": is_symmetric,
  }

  # one-shot transfers (not per-round) -- single min/max/avg cycle counts,
  # no ';'-joined list needed. See read_tic_toc_delta()/module docstring
  # for what these do and don't capture. h2d is split in two (matrix vs
  # seed) per the Graph500 construction-vs-per-search distinction -- see
  # the module docstring.
  for name, cycles in (("h2d_matrix", h2d_matrix_cycles), ("h2d_seed", h2d_seed_cycles),
                       ("d2h", d2h_cycles)):
    row[f"{name}_min_cycles"] = int(cycles.min())
    row[f"{name}_max_cycles"] = int(cycles.max())
    row[f"{name}_avg_cycles"] = f"{cycles.mean():.1f}"
    print(f"  {name:>18s}: min={int(cycles.min())} max={int(cycles.max())} "
          f"avg={cycles.mean():.1f}")

  print(f"rounds_completed = {rounds_completed} (profiled: {profiled_rounds})")
  phase_max_by_name = {}
  for name, start_slot, end_slot in PHASES:
    # cycles per round, per PE -- shape (P*P, profiled_rounds)
    cycles = ts[:, :, end_slot] - ts[:, :, start_slot]
    per_round_min = cycles.min(axis=0)
    per_round_max = cycles.max(axis=0)
    per_round_avg = cycles.mean(axis=0)
    phase_max_by_name[name] = per_round_max
    row[f"{name}_min_cycles"] = ";".join(str(int(v)) for v in per_round_min)
    row[f"{name}_max_cycles"] = ";".join(str(int(v)) for v in per_round_max)
    row[f"{name}_avg_cycles"] = ";".join(f"{v:.1f}" for v in per_round_avg)
    print(f"  {name:>18s}: min={per_round_min.tolist()} "
          f"max={per_round_max.tolist()} avg={np.round(per_round_avg, 1).tolist()}")

  # Graph500-style search time (GRAPH500_BENCHMARK.md section 3): seed h2d
  # (worst-case PE) + every profiled round's on-device phases (worst-case
  # PE per phase, summed) -- h2d_matrix/d2h stay excluded for now, see the
  # doc's placeholder sections.
  device_time_cycles = sum(int(phase_max_by_name[name].sum()) for name in SEARCH_TIME_PHASES)
  search_time_cycles = int(h2d_seed_cycles.max()) + device_time_cycles
  row["search_time_cycles"] = search_time_cycles
  print(f"[[ search_time_cycles (h2d_seed + device rounds, GRAPH500_BENCHMARK.md section 3): "
        f"{search_time_cycles} ]]")

  # m (edges traversed, GRAPH500_BENCHMARK.md section 4): needs the final
  # visited vector, which nothing else in this script actually decodes
  # (only its d2h transfer time is measured above) -- decode it now, same
  # helper test_iterative.py/plot_bfs_tree.py already use.
  visited_hwl = oned_to_hwl_colmajor(height, width, blk, visited_buf_1d, np.float32)
  visited = extract_diag_result(n, blk, P, visited_hwl)
  coo = A_csr.tocoo()
  if is_symmetric:
    # Graph500's own rule: dedup each undirected edge to one direction
    # (self-loops, if any, satisfy col == row and are kept once).
    m = int(np.sum(visited[coo.row] & (coo.col <= coo.row)))
    m_convention = "undirected_dedup"
  else:
    # directed graph, no mirror edge to dedup against -- count every
    # directed edge whose SOURCE (col, our row=dest/col=source convention)
    # was visited, matching what the SpMV kernel actually examines: every
    # visited vertex's out-edges get examined exactly once, the round it's
    # in the active frontier (see bool_pe.csl's compute()).
    m = int(np.sum(visited[coo.col]))
    m_convention = "directed_source_visited"
  row["visited_count"] = int(np.sum(visited))
  row["m_edges_traversed"] = m
  row["m_convention"] = m_convention
  print(f"[[ visited_count = {row['visited_count']} / {n}, m_edges_traversed = {m} "
        f"({m_convention}) ]]")

  # GTEPS (GRAPH500_BENCHMARK.md section 6): m / search_time_seconds, in
  # units of 10^9 edges/s -- the conventional Graph500-reporting unit
  # (raw TEPS values run into the millions/billions and are unwieldy).
  # search_time_cycles/h2d_matrix/d2h stay in scope-as-documented (section
  # 2/3) -- only cycles -> seconds -> GTEPS is new here.
  search_time_seconds = search_time_cycles / CLOCK_FREQ_HZ
  gteps = m / search_time_seconds / 1e9 if search_time_seconds > 0 else float("nan")
  row["clock_freq_hz"] = CLOCK_FREQ_HZ
  row["search_time_seconds"] = search_time_seconds
  row["gteps"] = gteps
  print(f"[[ GTEPS = {m} edges ({m_convention}) / {search_time_seconds * 1e6:.2f} us "
        f"(@{CLOCK_FREQ_HZ/1e6:.0f} MHz) = {gteps:.6f} GTEPS ]]"
        + ("" if is_symmetric else "  -- directed graph: not a Graph500-spec-comparable GTEPS, "
                                    "see m_convention"))

  csv_path = args.csv
  if csv_path is None:
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bfs_timing.csv")
  write_header = not os.path.exists(csv_path)
  if not write_header:
    with open(csv_path, newline="", encoding="utf-8") as f:
      existing_header = next(csv.reader(f), [])
    assert existing_header == list(row.keys()), (
        f"{csv_path}'s header doesn't match this run's columns (schema changed?) -- "
        "appending would silently misalign columns. Delete/rename the old CSV (it's a "
        "regenerable diagnostic log, not source data) or pass a different --csv path.")
  with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
      writer.writeheader()
    writer.writerow(row)
  print(f"appended timing row to {csv_path}")


if __name__ == "__main__":
  main()
