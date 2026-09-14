#!/usr/bin/env python3
# pylint: disable=too-many-function-args,wrong-import-position
""" Appliance-mode counterpart to run_bfs.py -- same single-source BFS on
  bool_diag_spmv's on-device f_spmv_iter kernel, same three reports (tree,
  correctness, timing/GTEPS), same bfs_timing.csv row and plot_bfs_timing.py
  call at the end. Only the host<->device interaction differs, because the
  Cerebras appliance's compile step and run step execute as two SEPARATE
  cluster jobs (the compile server is torn down once compilation finishes),
  unlike the simulator where one script does both in one process:

    1. python run_bfs.appliance.py --compile-only ...   # writes artifact_path.json
    2. python run_bfs.appliance.py ...                  # reads artifact_path.json, runs

  Plain `python`, NOT `cs_python` (this repo's local simulator-container
  wrapper) -- on the real ALCF cluster, cerebras.sdk.client talks to the
  job scheduler directly over the network from a plain Python process.
  sweep_bfs.sh's appliance branch invokes this with plain `python` accordingly.

  Everything host-side that ISN'T runner interaction (preprocess_bool,
  bfs_timing decode, scipy cross-check, tree plot, CSV row, plot_bfs_timing
  call) is UNCHANGED from run_bfs.py -- copied here rather than imported
  only because the runner-interaction section they're interleaved with in
  run_bfs.py's main() had to be rewritten. Keep the two files' non-runner
  sections in sync by hand if bfs_timing.py's schema or preprocess_bool.py's
  output ever changes.

  NOTE: cerebras.sdk.client / cerebras.appliance are only importable when
  actually connected to a Cerebras appliance -- this script cannot be
  exercised in a simulator-only environment.

  Two cluster/environment gotchas to watch for, both fixed here or in
  sweep_bfs.sh rather than being logic bugs:
  1. elf_dir's default ("out/latest") fails the appliance compile job
     because the remote sandbox has no pre-existing directory tree --
     cslc's -o flag needs a FLAT single-level name there, so main() passes
     os.path.basename(dirname) instead of dirname.
  2. SdkRuntime's memcpy_d2h streaming call can get its gRPC connection
     reset if the shell's https_proxy/HTTPS_PROXY also routes internal
     cluster traffic -- export no_proxy/NO_PROXY covering the cluster's
     internal network before invoking this script directly (sweep_bfs.sh's
     appliance branches already do this).

  How to compile and run (simulator, i.e. --simulator; drop it for real
  WSE-3 hardware -- see the fabric-dims branch in main() for why that also
  changes --fabric-dims)
     python run_bfs.appliance.py --num_pe_cols=8 --num_pe_rows=8 --channels=1
        --infile_mtx=<path to mtx file> --source=0 --simulator --compile-only
     python run_bfs.appliance.py --num_pe_cols=8 --num_pe_rows=8 --channels=1
        --infile_mtx=<path to mtx file> --source=0 --simulator
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import networkx as nx
import numpy as np

# This script lives in scripts/; device_io.py/graph_loader.py/preprocess_bool.py/
# bfs_timing.py live in ../implementation/, plot_bfs_timing.py/bfs_tree_plot.py
# in ../plots/.
BFS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BFS_ROOT, "implementation"))
sys.path.insert(0, os.path.join(BFS_ROOT, "plots"))

# See run_bfs.py's own matching constant for the full derivation --
# docs/ERRORS.md #28.
_SCIPY_CHECK_AUTO_DISABLE_N = 1_100_000

from graph_loader import load_graph
from preprocess_bool import preprocess
from scipy.sparse.csgraph import breadth_first_order

import plot_bfs_timing
from bfs_timing import (CLOCK_FREQ_HZ, NUM_TS_SLOTS, check_round_vs_total_communication,
                         compute_m_and_gteps, decode_pe_phase_cycles, decode_phase_row,
                         save_pe_phase_cycles, timed_transfer)
from bfs_tree_plot import build_digraph, invalid_parents, render_tree_comparison
from device_io import (csl_compile_core_appliance, derive_visited_from_parent,
                        extract_parent_result, hwl_to_oned_colmajor, prepare_h2d_chunked,
                        send_h2d_chunked, single_source_seed_pe)

from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder  # pylint: disable=import-error,no-name-in-module
from cerebras.sdk.client import SdkRuntime  # pylint: disable=import-error,no-name-in-module

ARTIFACT_PATH_FILENAME = "artifact_path.json"


def make_u48(words):
  return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


def read_tic_toc_delta_appliance(runner, sym_tsc_start, sym_tsc_end, height, width):
  """Same as bfs_timing.read_tic_toc_delta, but against the appliance
  MemcpyDataType/MemcpyOrder enums -- a different pb2-backed type than the
  simulator pybind module's that bfs_timing.py's version imports, so it
  can't be reused as-is here."""
  from bfs_timing import TSC_WORDS  # local: avoid importing the simulator pybind module at file scope

  def _read(sym):
    buf_1d = np.zeros(height * width * TSC_WORDS, np.uint32)
    runner.memcpy_d2h(buf_1d, sym, 0, 0, width, height, TSC_WORDS, streaming=False,
                       data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.COL_MAJOR,
                       nonblock=False)
    hwl = np.reshape(buf_1d, (height, width, TSC_WORDS), order="F")
    w0 = hwl[..., 0].astype(np.int64)
    w1 = hwl[..., 1].astype(np.int64)
    w2 = hwl[..., 2].astype(np.int64)
    return (w0 + (w1 << 16) + (w2 << 32)).reshape(-1)

  tic = _read(sym_tsc_start)
  toc = _read(sym_tsc_end)
  return toc - tic


def read_sync_corrected_span_appliance(runner, sym_tsc_start, sym_tsc_end, sym_tsc_ref,
                                        height, width):
  """Same as bfs_timing.read_sync_corrected_span, but against the appliance
  MemcpyDataType/MemcpyOrder enums -- see read_tic_toc_delta_appliance."""
  from bfs_timing import TSC_WORDS  # local: avoid importing the simulator pybind module at file scope

  def _read(sym):
    buf_1d = np.zeros(height * width * TSC_WORDS, np.uint32)
    runner.memcpy_d2h(buf_1d, sym, 0, 0, width, height, TSC_WORDS, streaming=False,
                       data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.COL_MAJOR,
                       nonblock=False)
    hwl = np.reshape(buf_1d, (height, width, TSC_WORDS), order="F")
    w0 = hwl[..., 0].astype(np.int64)
    w1 = hwl[..., 1].astype(np.int64)
    w2 = hwl[..., 2].astype(np.int64)
    return (w0 + (w1 << 16) + (w2 << 32)).reshape(-1)

  tic = _read(sym_tsc_start)
  toc = _read(sym_tsc_end)
  ref = _read(sym_tsc_ref)

  mid = width // 2
  pcol_id = np.tile(np.arange(width, dtype=np.int64), height)
  hop_distance = np.abs(pcol_id - mid)

  corrected_ref = ref - hop_distance
  corrected_tic = tic - corrected_ref
  corrected_toc = toc - corrected_ref
  return int(corrected_toc.max() - corrected_tic.min())


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--infile_mtx", required=True, help="the sparse matrix in MTX format")
  parser.add_argument("--num_pe_cols", type=int, required=True, help="width of the core rectangle")
  parser.add_argument("--num_pe_rows", type=int, required=True, help="height of the core rectangle")
  parser.add_argument("--fabric-dims", help="Fabric dimension, i.e. <W>,<H>")
  parser.add_argument("--compile-only", action="store_true", help="Compile only (writes hash.json)")
  parser.add_argument("--width-west-buf", default=0, type=int, help="width of west buffer")
  parser.add_argument("--width-east-buf", default=0, type=int, help="width of east buffer")
  parser.add_argument("--channels", default=1, type=int, help="number of I/O channels, 1-16")
  parser.add_argument("--arch", choices=["wse3"], default="wse3",
                       help="WSE-3 only -- this kernel is no longer tested/supported on WSE-2")
  parser.add_argument("--latestlink", default="out/latest", help="folder for the compiled ELFs")
  parser.add_argument("--source", type=int, default=0, help="single BFS source vertex")
  parser.add_argument("--simulator", action="store_true",
                       help="run the appliance-mode client against the simulator instead of real "
                            "hardware -- lets this code path be sanity-checked without appliance "
                            "access, though it hasn't been in the environment this was written in "
                            "(cerebras.sdk.client isn't installed there at all)")

  parser.add_argument("--notree", action="store_true", help="skip the tree comparison plot")
  parser.add_argument("--notimings", action="store_true",
                       help="skip per-round timing/GTEPS (also skips the h2d/d2h tsc "
                            "instrumentation itself, saving the real transfer time it costs)")
  parser.add_argument("--nocorrectness", action="store_true",
                       help="skip printing the scipy cross-check numbers")
  parser.add_argument("--force-scipy", action="store_true",
                       help=f"run the scipy cross-check (and/or tree plot) even above "
                            f"{_SCIPY_CHECK_AUTO_DISABLE_N} vertices, where it's auto-disabled by "
                            f"default (real OOM risk under this host's 100GiB per-user cgroup "
                            f"limit -- confirmed at RMAT s24 @ 750x750, see docs/ERRORS.md #28). "
                            f"Only pass this if you've checked host memory headroom yourself.")

  parser.add_argument("--max-rounds", type=int, default=10,
                       help="on-device cap on rounds actually profiled (bool_pe.csl's ts_buf)")
  parser.add_argument("--csv", default=None,
                       help="CSV file to append this run's timing row to "
                            "(default: results/sim/bfs_timing.csv or results/hw/bfs_timing.csv "
                            "next to this script, depending on --simulator)")
  parser.add_argument("--out-tree", default=None, help="tree plot output path")
  parser.add_argument("--out-timing", default=None, help="timing plot output path")
  parser.add_argument("--no-show-parent-mismatch", dest="show_parent_mismatch",
                       action="store_false", help="see run_bfs.py's own flag")
  parser.set_defaults(show_parent_mismatch=True)

  parser.add_argument("--dump-pe-timing", action="store_true",
                       help="save the full per-PE-per-round-per-phase cycle grid to a .npz file")
  parser.add_argument("--pe-timing-out", default=None, help="path for --dump-pe-timing's output")
  return parser.parse_args()


def main():
  """Main method to run the example code."""

  args = parse_args()
  need_timing = not args.notimings
  need_scipy = not args.notree or not args.nocorrectness

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

  A_coo = load_graph(infile_mtx)
  A_csr = A_coo.tocsr(copy=True)
  # A_coo's only other use was building A_csr just above -- free it now
  # rather than hold both full live copies of the whole matrix at once
  # (see docs/ERRORS.md #26, and graph_loader.py's own load_graph() for
  # the matching .data-dtype-shrink half of this fix). A_csc used to
  # exist here too (a third live copy) -- removed entirely, see the
  # preprocess() call site below.
  del A_coo
  A_csr = A_csr.sorted_indices()
  assert A_csr.has_sorted_indices == 1, "Error: A is not sorted"

  [nrows, ncols] = A_csr.shape
  assert nrows == ncols, "boolean diagonal-reduce SpMV requires a square matrix"
  n = nrows
  nnz = A_csr.nnz
  assert 0 <= source < n, f"--source={source} out of range [0, {n})"

  print(f"Load matrix A, {nrows}-by-{ncols} with {nnz} nonzeros (structural, boolean)")

  # Only auto-disable when the tree plot wasn't requested either -- see
  # run_bfs.py's own matching comment for why.
  if need_scipy and args.notree and n > _SCIPY_CHECK_AUTO_DISABLE_N and not args.force_scipy:
    print(f"[[ NOTE: scipy cross-check auto-disabled -- n={n} exceeds the RMAT-s20-scale "
          f"threshold ({_SCIPY_CHECK_AUTO_DISABLE_N}) where scipy's own breadth_first_order() "
          f"plus rebuilding a transposed CSR copy of A risks OOMing under this host's 100GiB "
          f"per-user cgroup limit (real, dmesg-confirmed at RMAT s24 @ 750x750, see "
          f"docs/ERRORS.md #28) -- pass --force-scipy to run it anyway. ]]")
    need_scipy = False

  is_symmetric = None
  if need_timing:
    is_symmetric = (A_csr != A_csr.T).nnz == 0
    if not is_symmetric:
      print("[[ NOTE: A_csr is not symmetric -- using the directed edges-traversed formula "
            "instead of Graph500's own undirected dedup rule. See docs/GRAPH500_BENCHMARK.md "
            "section 4. ]]")

  # Bottom-up is the only traversal strategy this kernel runs (see
  # docs/ERRORS.md #21) -- the host uploads the matrix ALREADY in
  # CSR-by-destination form, so preprocess() is fed A_csr's arrays into
  # its cscColPtr/cscRowInd parameter slot -- see run_bfs.py's own comment
  # on this same call for the full derivation. A separate CSC
  # representation of A used to be built here too (feeding preprocess()'s
  # now-removed csrRowPtr/csrColInd parameter pair, whose only output
  # turned out unread by any live caller) -- removed entirely, see
  # docs/ERRORS.md #26 follow-up and run_bfs.py's own matching comment.
  matrix_info = preprocess(
      nrows, ncols, nnz, np_cols, np_rows,
      A_csr.indptr, A_csr.indices,
  )

  max_local_nnz = matrix_info["max_local_nnz"]
  # Post-swap: see run_bfs.py's own comment on this same extraction (both
  # the field renaming AND the (px,py)-axis transpose fix -- a real bug
  # caught via a standalone test: without the transpose, every off-diagonal
  # PE silently received its transpose partner's block).
  max_local_nnz_rows = matrix_info["max_local_nnz_cols"]
  mat_rows_buf = np.transpose(matrix_info["mat_rows_buf"], (1, 0, 2))
  mat_row_idx_buf = np.transpose(matrix_info["mat_col_idx_buf"], (1, 0, 2))
  mat_row_loc_buf = np.transpose(matrix_info["mat_col_loc_buf"], (1, 0, 2))
  mat_row_len_buf = np.transpose(matrix_info["mat_col_len_buf"], (1, 0, 2))
  local_nnz = np.transpose(matrix_info["local_nnz"], (1, 0, 2))
  local_nnz_rows = np.transpose(matrix_info["local_nnz_cols"], (1, 0, 2))

  blk = math.ceil(n / P)
  bitmap_words = (blk + 31) // 32

  seed_px, seed_py, seed_local_x = single_source_seed_pe(source, blk, P)

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
    if args.simulator:
      # Simulator: minimal dims, same sizing rule run_bfs.py uses -- a huge
      # --fabric-dims would just make the simulator do more (pointless) work.
      fabric_width = min_fabric_width
      fabric_height = min_fabric_height
    else:
      # Real WSE-3 hardware: ALCF's docs are explicit this should be the
      # fabric's full physical size, not a minimally-computed rectangle.
      fabric_width, fabric_height = 762, 1172
      core_fabric_offset_x, core_fabric_offset_y = 4, 1
  if args.simulator:
    assert fabric_width >= min_fabric_width
    assert fabric_height >= min_fabric_height

  # Unlike run_bfs.py's code_csl (one joined absolute path), the appliance
  # client wants the containing directory and the bare filename separately
  # -- see csl_compile_core_appliance's own docstring.
  csl_dir = os.path.join(BFS_ROOT, "implementation", "src")
  csl_file = "layout_bool.csl"

  if args.compile_only:
    print("WARNING: compile only -- the appliance's compile server is torn down once this "
          "returns, so SdkRuntime can't be used in this same invocation")
    start = time.time()
    # os.path.basename(dirname), not dirname -- see module docstring gotcha #1.
    artifact_path = csl_compile_core_appliance(
        csl_dir, csl_file, os.path.basename(dirname), fabric_width, fabric_height,
        core_fabric_offset_x, core_fabric_offset_y, args.arch,
        np_cols, np_rows, blk, max_local_nnz, max_local_nnz_rows,
        channels, width_west_buf, width_east_buf, max_rounds=max_rounds,
    )
    print(f"Compilation done in {time.time()-start}s", flush=True)
    # {"artifact_path": ...} dict, matching ALCF's own documented format.
    with open(ARTIFACT_PATH_FILENAME, "w", encoding="utf-8") as f:
      json.dump({"artifact_path": artifact_path}, f)
    print(f"dumped artifact_path to {ARTIFACT_PATH_FILENAME}")
    print("COMPILE ONLY: EXIT")
    return

  print(f"load artifact_path from {ARTIFACT_PATH_FILENAME}")
  with open(ARTIFACT_PATH_FILENAME, encoding="utf-8") as f:
    artifact_path = json.load(f)["artifact_path"]

  start = time.time()
  # disable_version_check: ALCF's tutorial scripts pass this unconditionally
  # on both Compiler and Runtime -- see csl_compile_core_appliance.
  with SdkRuntime(artifact_path, simulator=args.simulator, disable_version_check=True) as runner:
    sym_x_bitmap = runner.get_id("x_bitmap")
    sym_parent_values = runner.get_id("parent_values")
    sym_rounds_completed = runner.get_id("rounds_completed")
    sym_mat_rows_buf = runner.get_id("mat_rows_buf")
    sym_mat_row_idx_buf = runner.get_id("mat_row_idx_buf")
    sym_mat_row_loc_buf = runner.get_id("mat_row_loc_buf")
    sym_mat_row_len_buf = runner.get_id("mat_row_len_buf")
    sym_local_nnz = runner.get_id("local_nnz")
    sym_local_nnz_rows = runner.get_id("local_nnz_rows")
    sym_parent_occupancy = runner.get_id("parent_occupancy")
    if need_timing:
      sym_ts_buf = runner.get_id("ts_buf")
      sym_tsc_start_buffer = runner.get_id("tsc_start_buffer")
      sym_tsc_end_buffer = runner.get_id("tsc_end_buffer")
      sym_tsc_ref_buffer = runner.get_id("tsc_ref_buffer")
      sym_nf_history = runner.get_id("nf_history")
      sym_parent_resolve_tic_buffer = runner.get_id("parent_resolve_tic_buffer")
      sym_parent_resolve_toc_buffer = runner.get_id("parent_resolve_toc_buffer")
      sym_round_trip_start_buffer = runner.get_id("round_trip_start_buffer")
      sym_round_trip_done_buffer = runner.get_id("round_trip_done_buffer")

    # load()/run() are called by SdkRuntime's own __enter__ in appliance mode.

    # All host-side marshaling for the matrix-structure upload happens here,
    # BEFORE the timed bracket below -- keeping reshape work out of the
    # tic/toc window is load-bearing, not stylistic (see bfs_timing.timed_transfer).
    mat_rows_prepared = prepare_h2d_chunked(mat_rows_buf, height, width, max_local_nnz, np.uint32)
    mat_row_idx_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_rows, mat_row_idx_buf,
                                              np.uint32)
    mat_row_loc_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_rows, mat_row_loc_buf,
                                              np.uint32)
    mat_row_len_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_rows, mat_row_len_buf,
                                              np.uint32)
    local_nnz_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz, np.uint32)
    local_nnz_rows_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz_rows, np.uint32)

    if need_timing:
      print("enabling tsc...")
      print("timing h2d: matrix structure upload (Graph500-style 'construction')...")

    def _send_h2d_matrix():
      send_h2d_chunked(runner, sym_mat_rows_buf, mat_rows_prepared, width, max_local_nnz,
                       MemcpyDataType.MEMCPY_16BIT, MemcpyOrder.COL_MAJOR, nonblock=True)
      runner.memcpy_h2d(sym_mat_row_idx_buf, mat_row_idx_buf_1d, 0, 0, width, height,
                         max_local_nnz_rows, streaming=False,
                         data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.COL_MAJOR,
                         nonblock=True)
      runner.memcpy_h2d(sym_mat_row_loc_buf, mat_row_loc_buf_1d, 0, 0, width, height,
                         max_local_nnz_rows, streaming=False,
                         data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.COL_MAJOR,
                         nonblock=True)
      runner.memcpy_h2d(sym_mat_row_len_buf, mat_row_len_buf_1d, 0, 0, width, height,
                         max_local_nnz_rows, streaming=False,
                         data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.COL_MAJOR,
                         nonblock=True)
      runner.memcpy_h2d(sym_local_nnz, local_nnz_1d, 0, 0, width, height, 1,
                         streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                         order=MemcpyOrder.COL_MAJOR, nonblock=True)
      runner.memcpy_h2d(sym_local_nnz_rows, local_nnz_rows_1d, 0, 0, width, height, 1,
                         streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                         order=MemcpyOrder.COL_MAJOR, nonblock=not need_timing)

    h2d_matrix_span_cycles = timed_transfer(
        runner, height, width, _send_h2d_matrix, need_timing=need_timing,
        span_reader=read_sync_corrected_span_appliance,
        sym_tsc_start=sym_tsc_start_buffer if need_timing else None,
        sym_tsc_end=sym_tsc_end_buffer if need_timing else None,
        sym_tsc_ref=sym_tsc_ref_buffer if need_timing else None,
        enable_tsc=True)
    if need_timing:
      print("timing h2d: seed x upload (Graph500-style per-search cost)...")

    def _send_h2d_seed():
      runner.memcpy_h2d(sym_x_bitmap, seed_local_x, seed_px, seed_py, 1, 1, bitmap_words,
                         streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                         order=MemcpyOrder.COL_MAJOR, nonblock=False)

    h2d_seed_span_cycles = timed_transfer(
        runner, height, width, _send_h2d_seed, need_timing=need_timing,
        span_reader=read_sync_corrected_span_appliance,
        sym_tsc_start=sym_tsc_start_buffer if need_timing else None,
        sym_tsc_end=sym_tsc_end_buffer if need_timing else None,
        sym_tsc_ref=sym_tsc_ref_buffer if need_timing else None)

    print("running f_spmv_iter...")
    runner.launch("f_spmv_iter", nonblock=False)

    if need_timing:
      print("timing d2h readback (parent_values -- the real BFS output)...")

    # #24 removed the on-device relay that used to resolve each row's P
    # per-PE candidates to a single winner at PE-column MID -- the host now
    # reads EVERY PE's own compact parent_values array and combines per-row
    # itself (see extract_parent_result()'s own comment). Real cost: this
    # transfer is now width*height*max_local_nnz_rows elements instead of
    # one column's height*blk -- ~P times more data, still far under the
    # d2h gRPC ~2GiB message-size ceiling at any scale this kernel targets.
    # #26 halved that again: parent_values is now u16 (a local column
    # offset, not a u32 global vertex id), so this is a 16-bit transfer,
    # not 32-bit -- data_type=MEMCPY_16BIT below reflects the DEVICE-side
    # symbol's width. The HOST-side numpy buffer still has to be uint32
    # regardless (same SDK requirement every other u16 device buffer in
    # this file already works around); passing uint16 here instead throws
    # "Internal data type of any memcpy_d2h()/memcpy_h2d() operation should
    # be 32 bit" at runtime.
    parent_values_1d = np.zeros(height * width * max_local_nnz_rows, np.uint32)

    def _read_d2h_parent():
      runner.memcpy_d2h(parent_values_1d, sym_parent_values, 0, 0, width, height,
                         max_local_nnz_rows, streaming=False,
                         data_type=MemcpyDataType.MEMCPY_16BIT,
                         order=MemcpyOrder.COL_MAJOR, nonblock=False)

    d2h_span_cycles = timed_transfer(
        runner, height, width, _read_d2h_parent, need_timing=need_timing,
        span_reader=read_sync_corrected_span_appliance,
        sym_tsc_start=sym_tsc_start_buffer if need_timing else None,
        sym_tsc_end=sym_tsc_end_buffer if need_timing else None,
        sym_tsc_ref=sym_tsc_ref_buffer if need_timing else None)

    rounds_buf = np.zeros(height * width, np.uint32)
    runner.memcpy_d2h(rounds_buf, sym_rounds_completed, 0, 0, width, height, 1,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    rounds_completed = int(np.reshape(rounds_buf, (height, width, 1), order="F")[(0, 0, 0)])

    # How many of each PE's blk local rows already have a real parent
    # candidate right before the one-time end-of-run reduce_select_any call.
    # Read back grid-wide (not just PE(0,0)) since occupancy is genuinely
    # per-PE, unlike nf_history which is identical everywhere by
    # construction.
    parent_occupancy_buf = np.zeros(height * width, np.uint32)
    runner.memcpy_d2h(parent_occupancy_buf, sym_parent_occupancy, 0, 0, width, height, 1,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    parent_occupancy_hwl = np.reshape(parent_occupancy_buf, (height, width, 1), order="F")[:, :, 0]
    parent_occupancy_frac = parent_occupancy_hwl.astype(np.float64) / float(blk)
    print(f"[[ parent_occupancy (Phase 1 sparse-reduce_select_any investigation): "
          f"min={parent_occupancy_frac.min():.4f}, max={parent_occupancy_frac.max():.4f}, "
          f"mean={parent_occupancy_frac.mean():.4f} (fraction of blk={blk} local rows with a "
          f"real parent, per-PE, right before reduce_select_any) ]]")

    ts_hwl_u32 = None
    if need_timing:
      ts_len = max_rounds * NUM_TS_SLOTS * 3
      ts_buf_1d = np.zeros(height * width * ts_len, np.uint32)
      runner.memcpy_d2h(ts_buf_1d, sym_ts_buf, 0, 0, width, height, ts_len,
                         streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                         order=MemcpyOrder.COL_MAJOR, nonblock=False)
      ts_hwl_u32 = np.reshape(ts_buf_1d, (height, width, ts_len), order="F")

      nf_history_1d = np.zeros(height * width * max_rounds, np.uint32)
      runner.memcpy_d2h(nf_history_1d, sym_nf_history, 0, 0, width, height, max_rounds,
                         streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                         order=MemcpyOrder.COL_MAJOR, nonblock=False)
      nf_history_hwl = np.reshape(nf_history_1d, (height, width, max_rounds), order="F")
      nf_history = nf_history_hwl[0, 0, :]

      # mpi_x.reduce_select_any()'s one-time end-of-run cost.
      parent_resolve_cycles = read_tic_toc_delta_appliance(
          runner, sym_parent_resolve_tic_buffer, sym_parent_resolve_toc_buffer, height, width)
      # Spatial (row, col) view -- needs the default C-order reshape, not
      # "F" (read_tic_toc_delta_appliance's own `.reshape(-1)` already uses
      # numpy's default order).
      parent_resolve_grid = parent_resolve_cycles.reshape((height, width))

      # Always-correct round-trip span, independent of max_rounds/ts_buf
      # truncation -- see round_trip_start_buffer/round_trip_done_buffer's
      # declaration in bool_pe.csl.
      round_trip_cycles = read_tic_toc_delta_appliance(
          runner, sym_round_trip_start_buffer, sym_round_trip_done_buffer, height, width)

    # stop() is called by SdkRuntime's own __exit__ in appliance mode.

  end = time.time()
  print(f"*** Run done in {end-start}s")

  parent_values_hwl = np.reshape(parent_values_1d, (height, width, max_local_nnz_rows), order="F")
  # #24 moved the real per-row parent combine off-device entirely -- the
  # on-device "parent_resolve" TSC bracket (parent_resolve_tic/toc_buffer)
  # now only brackets a trivial occupancy-count scan, NOT this. Without a
  # host-side timer, this cost was invisible to every measurement (it runs
  # after `end = time.time()` above, i.e. after search_time_cycles/GTEPS are
  # already computed from device+transfer cycles alone) -- time it
  # explicitly here instead.
  host_parent_combine_start = time.time()
  device_parent = extract_parent_result(
      n, blk, mat_row_idx_buf, local_nnz_rows, parent_values_hwl)
  host_parent_combine_seconds = time.time() - host_parent_combine_start
  print(f"[[ host_parent_combine_seconds: {host_parent_combine_seconds:.6f}s (host-side "
        f"per-row combine over parent_values_hwl, replacing the on-device relay removed in "
        f"docs/ERRORS.md #24 -- wall-clock, NOT device cycles; not included in "
        f"search_time_cycles/gteps, which stay device+transfer-only) ]]")
  device_parent[source] = source
  device_visited = derive_visited_from_parent(n, device_parent, source)

  scipy_parent = scipy_visited = scipy_levels = None
  mismatch = n_mismatch = scipy_ok = scipy_diff_device = None
  if need_scipy:
    print("scipy reference: breadth_first_order")
    A_fwd = A_csr.transpose().tocsr()
    scipy_order, scipy_pred = breadth_first_order(A_fwd, source, directed=True,
                                                   return_predecessors=True)
    scipy_visited = np.zeros(n, dtype=bool)
    scipy_visited[scipy_order] = True
    scipy_parent = scipy_pred.astype(np.int64)
    scipy_parent[scipy_parent < 0] = -1
    scipy_parent[source] = source

    n_mismatch_scipy_device = int(np.sum(scipy_visited != device_visited))
    bad_device = invalid_parents(device_parent, device_visited, A_csr, source)
    node_ids = np.arange(n)
    scipy_diff_device = ((device_parent != scipy_parent) & device_visited & scipy_visited
                          & (node_ids != source))

    mismatch = device_visited != scipy_visited
    n_mismatch = int(np.sum(mismatch))
    scipy_ok = n_mismatch_scipy_device == 0 and not bad_device

    if not args.nocorrectness:
      print(f"[[ scipy visited: {int(np.sum(scipy_visited))}/{n} ]]")
      print(f"[[ visited vs scipy mismatches: device={n_mismatch_scipy_device} ]]")
      print(f"[[ invalid device parents vs original graph: {len(bad_device)} ]]")
      if bad_device:
        print(f"  bad device parents at: {bad_device[:20]}"
              f"{' ...' if len(bad_device) > 20 else ''}")
      print(f"[[ parent differs from scipy's own pick (expected tie-break "
            f"difference, not a bug): device={int(np.sum(scipy_diff_device))} ]]")
      print(f"[[ mismatches (visited, device vs scipy): {n_mismatch} / {n} ]]")
      print(f"[[ scipy cross-check: {'OK' if scipy_ok else 'FAILED'} ]]")

    dist_from_source = nx.single_source_shortest_path_length(build_digraph(A_csr), source)
    scipy_levels = max(dist_from_source.values()) if dist_from_source else 0

  if not args.notree:
    matrix_stem = os.path.splitext(os.path.basename(infile_mtx))[0]
    out_tree = args.out_tree or os.path.join(
        BFS_ROOT, "results",
        "sim" if args.simulator else "hw", "tree",
        f"{matrix_stem}_{np_cols}x{np_rows}_src{source}.png")
    render_tree_comparison(
        A_csr, source, scipy_parent, scipy_visited, scipy_levels,
        device_parent, device_visited, rounds_completed,
        mismatch, n_mismatch, scipy_ok, scipy_diff_device,
        args.show_parent_mismatch, infile_mtx, np_cols, np_rows, out_tree)

  if need_timing:
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

    # parent_resolve has no sync bracket (on-device-only reduce, not a
    # host-device transfer) -- per-PE-max-of-self-delta is exactly right
    # there since no cross-PE clock sync is needed for a quantity that
    # never leaves the fabric.
    for name, cycles in (("parent_resolve", parent_resolve_cycles),):
      row[f"{name}_min_cycles"] = int(cycles.min())
      row[f"{name}_max_cycles"] = int(cycles.max())
      row[f"{name}_avg_cycles"] = f"{cycles.mean():.1f}"
      print(f"  {name:>18s}: min={int(cycles.min())} max={int(cycles.max())} "
            f"avg={cycles.mean():.1f}")

    # Sync-corrected cross-PE span (see read_sync_corrected_span_appliance) --
    # the true max(toc)-min(tic) across all PEs. The only h2d/d2h timing this
    # project records: the per-PE-max-of-self-delta approach it replaced is
    # only a structural lower bound on this span, never more accurate.
    for name, span in (("h2d_matrix", h2d_matrix_span_cycles),
                        ("h2d_seed", h2d_seed_span_cycles), ("d2h", d2h_span_cycles)):
      row[f"{name}_span_cycles"] = span
      print(f"  {name:>18s}: sync-corrected span={span}")

    (row_cols, device_time_cycles, profiled_rounds, round_duration_cycles,
     local_compute_max_cycles, local_term_cond_max_cycles) = decode_phase_row(
        ts_hwl_u32, height, width, max_rounds, rounds_completed, round_trip_cycles)
    print(f"rounds_completed = {rounds_completed} (profiled: {profiled_rounds})")
    row.update(row_cols)

    profiled_nf = [int(v) for v in nf_history[:profiled_rounds]]
    row["nf_history"] = ";".join(str(v) for v in profiled_nf)
    print(f"  nf per round (this round's own discovery count): {profiled_nf}")

    if args.dump_pe_timing:
      phase_cycles, round_start, round_end, _ = decode_pe_phase_cycles(
          ts_hwl_u32, height, width, max_rounds, rounds_completed)
      # Raw per-PE local_compute/local_term_cond grids, plus the two raw
      # round-boundary grids round_time is built from -- per-communication-
      # phase skew adjustment was tried here and dropped as unreliable (see
      # docs/GRAPH500_BENCHMARK.md).
      phase_cycles.update({"raw_round_start": round_start, "raw_round_end": round_end})

      matrix_stem = os.path.splitext(os.path.basename(infile_mtx))[0]
      run_id = f"{matrix_stem}_{np_cols}x{np_rows}_src{source}"
      pe_timing_out = args.pe_timing_out or os.path.join(
          BFS_ROOT, "results", "heatmap", run_id, f"{run_id}.npz")
      save_pe_phase_cycles(pe_timing_out, phase_cycles, {
          "infile_mtx": os.path.basename(infile_mtx),
          "pe_grid": f"{np_cols}x{np_rows}",
          "source": source,
          "rounds_completed": rounds_completed,
          "max_rounds": max_rounds,
          "profiled_rounds": profiled_rounds,
      }, structural_grids={
          "local_nnz": local_nnz[:, :, 0].astype(np.int64),
          "local_nnz_rows": local_nnz_rows[:, :, 0].astype(np.int64),
          # parent_resolve's own per-PE spatial grid (no round axis -- fires
          # once, at convergence) -- see run_bfs.py's matching comment.
          "parent_resolve_cycles": parent_resolve_grid.astype(np.int64),
      })
      print(f"saved per-PE timing grid to {pe_timing_out}")

    # device_time_cycles (from decode_phase_row) is the whole-run round_trip_
    # start_buffer -> round_trip_done_buffer span, excluding parent_resolve
    # (round_trip_done_buffer is captured before parent_resolve starts --
    # see bool_pe.csl), so it's added back in here; the full
    # search_time_cycles then adds the host transfer brackets on top.
    search_time_cycles_no_transfer = device_time_cycles + int(parent_resolve_cycles.max())
    search_time_cycles = (h2d_seed_span_cycles + search_time_cycles_no_transfer
                           + d2h_span_cycles)
    row["search_time_cycles"] = search_time_cycles
    print(f"[[ search_time_cycles (h2d_seed + device rounds [incl. parent_resolve] "
          f"+ d2h parent readback, docs/GRAPH500_BENCHMARK.md section 3): {search_time_cycles} ]]")

    coo = A_csr.tocoo()
    m, m_convention, search_time_seconds, gteps = compute_m_and_gteps(
        coo, device_visited, is_symmetric, search_time_cycles)
    row["visited_count"] = int(np.sum(device_visited))
    row["m_edges_traversed"] = m
    row["m_convention"] = m_convention
    print(f"[[ visited_count = {row['visited_count']} / {n}, m_edges_traversed = {m} "
          f"({m_convention}) ]]")

    row["clock_freq_hz"] = CLOCK_FREQ_HZ
    row["search_time_seconds"] = search_time_seconds
    row["gteps"] = gteps
    print(f"[[ GTEPS = {m} edges ({m_convention}) / {search_time_seconds * 1e6:.2f} us "
          f"(@{CLOCK_FREQ_HZ/1e6:.0f} MHz) = {gteps:.6f} GTEPS ]]"
          + ("" if is_symmetric else "  -- directed graph: not a Graph500-spec-comparable "
                                      "GTEPS, see m_convention"))

    # search_time_cycles minus the two host-transfer brackets (h2d_seed,
    # d2h) -- isolates on-device work (rounds + transpose + parent_resolve)
    # from host<->device transfer overhead, since those transfers can
    # otherwise dominate search_time_cycles for small/fast graphs.
    row["search_time_cycles_no_transfer"] = search_time_cycles_no_transfer
    _, _, search_time_seconds_no_transfer, gteps_no_transfer = compute_m_and_gteps(
        coo, device_visited, is_symmetric, search_time_cycles_no_transfer)
    row["gteps_no_transfer"] = gteps_no_transfer
    print(f"[[ GTEPS w/o h2d_seed/d2h = {m} edges ({m_convention}) / "
          f"{search_time_seconds_no_transfer * 1e6:.2f} us (@{CLOCK_FREQ_HZ/1e6:.0f} MHz) = "
          f"{gteps_no_transfer:.6f} GTEPS ]]")

    # Console-only, not a CSV column -- worth printing separately since
    # parent_resolve's share of on-device time grows at large scale/grid.
    _, _, search_time_seconds_excl_resolve, gteps_excl_resolve = compute_m_and_gteps(
        coo, device_visited, is_symmetric, device_time_cycles)
    print(f"[[ GTEPS w/o h2d_seed/d2h/parent_resolve = {m} edges ({m_convention}) / "
          f"{search_time_seconds_excl_resolve * 1e6:.2f} us (@{CLOCK_FREQ_HZ/1e6:.0f} MHz) = "
          f"{gteps_excl_resolve:.6f} GTEPS ]]")

    # Host wall-clock seconds, NOT device cycles -- deliberately a separate
    # unit/column from every *_cycles field above, and NOT folded into
    # search_time_cycles/gteps (those stay device+transfer-only, comparable
    # across runs the same way they always were). See docs/ERRORS.md #25.
    row["host_parent_combine_seconds"] = host_parent_combine_seconds

    # Consistency check (see check_round_vs_total_communication's own
    # docstring): sum(round_time - round_compute) across rounds should be
    # close to device_time - total_compute (compute = local_compute +
    # local_term_cond, summed over rounds) -- a large mismatch is a real
    # signal, not just normal per-round straggler variance.
    check_round_vs_total_communication(
        round_duration_cycles, local_compute_max_cycles, local_term_cond_max_cycles,
        device_time_cycles)

    csv_path = args.csv
    if csv_path is None:
      csv_path = os.path.join(BFS_ROOT, "results",
                               "sim" if args.simulator else "hw", "bfs_timing.csv")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    if not write_header:
      with open(csv_path, newline="", encoding="utf-8") as f:
        existing_header = next(csv.reader(f), [])
      assert existing_header == list(row.keys()), (
          f"{csv_path}'s header doesn't match this run's columns (schema changed?) -- "
          "appending would silently misalign columns. Delete/rename the old CSV or pass a "
          "different --csv path.")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=list(row.keys()))
      if write_header:
        writer.writeheader()
      writer.writerow(row)
    print(f"appended timing row to {csv_path}")

    plots_dir = os.path.join(BFS_ROOT, "results",
                              "sim" if args.simulator else "hw")
    out_timing = args.out_timing or plot_bfs_timing.default_out_path(
        plots_dir, row["infile_mtx"], row["pe_grid"], row["source"], row["channels"])
    plot_bfs_timing.plot_timing_row(row, out_timing)


if __name__ == "__main__":
  main()
