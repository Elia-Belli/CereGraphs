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

  Plain `python`, NOT `cs_python` -- cs_python is this repo's LOCAL
  simulator-container wrapper; on the real ALCF cluster, cerebras.sdk.client
  talks to the cluster job scheduler directly over the network from a plain
  Python process (confirmed from ALCF's own docs.alcf.anl.gov/ai-testbed/
  cerebras/csl/ sample output: "Initiating a new SDK compile job against the
  cluster server", "Job id: wsjob-..." -- the CLIENT submits the job, no
  separate bash/qsub/csrun wrapper is shown or needed). sweep_bfs.sh's own
  appliance branch invokes this with plain `python` accordingly.

  Everything host-side that ISN'T runner interaction (preprocess_bool,
  bfs_timing decode, scipy cross-check, tree plot, CSV row, plot_bfs_timing
  call) is UNCHANGED from run_bfs.py -- copied here rather than imported
  only because the runner-interaction section they're interleaved with in
  run_bfs.py's main() had to be rewritten, not because the logic itself
  differs. Keep the two files' non-runner sections in sync by hand if
  bfs_timing.py's schema or preprocess_bool.py's output ever changes.

  NOTE: cerebras.sdk.client / cerebras.appliance are only importable when
  actually connected to a Cerebras appliance -- this script cannot be
  exercised in a simulator-only environment (confirmed: both imports fail
  there). It was written and cross-checked against ALCF's own documented
  compile.py/run.py examples (docs.alcf.anl.gov/ai-testbed/cerebras/csl/,
  v2.10.0 csl-examples) but still needs validation against real hardware,
  not just a read-through -- in particular, the 4th positional argument to
  SdkCompiler.compile() (see device_io.csl_compile_core_appliance's own
  comment) is unverified.

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
from graph_loader import load_graph
from preprocess_bool import preprocess
from scipy.sparse.csgraph import breadth_first_order

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"))
import plot_bfs_timing
from bfs_timing import (CLOCK_FREQ_HZ, NUM_TS_SLOTS, compute_m_and_gteps, compute_skew_adjusted,
                         decode_pe_phase_cycles, decode_phase_row, save_pe_phase_cycles)
from bfs_tree_plot import build_digraph, invalid_parents, render_tree_comparison
from device_io import (csl_compile_core_appliance, derive_visited_from_parent,
                        extract_parent_result, hwl_to_oned_colmajor, single_source_seed_pe)

from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder  # pylint: disable=import-error,no-name-in-module
from cerebras.sdk.client import SdkRuntime  # pylint: disable=import-error,no-name-in-module

DEFAULT_TAU_SWITCH_FRAC = 0.15  # see run_bfs.py's own comment on this constant
ARTIFACT_PATH_FILENAME = "artifact_path.json"


def make_u48(words):
  return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


def read_tic_toc_delta_appliance(runner, sym_tsc_start, sym_tsc_end, height, width):
  """Same as bfs_timing.read_tic_toc_delta, but against the appliance
  MemcpyDataType/MemcpyOrder enums (a different pb2-backed type than the
  simulator pybind module's, even though the values mean the same thing --
  bfs_timing.py's own version imports the pybind one internally, so it
  can't be reused as-is here)."""
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
  parser.add_argument("--arch", help="wse2 or wse3 (default wse2)")
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

  parser.add_argument("--max-rounds", type=int, default=10,
                       help="on-device cap on rounds actually profiled (bool_pe.csl's ts_buf)")
  parser.add_argument("--directional", action="store_true",
                       help="enable the direction-optimizing BFS switch -- see run_bfs.py's own "
                            "flag for the full explanation")
  parser.add_argument("--csv", default=None,
                       help="CSV file to append this run's timing row to "
                            "(default: results/bfs_timing.csv next to this script)")
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
  A_csr = A_csr.sorted_indices()
  assert A_csr.has_sorted_indices == 1, "Error: A is not sorted"

  [nrows, ncols] = A_csr.shape
  assert nrows == ncols, "boolean diagonal-reduce SpMV requires a square matrix"
  n = nrows
  nnz = A_csr.nnz
  assert 0 <= source < n, f"--source={source} out of range [0, {n})"

  print(f"Load matrix A, {nrows}-by-{ncols} with {nnz} nonzeros (structural, boolean)")

  is_symmetric = None
  if need_timing:
    is_symmetric = (A_csr != A_csr.T).nnz == 0
    if not is_symmetric:
      print("[[ NOTE: A_csr is not symmetric -- using the directed edges-traversed formula "
            "instead of Graph500's own undirected dedup rule. See GRAPH500_BENCHMARK.md "
            "section 4. ]]")

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
  local_nnz = matrix_info["local_nnz"]
  local_nnz_cols = matrix_info["local_nnz_cols"]
  local_nnz_rows = matrix_info["local_nnz_rows"]

  blk = math.ceil(n / P)
  bitmap_words = (blk + 31) // 32

  tau_switch_count = None
  if args.directional:
    tau_switch_count = round(DEFAULT_TAU_SWITCH_FRAC * n)
    print(f"--directional: tau_switch_count = {tau_switch_count} "
          f"({DEFAULT_TAU_SWITCH_FRAC * 100:.0f}% of n={n})")

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
      # Real WSE-3 hardware: ALCF's own docs are explicit that this should be
      # the fabric's full physical size, NOT a minimally-computed rectangle
      # (docs.alcf.anl.gov/ai-testbed/cerebras/csl/: "--arch=wse3
      # --fabric-dims=762,1172 --fabric-offsets=4,1" -- "The only difference
      # between CS-3 and simulator run is the fabric_dims. It should be set
      # to minimum required for simulated runs" -- i.e. NOT for real ones).
      fabric_width, fabric_height = 762, 1172
      core_fabric_offset_x, core_fabric_offset_y = 4, 1
  if args.simulator:
    assert fabric_width >= min_fabric_width
    assert fabric_height >= min_fabric_height

  # Unlike run_bfs.py's code_csl (one joined absolute path), the appliance
  # client wants the containing directory and the bare filename separately
  # -- see csl_compile_core_appliance's own docstring.
  csl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
  csl_file = "layout_bool.csl"

  if args.compile_only:
    print("WARNING: compile only -- the appliance's compile server is torn down once this "
          "returns, so SdkRuntime can't be used in this same invocation")
    start = time.time()
    artifact_path = csl_compile_core_appliance(
        csl_dir, csl_file, dirname, fabric_width, fabric_height,
        core_fabric_offset_x, core_fabric_offset_y, args.arch,
        np_cols, np_rows, blk, max_local_nnz, max_local_nnz_cols, max_local_nnz_rows,
        channels, width_west_buf, width_east_buf, max_rounds=max_rounds,
        tau_switch_count=tau_switch_count,
    )
    print(f"Compilation done in {time.time()-start}s", flush=True)
    # {"artifact_path": ...} dict, matching ALCF's own documented format
    # exactly (NOT sdk-hypersparse-spmv/run.appliance.py's bare-string
    # json.dump(hashstr, f), which appears to be an older convention).
    with open(ARTIFACT_PATH_FILENAME, "w", encoding="utf-8") as f:
      json.dump({"artifact_path": artifact_path}, f)
    print(f"dumped artifact_path to {ARTIFACT_PATH_FILENAME}")
    print("COMPILE ONLY: EXIT")
    return

  print(f"load artifact_path from {ARTIFACT_PATH_FILENAME}")
  with open(ARTIFACT_PATH_FILENAME, encoding="utf-8") as f:
    artifact_path = json.load(f)["artifact_path"]

  start = time.time()
  # disable_version_check: see csl_compile_core_appliance's own comment --
  # ALCF's tutorial scripts pass this unconditionally on both Compiler and
  # Runtime.
  with SdkRuntime(artifact_path, simulator=args.simulator, disable_version_check=True) as runner:
    sym_x_bitmap = runner.get_id("x_bitmap")
    sym_parent_local_buf = runner.get_id("parent_local_buf")
    sym_rounds_completed = runner.get_id("rounds_completed")
    sym_mat_rows_buf = runner.get_id("mat_rows_buf")
    sym_mat_col_idx_buf = runner.get_id("mat_col_idx_buf")
    sym_mat_col_loc_buf = runner.get_id("mat_col_loc_buf")
    sym_mat_col_len_buf = runner.get_id("mat_col_len_buf")
    sym_local_nnz = runner.get_id("local_nnz")
    sym_local_nnz_cols = runner.get_id("local_nnz_cols")
    sym_local_nnz_rows = runner.get_id("local_nnz_rows")
    sym_nz_total = runner.get_id("nz_total")
    sym_is_bottom_up_dbg = runner.get_id("is_bottom_up_dbg")
    if need_timing:
      sym_ts_buf = runner.get_id("ts_buf")
      sym_tsc_start_buffer = runner.get_id("tsc_start_buffer")
      sym_tsc_end_buffer = runner.get_id("tsc_end_buffer")
      sym_nf_history = runner.get_id("nf_history")
      sym_direction_history = runner.get_id("direction_history")
      sym_transpose_tic_buffer = runner.get_id("transpose_tic_buffer")
      sym_transpose_toc_buffer = runner.get_id("transpose_toc_buffer")

    # load()/run() are called by SdkRuntime's own __enter__ in appliance mode.

    if need_timing:
      print("enabling tsc...")
      runner.launch("f_enable_tsc", nonblock=False)
      print("timing h2d: matrix structure upload (Graph500-style 'construction')...")
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
                       order=MemcpyOrder.COL_MAJOR, nonblock=not need_timing)

    h2d_matrix_cycles = None
    if need_timing:
      runner.launch("f_toc", nonblock=False)
      h2d_matrix_cycles = read_tic_toc_delta_appliance(
          runner, sym_tsc_start_buffer, sym_tsc_end_buffer, height, width)
      print("timing h2d: seed x upload (Graph500-style per-search cost)...")
      runner.launch("f_tic", nonblock=True)

    runner.memcpy_h2d(sym_x_bitmap, seed_local_x, seed_px, seed_py, 1, 1, bitmap_words,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)

    h2d_seed_cycles = None
    if need_timing:
      runner.launch("f_toc", nonblock=False)
      h2d_seed_cycles = read_tic_toc_delta_appliance(
          runner, sym_tsc_start_buffer, sym_tsc_end_buffer, height, width)

    print("running f_spmv_iter...")
    runner.launch("f_spmv_iter", nonblock=False)

    if need_timing:
      print("timing d2h readback (parent_local_buf -- the real BFS output)...")
      runner.launch("f_tic", nonblock=True)

    parent_local_buf_1d = np.zeros(height * width * blk, np.uint32)
    runner.memcpy_d2h(parent_local_buf_1d, sym_parent_local_buf, 0, 0, width, height, blk,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)

    d2h_cycles = None
    if need_timing:
      runner.launch("f_toc", nonblock=False)
      d2h_cycles = read_tic_toc_delta_appliance(
          runner, sym_tsc_start_buffer, sym_tsc_end_buffer, height, width)

    rounds_buf = np.zeros(height * width, np.uint32)
    runner.memcpy_d2h(rounds_buf, sym_rounds_completed, 0, 0, width, height, 1,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    rounds_completed = int(np.reshape(rounds_buf, (height, width, 1), order="F")[(0, 0, 0)])

    nz_total_buf = np.zeros(height * width, np.float32)
    runner.memcpy_d2h(nz_total_buf, sym_nz_total, 0, 0, width, height, 1,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    final_nz_total = float(np.reshape(nz_total_buf, (height, width, 1), order="F")[(0, 0, 0)])
    is_bottom_up_buf = np.zeros(height * width, np.uint32)
    runner.memcpy_d2h(is_bottom_up_buf, sym_is_bottom_up_dbg, 0, 0, width, height, 1,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    final_is_bottom_up = bool(np.reshape(is_bottom_up_buf, (height, width, 1), order="F")[(0, 0, 0)])
    print(f"[[ direction-optimizing Phase A: final nz_total={final_nz_total}, "
          f"is_bottom_up={final_is_bottom_up}"
          + (f", tau_switch_count={tau_switch_count}" if tau_switch_count is not None else "") + " ]]")

    ts_hwl_u32 = None
    direction_history = None
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
      direction_history_1d = np.zeros(height * width * max_rounds, np.uint32)
      runner.memcpy_d2h(direction_history_1d, sym_direction_history, 0, 0, width, height,
                         max_rounds, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                         order=MemcpyOrder.COL_MAJOR, nonblock=False)
      direction_history_hwl = np.reshape(direction_history_1d, (height, width, max_rounds),
                                         order="F")
      nf_history = nf_history_hwl[0, 0, :]
      direction_history = direction_history_hwl[0, 0, :]

      transpose_cycles = read_tic_toc_delta_appliance(
          runner, sym_transpose_tic_buffer, sym_transpose_toc_buffer, height, width)

    # stop() is called by SdkRuntime's own __exit__ in appliance mode.

  end = time.time()
  print(f"*** Run done in {end-start}s")

  device_parent = extract_parent_result(
      n, blk, P, np.reshape(parent_local_buf_1d, (height, width, blk), order="F"))
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
        os.path.dirname(os.path.abspath(__file__)), "plots", "tree",
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

    for name, cycles in (("h2d_matrix", h2d_matrix_cycles), ("h2d_seed", h2d_seed_cycles),
                         ("d2h", d2h_cycles)):
      row[f"{name}_min_cycles"] = int(cycles.min())
      row[f"{name}_max_cycles"] = int(cycles.max())
      row[f"{name}_avg_cycles"] = f"{cycles.mean():.1f}"
      print(f"  {name:>18s}: min={int(cycles.min())} max={int(cycles.max())} "
            f"avg={cycles.mean():.1f}")

    row_cols, device_time_cycles, profiled_rounds = decode_phase_row(
        ts_hwl_u32, height, width, max_rounds, rounds_completed)
    print(f"rounds_completed = {rounds_completed} (profiled: {profiled_rounds})")
    row.update(row_cols)

    profiled_directions = [int(v) for v in direction_history[:profiled_rounds]]
    profiled_nf = [int(v) for v in nf_history[:profiled_rounds]]
    row["direction_history"] = ";".join(str(v) for v in profiled_directions)
    row["nf_history"] = ";".join(str(v) for v in profiled_nf)
    dir_labels = ["BU" if d else "TD" for d in profiled_directions]
    print(f"  direction per round (TD=top-down, BU=bottom-up): {dir_labels}")
    print(f"  nf per round (this round's own discovery count): {profiled_nf}")

    switch_round = next((i for i, d in enumerate(profiled_directions) if d), None)
    transpose_min_list = [0] * profiled_rounds
    transpose_max_list = [0] * profiled_rounds
    transpose_avg_list = [0.0] * profiled_rounds
    if switch_round is not None:
      transpose_min_list[switch_round] = int(transpose_cycles.min())
      transpose_max_list[switch_round] = int(transpose_cycles.max())
      transpose_avg_list[switch_round] = float(transpose_cycles.mean())
    row["transpose_min_cycles"] = ";".join(str(v) for v in transpose_min_list)
    row["transpose_max_cycles"] = ";".join(str(v) for v in transpose_max_list)
    row["transpose_avg_cycles"] = ";".join(f"{v:.1f}" for v in transpose_avg_list)
    print(f"  {'transpose':>18s}: min={int(transpose_cycles.min())} "
          f"max={int(transpose_cycles.max())} avg={transpose_cycles.mean():.1f}"
          + (f"  (round {switch_round})" if switch_round is not None
             else "  (0 -- switch never fired)"))

    if args.dump_pe_timing:
      phase_cycles, raw_slots, _ = decode_pe_phase_cycles(ts_hwl_u32, height, width, max_rounds,
                                                           rounds_completed)
      skew = compute_skew_adjusted(phase_cycles, raw_slots, height, width)
      print(f"  relay_critical_path_cycles (real relay span, skew excluded): "
            f"{skew['relay_critical_path_cycles'].tolist()}")
      phase_cycles.update(skew)
      phase_cycles.update({f"raw_{name}": grid for name, grid in raw_slots.items()})

      matrix_stem = os.path.splitext(os.path.basename(infile_mtx))[0]
      run_id = f"{matrix_stem}_{np_cols}x{np_rows}_src{source}"
      pe_timing_out = args.pe_timing_out or os.path.join(
          os.path.dirname(os.path.abspath(__file__)), "plots", "heatmap", run_id, f"{run_id}.npz")
      save_pe_phase_cycles(pe_timing_out, phase_cycles, {
          "infile_mtx": os.path.basename(infile_mtx),
          "pe_grid": f"{np_cols}x{np_rows}",
          "source": source,
          "rounds_completed": rounds_completed,
          "max_rounds": max_rounds,
          "profiled_rounds": profiled_rounds,
      }, structural_grids={
          "local_nnz": local_nnz[:, :, 0].astype(np.int64),
          "local_nnz_cols": local_nnz_cols[:, :, 0].astype(np.int64),
          "local_nnz_rows": local_nnz_rows[:, :, 0].astype(np.int64),
      })
      print(f"saved per-PE timing grid to {pe_timing_out}")

    device_time_cycles_with_transpose = device_time_cycles + int(transpose_cycles.max())
    search_time_cycles = (int(h2d_seed_cycles.max()) + device_time_cycles_with_transpose
                           + int(d2h_cycles.max()))
    row["search_time_cycles"] = search_time_cycles
    print(f"[[ search_time_cycles (h2d_seed + device rounds [incl. transpose] + d2h parent "
          f"readback, GRAPH500_BENCHMARK.md section 3): {search_time_cycles} ]]")

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

    row["search_time_cycles_no_transfer"] = device_time_cycles_with_transpose
    _, _, search_time_seconds_no_transfer, gteps_no_transfer = compute_m_and_gteps(
        coo, device_visited, is_symmetric, device_time_cycles_with_transpose)
    row["gteps_no_transfer"] = gteps_no_transfer
    print(f"[[ GTEPS w/o h2d_seed/d2h = {m} edges ({m_convention}) / "
          f"{search_time_seconds_no_transfer * 1e6:.2f} us (@{CLOCK_FREQ_HZ/1e6:.0f} MHz) = "
          f"{gteps_no_transfer:.6f} GTEPS ]]")

    csv_path = args.csv
    if csv_path is None:
      csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                               "bfs_timing.csv")
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

    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    out_timing = args.out_timing or plot_bfs_timing.default_out_path(
        plots_dir, row["infile_mtx"], row["pe_grid"], row["source"], row["channels"])
    plot_bfs_timing.plot_timing_row(row, out_timing)


if __name__ == "__main__":
  main()
