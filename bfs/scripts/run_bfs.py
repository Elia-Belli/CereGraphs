#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args,wrong-import-position
""" run a single-source BFS on bool_diag_spmv's on-device f_spmv_iter kernel
  and report on it three ways -- one compile, one device launch, all three
  reports from that single run:

  1. **tree** (default on, --notree to skip): a two-panel plot comparing
     the on-device BFS tree against scipy.sparse.csgraph.breadth_first_order,
     a fully independent reference (see bfs_tree_plot.py). Saved to
     results/<hw|sim>/tree/.
  2. **correctness** (default on, --nocorrectness to skip): prints the same
     scipy cross-check as (1) as numbers -- visited-set mismatches, invalid
     parents, tie-break differences from scipy's own pick -- without
     needing the plot.
  3. **timing** (default on, --notimings to skip): per-round phase cycle
     counts, h2d/d2h transfer cycles, and a Graph500-style GTEPS estimate
     (see docs/GRAPH500_BENCHMARK.md) -- appended as one row to
     bfs_timing.csv, plus the per-round stacked-bar plot (plot_bfs_timing.py)
     saved to results/<hw|sim>/timing/ (hw vs sim matching --csv, see
     plot_bfs_timing.results_variant).

  The reusable pieces live in their own modules: device_io.py (host<->device
  marshaling), bfs_timing.py (timing constants/decoding), bfs_tree_plot.py
  (tree rendering), plot_bfs_timing.py (timing bar chart, also runnable
  standalone against an existing CSV row).

  How to compile and run (from the repo root)
     cs_python bfs/scripts/run_bfs.py --arch=wse3 --num_pe_cols=8 --num_pe_rows=8
        --channels=1 --driver=<path to cslc> --infile_mtx=<path to mtx file>
        --source=0
     cs_python bfs/scripts/run_bfs.py ... --notree                 # timing + correctness only
     cs_python bfs/scripts/run_bfs.py ... --notimings --nocorrectness  # tree only
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime, timezone

import networkx as nx
import numpy as np

# device_io.py/graph_loader.py/preprocess_bool.py/bfs_timing.py live in
# ../implementation/, plot_bfs_timing.py/bfs_tree_plot.py in ../plots/ --
# add both to sys.path so the imports below resolve as siblings.
BFS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BFS_ROOT, "implementation"))
sys.path.insert(0, os.path.join(BFS_ROOT, "plots"))

# Above this many vertices, the scipy cross-check (breadth_first_order() +
# rebuilding a transposed CSR copy of A) risks OOMing under this host's
# 100GiB per-user cgroup limit (docs/ERRORS.md #26) -- confirmed for real at
# RMAT s24 @ 750x750 (n=16,777,500): the run printed NOTHING at all (not
# even "Run done in Xs", which always appears before the scipy check
# starts) before getting OOM-killed at ~99.8GiB anon-rss (dmesg-confirmed),
# even though preprocess() and the on-device execution both succeed fine at
# that scale on their own. RMAT s20's own n (1,049,250) is the largest scale
# this project has verified the scipy cross-check itself against without
# incident, so that's the threshold -- see docs/ERRORS.md #28.
_SCIPY_CHECK_AUTO_DISABLE_N = 1_100_000

from graph_loader import load_graph
from preprocess_bool import preprocess
from scipy.sparse.csgraph import breadth_first_order

import plot_bfs_timing
from bfs_timing import (CLOCK_FREQ_HZ, NUM_TS_SLOTS, check_round_vs_total_communication,
                         compute_m_and_gteps, decode_pe_phase_cycles, decode_phase_row,
                         read_tic_toc_delta, save_pe_phase_cycles, timed_transfer)
from bfs_tree_plot import build_digraph, invalid_parents, render_tree_comparison
from device_io import (csl_compile_core, derive_visited_from_parent,
                        extract_parent_result, hwl_to_oned_colmajor, prepare_h2d_chunked,
                        send_h2d_chunked, single_source_seed_pe)

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)

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
  parser.add_argument("--arch", choices=["wse3"], default="wse3",
                       help="WSE-3 only -- this kernel is no longer tested/supported on WSE-2")
  parser.add_argument("--latestlink", default="out/latest", help="folder for the compiled ELFs")
  parser.add_argument("--source", type=int, default=0, help="single BFS source vertex")

  parser.add_argument("--notree", action="store_true", help="skip the tree comparison plot")
  parser.add_argument("--notimings", action="store_true",
                       help="skip per-round timing/GTEPS (also skips the h2d/d2h tsc "
                            "instrumentation itself, saving the real transfer time it costs)")
  parser.add_argument("--nocorrectness", action="store_true",
                       help="skip printing the scipy cross-check numbers (visited mismatches, "
                            "invalid parents, tie-break diffs). Note: if --notree is NOT set, "
                            "the tree plot still needs -- and computes -- the scipy reference "
                            "regardless, since it's the plot's left panel; this flag only "
                            "silences the printed summary")
  parser.add_argument("--force-scipy", action="store_true",
                       help=f"run the scipy cross-check (and/or tree plot) even above "
                            f"{_SCIPY_CHECK_AUTO_DISABLE_N} vertices, where it's auto-disabled by "
                            f"default (real OOM risk under this host's 100GiB per-user cgroup "
                            f"limit -- confirmed at RMAT s24 @ 750x750, see docs/ERRORS.md #28). "
                            f"Only pass this if you've checked host memory headroom yourself.")

  parser.add_argument("--max-rounds", type=int, default=10,
                       help="on-device cap on rounds actually profiled for --notimings=False "
                            "(bool_pe.csl's ts_buf) -- rounds beyond this still run correctly, "
                            "just aren't timestamped; bump this if a run reports truncation")
  parser.add_argument("--csv", default=None,
                       help="CSV file to append this run's timing row to "
                            "(default: results/sim/bfs_timing.csv, a sibling of this script's "
                            "own scripts/ directory)")
  parser.add_argument("--out-tree", default=None,
                       help="tree plot output path (default: results/<hw|sim>/tree/<matrix>_<grid>_"
                            "src<N>.png, hw vs sim matching --csv, see plot_bfs_timing."
                            "results_variant)")
  parser.add_argument("--out-timing", default=None,
                       help="timing plot output path (default: results/<hw|sim>/timing/timing_"
                            "<matrix>_<grid>_src<N>_ch<C>.png, hw vs sim matching --csv)")
  parser.add_argument("--no-show-parent-mismatch", dest="show_parent_mismatch",
                       action="store_false",
                       help="don't color-highlight (orange, tree plot only) nodes where our "
                            "parent choice differs from scipy's own breadth_first_order pick -- "
                            "on by default. These are EXPECTED whenever a node has multiple "
                            "valid predecessors (see bfs_tree_plot.invalid_parents()'s "
                            "docstring), not a bug")
  parser.set_defaults(show_parent_mismatch=True)

  parser.add_argument("--dump-pe-timing", action="store_true",
                       help="save the full per-PE-per-round-per-phase cycle grid to a .npz file "
                            "(default off -- diagnostic only, for offline per-PE analysis; not "
                            "part of the default tree/timing/correctness reports)")
  parser.add_argument("--pe-timing-out", default=None,
                       help="path for --dump-pe-timing's .npz output (default: results/<hw|sim>/"
                            "heatmap/<matrix>_<grid>_src<N>/<matrix>_<grid>_src<N>.npz; hw vs sim "
                            "matching --csv)")
  return parser.parse_args()


def _default_csv_path():
  """results/sim/bfs_timing.csv, sibling of scripts/ -- a hw run always
  passes --csv explicitly (see plot_bfs_timing_poster.py's
  default_out_path), so this default is sim-only."""
  return os.path.join(BFS_ROOT, "results", "sim", "bfs_timing.csv")


def main():
  """Main method to run the example code."""

  args = parse_args()
  need_timing = not args.notimings
  need_scipy = not args.notree or not args.nocorrectness

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

  # Only auto-disable when the tree plot wasn't requested either -- if
  # --notree is NOT set, need_scipy is guaranteed True regardless of
  # --nocorrectness (the tree plot's own left panel needs it), and
  # render_tree_comparison() below is called unconditionally on
  # scipy_parent/scipy_visited/scipy_levels; silently forcing need_scipy
  # False here would pass it None data instead of skipping cleanly. Tree
  # plots are only ever requested at smoke scale in practice, so this
  # just means an explicit --notree big-graph run is what actually gets
  # the auto-disable, matching how this threshold was discovered.
  if need_scipy and args.notree and n > _SCIPY_CHECK_AUTO_DISABLE_N and not args.force_scipy:
    print(f"[[ NOTE: scipy cross-check auto-disabled -- n={n} exceeds the RMAT-s20-scale "
          f"threshold ({_SCIPY_CHECK_AUTO_DISABLE_N}) where scipy's own breadth_first_order() "
          f"plus rebuilding a transposed CSR copy of A risks OOMing under this host's 100GiB "
          f"per-user cgroup limit (real, dmesg-confirmed at RMAT s24 @ 750x750, see "
          f"docs/ERRORS.md #28) -- pass --force-scipy to run it anyway. ]]")
    need_scipy = False

  is_symmetric = None
  if need_timing:
    # Graph500's m formula (docs/GRAPH500_BENCHMARK.md sec 4) assumes a
    # symmetrized graph -- true for gen_rmat.py's output but not guaranteed
    # for an arbitrary --infile_mtx; only GTEPS's m needs it, so only
    # computed when timing is actually wanted.
    is_symmetric = (A_csr != A_csr.T).nnz == 0
    if not is_symmetric:
      print("[[ NOTE: A_csr is not symmetric (a directed graph, not gen_rmat.py's undirected "
            "style) -- using the directed edges-traversed formula instead of Graph500's own "
            "undirected dedup rule; not directly comparable to a Graph500-spec TEPS number, but "
            "still a real, meaningful edges-traversed count for this graph. See "
            "docs/GRAPH500_BENCHMARK.md section 4. ]]")

  # Bottom-up is the only traversal strategy this kernel runs (see
  # docs/ERRORS.md #21) -- the host uploads the matrix ALREADY in
  # CSR-by-destination form, so preprocess() is fed A_csr's arrays into its
  # cscColPtr/cscRowInd parameter slot (not a mismatched name -- see
  # preprocess_bool.py's own top-of-function comment): for a square matrix
  # on a square grid with .sorted_indices() applied, csc(A^T) == csr(A),
  # so treating A_csr's own CSR structure as if it were "CSC of A^T" makes
  # preprocess() build its per-nonzero "column-grouped" structure for A^T
  # -- whose columns are A's rows -- with no on-device transpose needed.
  # preprocess() itself needs no code change for this; only which
  # physical array's structure it's told to treat as CSC-ordered.
  # A SEPARATE CSC representation of A used to be built here too (feeding
  # preprocess()'s now-removed csrRowPtr/csrColInd parameter pair) -- that
  # parameter pair fed a full second full-nnz sort whose only output
  # (local_nzrows/max_local_nnz_rows) turned out to be read by NEITHER
  # this script nor run_bfs.appliance.py (see docs/ERRORS.md #26 follow-up
  # for the full trace) -- removed entirely, along with the .tocsc() call
  # and its own .sorted_indices() pass that used to be needed to build it.
  matrix_info = preprocess(
      nrows, ncols, nnz, np_cols, np_rows,
      A_csr.indptr, A_csr.indices,
  )

  max_local_nnz = matrix_info["max_local_nnz"]
  # matrix_info["max_local_nnz_cols"]/["mat_col_*_buf"]/["local_nnz_cols"]
  # are the CSR-by-destination fields (see the swap comment above) -- this
  # script's own "_rows" naming below is post-swap, matching every other
  # extraction in this function, not a typo.
  max_local_nnz_rows = matrix_info["max_local_nnz_cols"]
  # preprocess()'s own block_id math (row_b*fabx+col_b) treats whichever
  # array was fed into the cscColPtr/cscRowInd role as the FIRST reshape
  # axis -- under the swap above that's the CSR-role data (A_csr, indexed
  # by ROW), so its per-nonzero "row_b"/"col_b" internals end up meaning
  # actual (px, py) instead of the normal (py, px). Confirmed empirically
  # (a standalone 2x2/4x4/8x8-grid test comparing every PE's block against
  # A_csr's own ground truth) -- caught a real bug here: without this
  # transpose, every off-diagonal PE silently received its transpose
  # partner's block, passing compile but producing wrong BFS results
  # (visited-set mismatches vs scipy). local_nnz's own per-PE count is
  # computed from the same (px,py)-ordered internals, so it needs the same
  # fix even though nothing on-device reads it anymore (compute_topdown()/
  # transpose_structure(), local_nnz[0]'s only consumers, are both
  # removed) -- kept transposed for the host-side structural-grid
  # diagnostic dump (--dump-pe-timing) to stay meaningful.
  mat_rows_buf = np.transpose(matrix_info["mat_rows_buf"], (1, 0, 2))
  mat_row_idx_buf = np.transpose(matrix_info["mat_col_idx_buf"], (1, 0, 2))
  mat_row_loc_buf = np.transpose(matrix_info["mat_col_loc_buf"], (1, 0, 2))
  mat_row_len_buf = np.transpose(matrix_info["mat_col_len_buf"], (1, 0, 2))
  local_nnz = np.transpose(matrix_info["local_nnz"], (1, 0, 2))
  local_nnz_rows = np.transpose(matrix_info["local_nnz_cols"], (1, 0, 2))

  blk = math.ceil(n / P)
  bitmap_words = (blk + 31) // 32

  # Single-source seed, not a dense multi-source frontier (which would mask
  # most rounds' cost behind one giant first round, and wouldn't be a single
  # tree). Only the PE owning `source` needs a real host write -- see
  # single_source_seed_pe()'s docstring for why every other PE's x_bitmap is
  # already zero.
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
    fabric_width = min_fabric_width
    fabric_height = min_fabric_height
  assert fabric_width >= min_fabric_width
  assert fabric_height >= min_fabric_height

  code_csl = os.path.join(BFS_ROOT, "implementation", "src", "layout_bool.csl")

  start = time.time()
  csl_compile_core(
      cslc, code_csl, dirname, fabric_width, fabric_height,
      core_fabric_offset_x, core_fabric_offset_y, args.run_only, args.arch,
      np_cols, np_rows, blk, max_local_nnz, max_local_nnz_rows,
      channels, width_west_buf, width_east_buf, max_rounds=max_rounds,
  )
  print(f"Compilation done in {time.time()-start}s", flush=True)

  if args.compile_only:
    print("COMPILE ONLY: EXIT")
    return

  runner = SdkRuntime(dirname, cmaddr=args.cmaddr, simfab_numthreads=64, suppress_simfab_trace=True)

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

  runner.load()
  runner.run()

  # All host-side marshaling happens here, before the timed bracket below --
  # see bfs_timing.timed_transfer's docstring: keeping marshaling out of the
  # tic/toc window avoids counting host reshape time as transfer time.
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
                       max_local_nnz_rows, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=True)
    runner.memcpy_h2d(sym_mat_row_loc_buf, mat_row_loc_buf_1d, 0, 0, width, height,
                       max_local_nnz_rows, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=True)
    runner.memcpy_h2d(sym_mat_row_len_buf, mat_row_len_buf_1d, 0, 0, width, height,
                       max_local_nnz_rows, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=True)
    runner.memcpy_h2d(sym_local_nnz, local_nnz_1d, 0, 0, width, height, 1,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=True)
    runner.memcpy_h2d(sym_local_nnz_rows, local_nnz_rows_1d, 0, 0, width, height, 1,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=not need_timing)

  h2d_matrix_span_cycles = timed_transfer(
      runner, height, width, _send_h2d_matrix, need_timing=need_timing,
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
      sym_tsc_start=sym_tsc_start_buffer if need_timing else None,
      sym_tsc_end=sym_tsc_end_buffer if need_timing else None,
      sym_tsc_ref=sym_tsc_ref_buffer if need_timing else None)

  print("running f_spmv_iter...")
  runner.launch("f_spmv_iter", nonblock=False)

  if need_timing:
    # Graph500's spec output is exactly the predecessor/parent array (docs/
    # GRAPH500_BENCHMARK.md sec 1) -- no separate "visited" readback needed,
    # since derive_visited_from_parent() recovers it from the parent vector
    # alone. So only this one transfer is timed as the output cost.
    print("timing d2h readback (parent_values -- the real BFS output)...")

  # #24 removed the on-device relay that used to resolve each row's P
  # per-PE candidates to a single winner at PE-column MID -- the host now
  # reads EVERY PE's own compact parent_values array and combines per-row
  # itself (see extract_parent_result()'s own comment). Real cost of that:
  # this transfer is now width*height*max_local_nnz_rows elements instead
  # of one column's height*blk -- ~P times more data, still far under the
  # d2h gRPC ~2GiB message-size ceiling (docs/GRAPH500_BENCHMARK.md) at any
  # scale this kernel targets. #26 halved that again: parent_values is now
  # u16 (a local column offset, not a u32 global vertex id), so this is a
  # 16-bit transfer, not 32-bit -- data_type=MEMCPY_16BIT below reflects
  # the DEVICE-side symbol's width. The HOST-side numpy buffer still has
  # to be uint32 regardless (same SDK requirement every other u16 device
  # buffer in this file already works around -- see mat_row_idx_buf_1d/
  # rounds_buf/ts_buf_1d/nf_history_1d's own np.uint32 buffers alongside
  # their own MEMCPY_16BIT transfers); passing uint16 here instead throws
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
      sym_tsc_start=sym_tsc_start_buffer if need_timing else None,
      sym_tsc_end=sym_tsc_end_buffer if need_timing else None,
      sym_tsc_ref=sym_tsc_ref_buffer if need_timing else None)

  # Needed regardless of --notree/--notimings (tree plot's round-count
  # label, timing's phase decoding) -- always read.
  rounds_buf = np.zeros(height * width, np.uint32)
  runner.memcpy_d2h(rounds_buf, sym_rounds_completed, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)
  rounds_completed = int(np.reshape(rounds_buf, (height, width, 1), order="F")[(0, 0, 0)])

  # How many of each PE's blk local rows already have a real parent
  # candidate right before the end-of-run reduce_select_any call. Read back
  # grid-wide (not just PE(0,0)) since occupancy is genuinely per-PE, unlike
  # nf_history which is identical everywhere.
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

    # Frontier size is identical on every PE by construction (flooded by
    # bool_pe.csl's term_col_bcast_done()), so reading PE(0,0)'s copy alone
    # is enough -- no aggregation needed.
    nf_history_1d = np.zeros(height * width * max_rounds, np.uint32)
    runner.memcpy_d2h(nf_history_1d, sym_nf_history, 0, 0, width, height, max_rounds,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    nf_history_hwl = np.reshape(nf_history_1d, (height, width, max_rounds), order="F")
    nf_history = nf_history_hwl[0, 0, :]

    # mpi_x.reduce_select_any()'s one-time end-of-run cost.
    parent_resolve_cycles = read_tic_toc_delta(
        runner, sym_parent_resolve_tic_buffer, sym_parent_resolve_toc_buffer, height, width)
    # Spatial (row, col) view, to check the relay's critical path is really
    # root-relay-position-driven rather than random skew. read_tic_toc_delta
    # flattens in C order (not the "F" order used for the raw device-buffer
    # reshape earlier), so reshaping back here must use the default (C)
    # order, not "F".
    parent_resolve_grid = parent_resolve_cycles.reshape((height, width))

    # Round-trip span that stays correct even when the run has more rounds
    # than max_rounds can profile in detail (see round_trip_start/done_buffer
    # in bool_pe.csl and decode_phase_row).
    round_trip_cycles = read_tic_toc_delta(
        runner, sym_round_trip_start_buffer, sym_round_trip_done_buffer, height, width)

  runner.stop()

  parent_values_hwl = np.reshape(parent_values_1d, (height, width, max_local_nnz_rows), order="F")
  # #24 moved the real per-row parent combine off-device entirely -- the
  # on-device "parent_resolve" TSC bracket (parent_resolve_tic/toc_buffer)
  # now only brackets a trivial occupancy-count scan, NOT this. Without a
  # host-side timer, this cost was invisible to every measurement (it runs
  # after search_time_cycles/GTEPS are already computed from device+
  # transfer cycles alone) -- time it explicitly here instead.
  host_parent_combine_start = time.time()
  device_parent = extract_parent_result(
      n, blk, mat_row_idx_buf, local_nnz_rows, parent_values_hwl)
  host_parent_combine_seconds = time.time() - host_parent_combine_start
  print(f"[[ host_parent_combine_seconds: {host_parent_combine_seconds:.6f}s (host-side "
        f"per-row combine over parent_values_hwl, replacing the on-device relay removed in "
        f"docs/ERRORS.md #24 -- wall-clock, NOT device cycles; not included in "
        f"search_time_cycles/gteps, which stay device+transfer-only) ]]")
  device_parent[source] = source  # root, not "undiscovered" -- see bool_pe.csl's module docstring
  device_visited = derive_visited_from_parent(n, device_parent, source)

  scipy_parent = scipy_visited = scipy_levels = None
  mismatch = n_mismatch = scipy_ok = scipy_diff_device = None
  if need_scipy:
    # Fully independent reference: scipy's own BFS, with no dependency on
    # bool_pe.csl/preprocess_bool.py/the CSL matrix partitioning -- catches
    # bugs shared by the on-device pipeline's own building blocks.
    print("scipy reference: breadth_first_order")
    # transpose because A_csr is row=dest/col=source but breadth_first_order
    # needs row=source/col=dest (csgraph[i,j] != 0 means edge i -> j).
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
        plot_bfs_timing.results_variant(args.csv or _default_csv_path()), "tree",
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

    # parent_resolve is on-device only (no host-device transfer), so the
    # per-PE-max-of-self-delta approach is fine here -- no cross-PE clock
    # sync needed for a quantity that never leaves the fabric.
    for name, cycles in (("parent_resolve", parent_resolve_cycles),):
      row[f"{name}_min_cycles"] = int(cycles.min())
      row[f"{name}_max_cycles"] = int(cycles.max())
      row[f"{name}_avg_cycles"] = f"{cycles.mean():.1f}"
      print(f"  {name:>18s}: min={int(cycles.min())} max={int(cycles.max())} "
            f"avg={cycles.mean():.1f}")

    # Sync-corrected cross-PE span (bfs_timing.read_sync_corrected_span) --
    # the true max(toc)-min(tic) across all PEs. The only h2d/d2h timing
    # recorded: the per-PE-max-of-self-delta approach it replaced is a
    # structural lower bound on this span, never more accurate.
    for name, span in (("h2d_matrix", h2d_matrix_span_cycles),
                        ("h2d_seed", h2d_seed_span_cycles), ("d2h", d2h_span_cycles)):
      row[f"{name}_span_cycles"] = span
      print(f"  {name:>18s}: sync-corrected span={span}")

    (row_cols, device_time_cycles, profiled_rounds, round_duration_cycles,
     local_compute_max_cycles, local_term_cond_max_cycles) = decode_phase_row(
        ts_hwl_u32, height, width, max_rounds, rounds_completed, round_trip_cycles)
    print(f"rounds_completed = {rounds_completed} (profiled: {profiled_rounds})")
    row.update(row_cols)

    # Per-round frontier size, alongside the phase timing decoded above --
    # see bool_pe.csl's nf_history.
    profiled_nf = [int(v) for v in nf_history[:profiled_rounds]]
    row["nf_history"] = ";".join(str(v) for v in profiled_nf)
    print(f"  nf per round (this round's own discovery count): {profiled_nf}")

    if args.dump_pe_timing:
      phase_cycles, round_start, round_end, _ = decode_pe_phase_cycles(
          ts_hwl_u32, height, width, max_rounds, rounds_completed)
      # Raw per-PE local_compute/local_term_cond grids, plus the two raw
      # round-boundary grids (round_start/round_end) round_time is built
      # from.
      phase_cycles.update({"raw_round_start": round_start, "raw_round_end": round_end})

      matrix_stem = os.path.splitext(os.path.basename(infile_mtx))[0]
      run_id = f"{matrix_stem}_{np_cols}x{np_rows}_src{source}"
      # Lives inside results/<hw|sim>/heatmap/<run_id>/. hw vs sim mirrors
      # --csv's own results/hw or results/sim (see
      # plot_bfs_timing.results_variant), matching out_timing's default
      # below.
      pe_timing_out = args.pe_timing_out or os.path.join(
          BFS_ROOT, "results",
          plot_bfs_timing.results_variant(args.csv or _default_csv_path()),
          "heatmap", run_id, f"{run_id}.npz")
      save_pe_phase_cycles(pe_timing_out, phase_cycles, {
          "infile_mtx": os.path.basename(infile_mtx),
          "pe_grid": f"{np_cols}x{np_rows}",
          "source": source,
          "rounds_completed": rounds_completed,
          "max_rounds": max_rounds,
          "profiled_rounds": profiled_rounds,
      }, structural_grids={
          # Host-side partition structure (preprocess_bool.py) -- check by
          # eye whether a phase's per-PE imbalance tracks the matrix's own
          # sparsity distribution.
          "local_nnz": local_nnz[:, :, 0].astype(np.int64),
          "local_nnz_rows": local_nnz_rows[:, :, 0].astype(np.int64),
          # parent_resolve's per-PE spatial grid (no round axis -- fires
          # once, at convergence) -- confirms whether its min/max spread
          # (docs/GRAPH500_BENCHMARK.md) is root-relay-hop-distance-driven
          # rather than random idle-wait skew.
          "parent_resolve_cycles": parent_resolve_grid.astype(np.int64),
      })
      print(f"saved per-PE timing grid to {pe_timing_out}")

    # device_time_cycles is the whole-run round_trip_start->done_buffer span,
    # excluding parent_resolve (round_trip_done_buffer is captured before it
    # starts -- see bool_pe.csl), so it's added back in below;
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

    # search_time_cycles minus the h2d_seed/d2h brackets -- isolates
    # on-device work from transfer overhead, which can otherwise dominate
    # for small/fast graphs.
    row["search_time_cycles_no_transfer"] = search_time_cycles_no_transfer
    _, _, search_time_seconds_no_transfer, gteps_no_transfer = compute_m_and_gteps(
        coo, device_visited, is_symmetric, search_time_cycles_no_transfer)
    row["gteps_no_transfer"] = gteps_no_transfer
    print(f"[[ GTEPS w/o h2d_seed/d2h = {m} edges ({m_convention}) / "
          f"{search_time_seconds_no_transfer * 1e6:.2f} us (@{CLOCK_FREQ_HZ/1e6:.0f} MHz) = "
          f"{gteps_no_transfer:.6f} GTEPS ]]")

    # Console-only, not a CSV column: device_time_cycles already excludes
    # parent_resolve (a single end-of-run step whose share of on-device
    # time grows large at big scale/grid) -- print the further-excluded
    # figure too so terminal output isn't misleadingly dominated by it.
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

    # Consistency check (see check_round_vs_total_communication):
    # sum(round_time - round_compute) across rounds should be close to
    # device_time - total_compute -- a large mismatch is a real signal, not
    # normal straggler variance.
    check_round_vs_total_communication(
        round_duration_cycles, local_compute_max_cycles, local_term_cond_max_cycles,
        device_time_cycles)

    csv_path = args.csv or _default_csv_path()
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
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

    plots_dir = os.path.join(BFS_ROOT, "results",
                              plot_bfs_timing.results_variant(csv_path))
    out_timing = args.out_timing or plot_bfs_timing.default_out_path(
        plots_dir, row["infile_mtx"], row["pe_grid"], row["source"], row["channels"])
    plot_bfs_timing.plot_timing_row(row, out_timing)


if __name__ == "__main__":
  main()
