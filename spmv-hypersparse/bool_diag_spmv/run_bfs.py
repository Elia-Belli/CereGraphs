#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args,wrong-import-position
""" run a single-source BFS on bool_diag_spmv's on-device f_spmv_iter kernel
  and report on it three ways -- one compile, one device launch, all three
  reports from that single run:

  1. **tree** (default on, --notree to skip): a two-panel plot comparing
     the on-device BFS tree against scipy.sparse.csgraph.breadth_first_order,
     a fully independent reference (see bfs_tree_plot.py). Saved to
     plots/tree/.
  2. **correctness** (default on, --nocorrectness to skip): prints the same
     scipy cross-check as (1) as numbers -- visited-set mismatches, invalid
     parents, tie-break differences from scipy's own pick -- without
     needing the plot. This is a single-source, real-BFS-shaped check;
     run_host_driven_bfs.py (formerly test_iterative.py) is the *stress
     test* for the iterative machinery itself (random multi-source
     frontier, checked bit-for-bit against a host-driven baseline) and is
     intentionally kept as its own separate script, not folded in here.
  3. **timing** (default on, --notimings to skip): per-round phase cycle
     counts (bfs_timing.py/record_ts()), h2d/d2h transfer cycles, and a
     Graph500-style GTEPS estimate (see GRAPH500_BENCHMARK.md) -- appended
     as one row to bfs_timing.csv, plus the per-round stacked-bar plot
     (plot_bfs_timing.py, h2d/rounds/d2h/local_compute-split panels side by
     side in one PNG) saved to plots/timing/.

  Replaces plot_bfs_tree.py and bench_timing.py (deleted -- this script
  does both, without the double compile+launch cost of running them
  separately). The reusable pieces those two scripts were built from now
  live in their own modules: device_io.py (host<->device marshaling),
  bfs_timing.py (timing constants/decoding), bfs_tree_plot.py (tree
  rendering), plot_bfs_timing.py (timing bar chart, also still runnable
  standalone against an existing CSV row).

  How to compile and run
     cs_python run_bfs.py --arch=wse3 --num_pe_cols=8 --num_pe_rows=8
        --channels=1 --driver=<path to cslc> --infile_mtx=<path to mtx file>
        --source=0
     cs_python run_bfs.py ... --notree                 # timing + correctness only
     cs_python run_bfs.py ... --notimings --nocorrectness  # tree only
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
from preprocess_bool import preprocess
from scipy.io import mmread
from scipy.sparse.csgraph import breadth_first_order

# plot_bfs_timing.py/bfs_tree_plot.py live in plots/ (see that folder's own
# scripts for the matching bootstrap back to this directory) -- add it to
# sys.path so the plain imports below keep working regardless of it being a
# sibling directory now, not this same one.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"))
import plot_bfs_timing
from bfs_timing import (CLOCK_FREQ_HZ, NUM_TS_SLOTS, compute_m_and_gteps, compute_skew_adjusted,
                         decode_pe_phase_cycles, decode_phase_row, read_tic_toc_delta,
                         save_pe_phase_cycles)
from bfs_tree_plot import build_digraph, invalid_parents, render_tree_comparison
from device_io import (csl_compile_core, derive_visited_from_parent,
                        extract_parent_result, hwl_to_oned_colmajor, single_source_seed_pe)

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
  parser.add_argument("--arch", help="wse2 or wse3 (default wse2)")
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

  parser.add_argument("--max-rounds", type=int, default=10,
                       help="on-device cap on rounds actually profiled for --notimings=False "
                            "(bool_pe.csl's ts_buf) -- rounds beyond this still run correctly, "
                            "just aren't timestamped; bump this if a run reports truncation")
  parser.add_argument("--csv", default=None,
                       help="CSV file to append this run's timing row to "
                            "(default: results/bfs_timing.csv next to this script)")
  parser.add_argument("--out-tree", default=None,
                       help="tree plot output path (default: plots/tree/<matrix>_<grid>_src<N>.png)")
  parser.add_argument("--out-timing", default=None,
                       help="timing plot output path (default: plots/timing/timing_<matrix>_"
                            "<grid>_src<N>_ch<C>.png)")
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
                            "(default off -- diagnostic only, for plot_pe_heatmap.py; not part "
                            "of the default tree/timing/correctness reports)")
  parser.add_argument("--pe-timing-out", default=None,
                       help="path for --dump-pe-timing's .npz output (default: plots/heatmap/"
                            "<matrix>_<grid>_src<N>/<matrix>_<grid>_src<N>.npz -- the same "
                            "per-run folder plot_pe_heatmap.py renders its PNGs into)")
  return parser.parse_args()


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

  is_symmetric = None
  if need_timing:
    # Graph500's own m formula (the undirected dedup rule, see
    # GRAPH500_BENCHMARK.md section 4) only makes sense for a symmetrized
    # graph -- true for gen_rmat.py's output but NOT guaranteed for an
    # arbitrary --infile_mtx. The BFS kernel itself has no such requirement
    # (correct on any square boolean matrix, directed or not) -- only
    # Graph500-style edge counting cares. Only needed for GTEPS's m, so
    # only computed when timing is actually wanted.
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
  local_nnz = matrix_info["local_nnz"]
  local_nnz_cols = matrix_info["local_nnz_cols"]
  local_nnz_rows = matrix_info["local_nnz_rows"]

  blk = math.ceil(n / P)
  bitmap_words = (blk + 31) // 32

  # single-source seed -- a real BFS workload, not run_host_driven_bfs.py's
  # random ~50%-density stress frontier (which would mask most rounds'
  # costs behind one giant first round, and wouldn't be a single tree).
  # Only the one diagonal PE owning `source` needs a real host write -- see
  # single_source_seed_pe()'s own docstring for why every other PE's x_bitmap
  # is already provably zero.
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

  runner = SdkRuntime(dirname, cmaddr=args.cmaddr, simfab_numthreads=64, suppress_simfab_trace=True)

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
  if need_timing:
    sym_ts_buf = runner.get_id("ts_buf")
    sym_tsc_start_buffer = runner.get_id("tsc_start_buffer")
    sym_tsc_end_buffer = runner.get_id("tsc_end_buffer")

  runner.load()
  runner.run()

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
    runner.launch("f_toc", nonblock=False)  # blocks -> every matrix-structure h2d above is done
    h2d_matrix_cycles = read_tic_toc_delta(
        runner, sym_tsc_start_buffer, sym_tsc_end_buffer, height, width)
    print("timing h2d: seed x upload (Graph500-style per-search cost)...")
    runner.launch("f_tic", nonblock=True)

  runner.memcpy_h2d(sym_x_bitmap, seed_local_x, seed_px, seed_py, 1, 1, bitmap_words,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)

  h2d_seed_cycles = None
  if need_timing:
    runner.launch("f_toc", nonblock=False)  # blocks -> seed x h2d above is done
    h2d_seed_cycles = read_tic_toc_delta(
        runner, sym_tsc_start_buffer, sym_tsc_end_buffer, height, width)

  print("running f_spmv_iter...")
  runner.launch("f_spmv_iter", nonblock=False)

  if need_timing:
    # Graph500's own output is exactly the predecessor/parent array (see
    # GRAPH500_BENCHMARK.md section 1 -- the reference implementation's
    # run_bfs(root, pred) signature) -- no separate "visited" readback is
    # part of the spec, and derive_visited_from_parent() below recovers it
    # from parent_local_buf alone, so only that one transfer needs to be
    # timed as the search's "output written to memory" cost.
    print("timing d2h readback (parent_local_buf -- the real BFS output)...")
    runner.launch("f_tic", nonblock=True)

  parent_local_buf_1d = np.zeros(height * width * blk, np.uint32)
  runner.memcpy_d2h(parent_local_buf_1d, sym_parent_local_buf, 0, 0, width, height, blk,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)

  d2h_cycles = None
  if need_timing:
    runner.launch("f_toc", nonblock=False)  # blocks -> the d2h read above is done
    d2h_cycles = read_tic_toc_delta(runner, sym_tsc_start_buffer, sym_tsc_end_buffer, height, width)

  # rounds_completed is needed regardless (tree plot's round-count label,
  # timing's phase decoding) -- always read.
  rounds_buf = np.zeros(height * width, np.uint32)
  runner.memcpy_d2h(rounds_buf, sym_rounds_completed, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)
  rounds_completed = int(np.reshape(rounds_buf, (height, width, 1), order="F")[(0, 0, 0)])

  ts_hwl_u32 = None
  if need_timing:
    ts_len = max_rounds * NUM_TS_SLOTS * 3
    ts_buf_1d = np.zeros(height * width * ts_len, np.uint32)
    runner.memcpy_d2h(ts_buf_1d, sym_ts_buf, 0, 0, width, height, ts_len,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    ts_hwl_u32 = np.reshape(ts_buf_1d, (height, width, ts_len), order="F")

  runner.stop()

  device_parent = extract_parent_result(
      n, blk, P, np.reshape(parent_local_buf_1d, (height, width, blk), order="F"))
  device_parent[source] = source  # root, not "undiscovered" -- see bool_pe.csl's module docstring
  device_visited = derive_visited_from_parent(n, device_parent, source)

  scipy_parent = scipy_visited = scipy_levels = None
  mismatch = n_mismatch = scipy_ok = scipy_diff_device = None
  if need_scipy:
    # Fully independent reference: scipy's own BFS, with zero dependency on
    # bool_pe.csl, preprocess_bool.py, or the CSL matrix partitioning -- a
    # bug shared by the on-device pipeline's own building blocks would
    # never show up as a self-comparison.
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

    if args.dump_pe_timing:
      phase_cycles, raw_slots, _ = decode_pe_phase_cycles(ts_hwl_u32, height, width, max_rounds,
                                                           rounds_completed)
      # skew-adjusted per-phase wait/adjusted grids + the honest end-to-end
      # relay_critical_path_cycles scalar (see bfs_timing.compute_skew_adjusted's
      # own docstring / GRAPH500_BENCHMARK.md section 10) -- folded straight
      # into phase_cycles since save_pe_phase_cycles just splats **phase_cycles
      # into the npz regardless of which keys are in it.
      skew = compute_skew_adjusted(phase_cycles, raw_slots, height, width)
      print(f"  relay_critical_path_cycles (real relay span, skew excluded): "
            f"{skew['relay_critical_path_cycles'].tolist()}")
      phase_cycles.update(skew)
      # also keep the raw absolute per-slot timestamps (prefixed to avoid
      # colliding with phase/skew names) -- otherwise a future change to
      # compute_skew_adjusted's own methodology can only be re-verified by
      # re-running the device, since decode_pe_phase_cycles() normally
      # discards these after computing phase diffs.
      phase_cycles.update({f"raw_{name}": grid for name, grid in raw_slots.items()})

      matrix_stem = os.path.splitext(os.path.basename(infile_mtx))[0]
      run_id = f"{matrix_stem}_{np_cols}x{np_rows}_src{source}"
      # lives inside plots/heatmap/<run_id>/ -- the same per-run folder
      # plot_pe_heatmap.py renders its PNGs into (it derives that folder
      # from wherever this .npz actually is, see its default_run_dir()),
      # so the raw data and its plots stay together as one self-contained
      # bundle rather than scattered across two top-level directories.
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
          # host-side partition structure, computed by preprocess_bool.py
          # before any device interaction -- for checking by eye whether a
          # phase's per-PE imbalance (e.g. local_compute) actually tracks
          # the matrix's own sparsity distribution across PEs.
          "local_nnz": local_nnz[:, :, 0].astype(np.int64),
          "local_nnz_cols": local_nnz_cols[:, :, 0].astype(np.int64),
          "local_nnz_rows": local_nnz_rows[:, :, 0].astype(np.int64),
      })
      print(f"saved per-PE timing grid to {pe_timing_out}")

    search_time_cycles = int(h2d_seed_cycles.max()) + device_time_cycles + int(d2h_cycles.max())
    row["search_time_cycles"] = search_time_cycles
    print(f"[[ search_time_cycles (h2d_seed + device rounds + d2h parent readback, "
          f"GRAPH500_BENCHMARK.md section 3): {search_time_cycles} ]]")

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
          "appending would silently misalign columns. Delete/rename the old CSV (it's a "
          "regenerable diagnostic log, not source data) or pass a different --csv path.")
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
