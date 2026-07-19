#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args,wrong-import-position
""" Full Graph500-style BFS benchmark for bool_diag_spmv's on-device
  f_spmv_iter kernel: one compile, one matrix upload, then --num-searches
  (default 64, per the spec) single-source BFS searches from distinct
  random roots, each timed individually as its own Kernel 2 search (see
  GRAPH500_BENCHMARK.md section 1) -- and the harmonic mean GTEPS across
  all of them, the spec's own rule for combining per-search rates into one
  number.

  Kernel 1 / Kernel 2 split (GRAPH500_BENCHMARK.md section 1-2):
    - matrix structure (mat_rows_buf, mat_col_idx/loc/len_buf,
      local_nnz*) is uploaded to the device exactly ONCE,
      timed separately as "construction" -- never part of any search's own
      time, same as Graph500's own Kernel 1.
    - each search re-uploads only x_bitmap (the new root's one-hot seed) and
      re-launches f_spmv_iter. bool_pe.csl's start_spmv() already resets
      visited_bitmap/rounds_completed/parent_local_buf/ts_round itself on
      EVERY fresh f_spmv_iter() call (see its own comments on why
      overwriting, not OR-ing, is safe for repeated launches in the same
      session) -- no separate host-side reset step is needed or sent.
    - each search's own timed portion runs from seeding x_bitmap through
      reading parent_local_buf back into host memory -- the reference
      implementation's own run_bfs(root, pred) signature makes the
      predecessor array the sole official output (no separate "visited"
      readback exists in the spec at all), so that's the only d2h transfer
      that needs to be part of search_time_cycles. visited is derived
      host-side from parent alone (device_io.derive_visited_from_parent) --
      provably equivalent to reading visited_bitmap separately, see its own
      docstring -- so no other readback is needed or timed.

  run_bfs.py remains the single-search deep-dive tool (tree plot, verbose
  per-phase breakdown, --show-parent-mismatch); this script trades that
  per-search detail for running the full 64-search sweep the Graph500 spec
  actually asks for and reporting one aggregate number.

  How to compile and run
     cs_python run_graph500.py --arch=wse3 --num_pe_cols=8 --num_pe_rows=8
        --channels=1 --driver=<path to cslc> --infile_mtx=<path to mtx file>
     cs_python run_graph500.py ... --num-searches=16 --seed=1
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from preprocess_bool import preprocess
from scipy.io import mmread
from scipy.sparse.csgraph import breadth_first_order

# bfs_tree_plot.py lives in plots/ now -- see that folder's own scripts for
# the matching bootstrap back to this directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"))
from bfs_timing import (CLOCK_FREQ_HZ, NUM_TS_SLOTS, compute_m_and_gteps, decode_phase_row,
                         read_tic_toc_delta)
from bfs_tree_plot import invalid_parents
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

  parser.add_argument("--num-searches", type=int, default=64,
                       help="number of random-root BFS searches to run (Graph500 spec: 64)")
  parser.add_argument("--seed", type=int, default=0,
                       help="RNG seed for picking search roots (reproducible sampling)")
  parser.add_argument("--sources", default=None,
                       help="comma-separated explicit root list, overrides --num-searches/--seed "
                            "-- e.g. for reproducing one specific search from a prior run")
  parser.add_argument("--max-rounds", type=int, default=15,
                       help="on-device cap on rounds actually profiled per search (bool_pe.csl's "
                            "ts_buf) -- rounds beyond this still run correctly, just aren't "
                            "timestamped; bump this if a run reports truncation")
  parser.add_argument("--nocorrectness", action="store_true",
                       help="skip the per-search scipy breadth_first_order cross-check -- on by "
                            "default since it's cheap relative to the device time and this is "
                            "exactly the kind of first full pass where a real bug should still "
                            "be caught, not just assumed correct because run_bfs.py passed once")
  parser.add_argument("--csv", default=None,
                       help="per-search CSV path (default: results/graph500_searches.csv next "
                            "to this script)")
  parser.add_argument("--summary-csv", default=None,
                       help="one-row-per-benchmark-run summary CSV (default: "
                            "results/graph500_summary.csv next to this script)")
  return parser.parse_args()


def pick_sources(args, A_csc, n):
  if args.sources is not None:
    sources = [int(s) for s in args.sources.split(",")]
    for s in sources:
      assert 0 <= s < n, f"--sources entry {s} out of range [0, {n})"
    return sources

  # Graph500's own rule (https://graph500.org/?page_id=12 section 5): sample
  # roots with degree >= 1, NOT COUNTING SELF-LOOPS -- a source with only a
  # self-loop (or no edges at all) can never discover anything new, so
  # m/GTEPS for it would be a meaningless divide-by-(effectively)-zero.
  # row=dest/col=source (see generate_boolean_reference in
  # run_single_spmv.py), so raw out-degree is the per-column nnz count;
  # subtract 1 for any column that also has a diagonal (self-loop) entry.
  # gen_rmat.py's own output never has self-loops, but an arbitrary
  # --infile_mtx (e.g. data/rand600.mtx) can.
  outdeg = np.diff(A_csc.indptr) - (A_csc.diagonal() != 0).astype(np.int64)
  candidates = np.nonzero(outdeg > 0)[0]
  if len(candidates) < args.num_searches:
    print(f"[[ NOTE: only {len(candidates)} vertices have degree >= 1 (excluding self-loops) -- "
          f"requested --num-searches={args.num_searches}, using all {len(candidates)} available "
          "instead ]]")
  num = min(args.num_searches, len(candidates))
  rng = np.random.default_rng(args.seed)
  return rng.choice(candidates, size=num, replace=False).tolist()


def harmonic_mean(values):
  values = [v for v in values if v > 0 and np.isfinite(v)]
  if not values:
    return float("nan")
  return len(values) / sum(1.0 / v for v in values)


def main():
  """Main method to run the example code."""

  args = parse_args()
  need_correctness = not args.nocorrectness

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
  print(f"infile_mtx = {infile_mtx}, num_searches = {args.num_searches}, max_rounds = {max_rounds}")

  A_coo = mmread(infile_mtx)
  A_csr = A_coo.tocsr(copy=True)
  A_csr = A_csr.sorted_indices()
  assert A_csr.has_sorted_indices == 1, "Error: A is not sorted"

  [nrows, ncols] = A_csr.shape
  assert nrows == ncols, "boolean diagonal-reduce SpMV requires a square matrix"
  n = nrows
  nnz = A_csr.nnz
  print(f"Load matrix A, {nrows}-by-{ncols} with {nnz} nonzeros (structural, boolean)")

  # Graph500's own m formula (the undirected dedup rule, see
  # GRAPH500_BENCHMARK.md section 4) only makes sense for a symmetrized
  # graph -- true for gen_rmat.py's output but not guaranteed for an
  # arbitrary --infile_mtx.
  is_symmetric = (A_csr != A_csr.T).nnz == 0
  if not is_symmetric:
    print("[[ NOTE: A_csr is not symmetric -- using the directed edges-traversed formula "
          "instead of Graph500's own undirected dedup rule for every search's m; not directly "
          "comparable to a Graph500-spec TEPS number. See GRAPH500_BENCHMARK.md section 4. ]]")

  A_csc = A_csr.tocsc(copy=True)
  A_csc = A_csc.sorted_indices()
  assert A_csc.has_sorted_indices == 1, "Error: A is not sorted"
  A_coo_static = A_csr.tocoo()  # reused by compute_m_and_gteps every search

  sources = pick_sources(args, A_csc, n)
  print(f"selected {len(sources)} search roots (seed={args.seed}): {sources}")

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

  blk = -(-n // P)  # ceil(n / P)
  bitmap_words = (blk + 31) // 32

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
  sym_ts_buf = runner.get_id("ts_buf")
  sym_tsc_start_buffer = runner.get_id("tsc_start_buffer")
  sym_tsc_end_buffer = runner.get_id("tsc_end_buffer")

  runner.load()
  runner.run()

  print("enabling tsc...")
  runner.launch("f_enable_tsc", nonblock=False)

  # --- Kernel 1 (construction): matrix structure upload, ONCE ---
  print("timing h2d: matrix structure upload (Kernel 1, construction -- done once)...")
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
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)

  runner.launch("f_toc", nonblock=False)  # blocks -> every matrix-structure h2d above is done
  h2d_matrix_cycles = read_tic_toc_delta(runner, sym_tsc_start_buffer, sym_tsc_end_buffer,
                                          height, width)
  construction_time_seconds = int(h2d_matrix_cycles.max()) / CLOCK_FREQ_HZ
  print(f"construction (h2d_matrix): max={int(h2d_matrix_cycles.max())} cycles "
        f"({construction_time_seconds * 1e6:.2f} us) -- NOT part of any search's own time")

  # --- Kernel 2: one BFS search per root, each timed individually ---
  ts_len = max_rounds * NUM_TS_SLOTS * 3
  search_rows = []
  n_correctness_fail = 0
  for i, source in enumerate(sources):
    # single-source seed: only the ONE diagonal PE owning `source` ever
    # needs a real host write -- see single_source_seed_pe()'s own
    # docstring for why every other PE's x_bitmap is already provably zero.
    px, py, local_x = single_source_seed_pe(source, blk, P)

    runner.launch("f_tic", nonblock=True)
    runner.memcpy_h2d(sym_x_bitmap, local_x, px, py, 1, 1, bitmap_words,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    runner.launch("f_toc", nonblock=False)  # blocks -> seed x h2d above is done
    h2d_seed_cycles = read_tic_toc_delta(runner, sym_tsc_start_buffer, sym_tsc_end_buffer,
                                          height, width)

    # the only per-search "reset": bool_pe.csl's start_spmv() reinitializes
    # visited_bitmap/rounds_completed/parent_local_buf/ts_round itself on every
    # fresh f_spmv_iter() call -- see the module docstring above.
    runner.launch("f_spmv_iter", nonblock=False)

    # Graph500's own output is exactly the predecessor/parent array (see
    # GRAPH500_BENCHMARK.md section 1 -- the reference implementation's
    # run_bfs(root, pred) signature) -- derive_visited_from_parent() below
    # recovers visited from parent_local_buf alone, so only that one
    # transfer needs to be timed as the search's "output written to memory"
    # cost, and it's folded into search_time_cycles below.
    runner.launch("f_tic", nonblock=True)
    parent_local_buf_1d = np.zeros(height * width * blk, np.uint32)
    runner.memcpy_d2h(parent_local_buf_1d, sym_parent_local_buf, 0, 0, width, height, blk,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    runner.launch("f_toc", nonblock=False)  # blocks -> the d2h read above is done
    d2h_cycles = read_tic_toc_delta(runner, sym_tsc_start_buffer, sym_tsc_end_buffer,
                                     height, width)

    rounds_buf = np.zeros(height * width, np.uint32)
    runner.memcpy_d2h(rounds_buf, sym_rounds_completed, 0, 0, width, height, 1,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    rounds_completed = int(np.reshape(rounds_buf, (height, width, 1), order="F")[(0, 0, 0)])

    ts_buf_1d = np.zeros(height * width * ts_len, np.uint32)
    runner.memcpy_d2h(ts_buf_1d, sym_ts_buf, 0, 0, width, height, ts_len,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    ts_hwl_u32 = np.reshape(ts_buf_1d, (height, width, ts_len), order="F")

    device_parent = extract_parent_result(
        n, blk, P, np.reshape(parent_local_buf_1d, (height, width, blk), order="F"))
    device_parent[source] = source
    device_visited = derive_visited_from_parent(n, device_parent, source)

    scipy_ok = None
    if need_correctness:
      A_fwd = A_csr.transpose().tocsr()
      scipy_order, _ = breadth_first_order(A_fwd, source, directed=True,
                                            return_predecessors=True)
      scipy_visited = np.zeros(n, dtype=bool)
      scipy_visited[scipy_order] = True
      bad_device = invalid_parents(device_parent, device_visited, A_csr, source)
      n_mismatch = int(np.sum(device_visited != scipy_visited))
      scipy_ok = n_mismatch == 0 and not bad_device
      if not scipy_ok:
        n_correctness_fail += 1
        print(f"[[ search {i} (source={source}): CORRECTNESS FAILED -- "
              f"visited mismatches={n_mismatch}, invalid parents={len(bad_device)} ]]")

    row_cols, device_time_cycles, profiled_rounds = decode_phase_row(
        ts_hwl_u32, height, width, max_rounds, rounds_completed, verbose=False)
    search_time_cycles = int(h2d_seed_cycles.max()) + device_time_cycles + int(d2h_cycles.max())
    m, m_convention, search_time_seconds, gteps = compute_m_and_gteps(
        A_coo_static, device_visited, is_symmetric, search_time_cycles)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "infile_mtx": os.path.basename(infile_mtx),
        "n": n,
        "nnz": nnz,
        "pe_grid": f"{np_cols}x{np_rows}",
        "channels": channels,
        "matrix_symmetric": is_symmetric,
        "search_idx": i,
        "source": source,
        "rounds_completed": rounds_completed,
        "profiled_rounds": profiled_rounds,
        "max_rounds": max_rounds,
        "correctness_ok": scipy_ok,
        "h2d_seed_max_cycles": int(h2d_seed_cycles.max()),
        "d2h_max_cycles": int(d2h_cycles.max()),
        "search_time_cycles": search_time_cycles,
        "visited_count": int(np.sum(device_visited)),
        "m_edges_traversed": m,
        "m_convention": m_convention,
        "clock_freq_hz": CLOCK_FREQ_HZ,
        "search_time_seconds": search_time_seconds,
        "gteps": gteps,
    }
    row.update(row_cols)
    search_rows.append(row)

    print(f"search {i+1}/{len(sources)}: source={source} visited={row['visited_count']}/{n} "
          f"rounds={rounds_completed} m={m} time={search_time_seconds*1e6:.2f}us "
          f"gteps={gteps:.6f}"
          + ("" if scipy_ok is None else (" OK" if scipy_ok else " CORRECTNESS-FAILED")))

  runner.stop()

  # --- Aggregate, per GRAPH500_BENCHMARK.md section 1: harmonic mean of the
  # per-search rates, plus min/max/median for context (the spec reports
  # quartiles/min/max alongside the harmonic mean too). ---
  gteps_values = [r["gteps"] for r in search_rows]
  finite_gteps = [g for g in gteps_values if np.isfinite(g) and g > 0]
  hmean_gteps = harmonic_mean(gteps_values)
  summary = {
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "infile_mtx": os.path.basename(infile_mtx),
      "n": n,
      "nnz": nnz,
      "pe_grid": f"{np_cols}x{np_rows}",
      "channels": channels,
      "matrix_symmetric": is_symmetric,
      "num_searches": len(search_rows),
      "num_correctness_fail": n_correctness_fail,
      "construction_time_seconds": construction_time_seconds,
      "harmonic_mean_gteps": hmean_gteps,
      "min_gteps": min(finite_gteps) if finite_gteps else float("nan"),
      "median_gteps": float(np.median(finite_gteps)) if finite_gteps else float("nan"),
      "max_gteps": max(finite_gteps) if finite_gteps else float("nan"),
  }

  print(f"\n[[ {len(search_rows)} searches complete, {n_correctness_fail} correctness FAILURES ]]")
  print(f"[[ construction (Kernel 1, h2d_matrix, not in any search's time): "
        f"{construction_time_seconds*1e6:.2f} us ]]")
  print(f"[[ GTEPS across searches: harmonic_mean={hmean_gteps:.6f}, "
        f"min={summary['min_gteps']:.6f}, median={summary['median_gteps']:.6f}, "
        f"max={summary['max_gteps']:.6f} ]]"
        + ("" if is_symmetric else "  -- directed graph: not Graph500-spec-comparable, "
                                    "see m_convention"))

  csv_path = args.csv or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                       "graph500_searches.csv")
  os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
  write_header = not os.path.exists(csv_path)
  if not write_header:
    with open(csv_path, newline="", encoding="utf-8") as f:
      existing_header = next(csv.reader(f), [])
    assert existing_header == list(search_rows[0].keys()), (
        f"{csv_path}'s header doesn't match this run's columns (schema changed?) -- "
        "appending would silently misalign columns. Delete/rename the old CSV (it's a "
        "regenerable diagnostic log, not source data) or pass a different --csv path.")
  with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(search_rows[0].keys()))
    if write_header:
      writer.writeheader()
    writer.writerows(search_rows)
  print(f"appended {len(search_rows)} per-search rows to {csv_path}")

  summary_csv_path = args.summary_csv or os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "results", "graph500_summary.csv")
  os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
  write_header = not os.path.exists(summary_csv_path)
  if not write_header:
    with open(summary_csv_path, newline="", encoding="utf-8") as f:
      existing_header = next(csv.reader(f), [])
    assert existing_header == list(summary.keys()), (
        f"{summary_csv_path}'s header doesn't match this run's columns (schema changed?) -- "
        "appending would silently misalign columns. Delete/rename the old CSV or pass a "
        "different --summary-csv path.")
  with open(summary_csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
    if write_header:
      writer.writeheader()
    writer.writerow(summary)
  print(f"appended summary row to {summary_csv_path}")


if __name__ == "__main__":
  main()
