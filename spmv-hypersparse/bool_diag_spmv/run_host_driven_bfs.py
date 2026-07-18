#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args
""" verify bool_diag_spmv's on-device iterative SpMV (f_spmv_iter) against
  the same, self-terminating sequence of single-shot f_spmv launches,
  host-driven.

  bool_pe.csl's f_spmv_iter runs rounds of (broadcast -> local boolean
  multiply -> reduce-to-diagonal) on-device, for as many rounds as real BFS
  convergence takes -- no fixed round count. At the diagonal PEs,
  reduce_done() masks each round's raw result against a cumulative
  visited_bitmap before feeding it back as the next round's x -- only
  genuinely new discoveries propagate, same as this file's own host-driven
  baseline below and bfs_spmv/run_bfs.py's host-side
  `new_mask = candidate & ~visited`. After masking, a 4-phase relay
  (column-reduce, row-reduce, row-broadcast, column-broadcast, all rooted at
  MID = pcols/2 -- see reduce_done()/term_col_done()/term_row_done()/
  term_row_bcast_done()/term_col_bcast_done() in bool_pe.csl) checks whether
  ANY row found something new this round; if not, the whole grid agrees to
  stop.

  Parent tracking (which vertex discovered a given node) can't go through
  the row-reduce -- it's a combine (OR), not a selection -- so it's tracked
  per-PE BEFORE the reduce instead (see compute() in bool_pe.csl): every PE knows
  exactly which of its own local columns touched which local row this
  round, and -- gated on that row not being visited yet, which is what
  makes this a genuine one-hop-closer BFS parent rather than merely a
  valid-but-arbitrary predecessor, see bool_pe.csl's module docstring --
  keeps the lowest-global-index one it's seen in parent_local_buf. Because
  different PEs in the same row cover different column ranges, the row's
  true parent is the min across all P PEs in that row -- <collectives_2d>
  has no min-reduce, so that last step is done here, host-side, in
  extract_parent_result() below, after memcpy_d2h reads back the full (not
  diagonal-only) parent_local_buf rectangle.

  This script checks four things: (1) visited_bitmap (the cumulative
  discovered set) is bit-identical between running f_spmv_iter once and
  calling single-shot f_spmv from the host in a loop that applies the
  identical visited-mask and stops the same way; (2) the terminating
  round's RAW (unmasked) y_bitmap_reduced (unpacked to dense) is also
  bit-identical -- NOT the masked x_bitmap, which the loop's own stop condition
  forces to all-zero on both sides regardless of correctness, so comparing
  it would be vacuous; y_bitmap_reduced is genuinely data-dependent (it can
  be nonzero, full of entries that happen to already be in visited_bitmap) and
  is where a bug in the terminating round's own SpMV computation would
  actually show up; (3) rounds_completed
  (a host-visible counter, purely for this test) shows the device stopped at
  exactly the same round the host independently computed, not some other
  round; (4) the assembled parent vector matches a host-side reference that
  independently recomputes, for every round, the lowest-index active
  frontier member with a real edge to each not-yet-visited row (mirroring
  the device's own rule exactly -- see update_parent_reference()'s
  docstring). Masked x_bitmap being genuinely all-zero at stop time
  is checked too, as a sanity invariant rather than a host comparison. (The
  device side has no round cap at all -- runner.launch(..., nonblock=False)
  blocking on f_spmv_iter and returning is itself proof the on-device relay
  terminated at all; the host loop below keeps a generous n-round safety
  net purely so a genuine bug can't hang this *script*, not because the
  device needs one.) Both entrypoints are exported from the SAME compiled
  kernel, so this only needs one compile + one SdkRuntime session.

  How to compile and run
     python run_host_driven_bfs.py --arch=wse3 --num_pe_cols=4 --num_pe_rows=4
        --channels=1 --driver=<path to cslc> --infile_mtx=<path to mtx file>
"""

import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
from cmd_parser import parse_args
from device_io import (assert_n_supported, csl_compile_core, dist_x_to_diag_hwl,
                        extract_diag_result, extract_parent_result, hwl_to_oned_colmajor,
                        pack_dense_to_bitmap, unpack_bitmap_to_dense)
from preprocess_bool import preprocess
from scipy.io import mmread

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                         "iterative_results.jsonl")


def log_run(record):
  os.makedirs(os.path.dirname(os.path.abspath(LOG_FILE)), exist_ok=True)
  with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
  print(f"[run_host_driven_bfs] appended run record to {LOG_FILE}")


def update_parent_reference(host_parent, A_csr, frontier_bool, visited):
  """Host-side reference mirroring bool_pe.csl's compute() exactly: for
  every row v NOT YET VISITED (as of the start of this round -- see below),
  if any node currently in frontier_bool has a real edge to v (A_csr[v, :]
  is row v's predecessor columns, per bool_diag_spmv's row=dest/col=source
  convention -- see generate_boolean_reference in run_single_spmv.py), take
  the lowest-index one as v's parent.

  The `visited` gate is what makes this a genuine, textbook one-hop-closer
  BFS parent (matching bfs_spmv/run_bfs.py's find_parents(), which only
  considers the frontier immediately preceding a node's first discovery):
  no row can ever receive a hit before its own true discovery round (a hit
  from ANY frontier member immediately makes that row visited by the end of
  the same round -- see bool_pe.csl's f_spmv OR-reduce), so gating on
  `visited` restricts this to exactly a row's discovery round. Earlier
  versions of this function had no such gate (matching an earlier, buggier
  version of compute() that could let an unrelated, much-later round's
  frontier member overwrite an already-visited row's parent whenever its
  index happened to be lower) -- see visited_bitmap's role in bool_pe.csl's
  module docstring for the on-device side of this fix.
  """
  frontier_idx = np.nonzero(frontier_bool)[0]
  if frontier_idx.size == 0:
    return
  frontier_set = set(frontier_idx.tolist())
  n = len(host_parent)
  for v in range(n):
    if visited[v]:
      continue
    start, end = A_csr.indptr[v], A_csr.indptr[v + 1]
    best = -1
    for u in A_csr.indices[start:end]:
      if u in frontier_set and (best == -1 or u < best):
        best = u
    if best != -1:
      host_parent[v] = best


def main():
  """Main method to run the example code."""

  args = parse_args()

  cslc = "cslc"
  if args.driver is not None:
    cslc = args.driver

  width_west_buf = args.width_west_buf
  width_east_buf = args.width_east_buf
  channels = args.channels
  assert channels <= 16, "only support up to 16 I/O channels"
  assert channels >= 1, "number of I/O channels must be at least 1"

  dirname = args.latestlink

  np_cols = args.num_pe_cols
  np_rows = args.num_pe_rows
  assert np_cols == np_rows, "diagonal-reduce design requires a square PE grid"
  P = np_cols

  width = np_cols
  height = np_rows

  infile_mtx = args.infile_mtx
  print(f"infile_mtx = {infile_mtx}")

  A_coo = mmread(infile_mtx)
  A_csr = A_coo.tocsr(copy=True)
  A_csr = A_csr.sorted_indices()
  assert A_csr.has_sorted_indices == 1, "Error: A is not sorted"

  [nrows, ncols] = A_csr.shape
  assert nrows == ncols, "boolean diagonal-reduce SpMV requires a square matrix"
  n = nrows
  assert_n_supported(n)
  nnz = A_csr.nnz

  print(f"Load matrix A, {nrows}-by-{ncols} with {nnz} nonzeros (structural, boolean)")

  csrRowPtr = A_csr.indptr
  csrColInd = A_csr.indices

  A_csc = A_csr.tocsc(copy=True)
  A_csc = A_csc.sorted_indices()
  assert A_csc.has_sorted_indices == 1, "Error: A is not sorted"

  cscColPtr = A_csc.indptr
  cscRowInd = A_csc.indices

  start = time.time()
  matrix_info = preprocess(
      nrows,
      ncols,
      nnz,
      np_cols,
      np_rows,
      csrRowPtr,
      csrColInd,
      cscColPtr,
      cscRowInd,
  )
  end = time.time()
  print(f"prepare the structure for spmv kernel: {end-start}s", flush=True)

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

  # blk = per-PE dense vector chunk size = ceil(n / P), same on both axes
  # since the matrix and grid are both square.
  blk = math.ceil(n / P)
  # bitmap_words: matches bool_pe.csl's BITMAP_WORDS = ceil(blk/32) exactly --
  # y_bitmap_reduced is packed this way, see device_io.unpack_bitmap_to_dense.
  bitmap_words = (blk + 31) // 32

  np.random.seed(0)
  x_bool0 = np.random.rand(n) < 0.5
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

  print(f"fabric_width = {fabric_width}, fabric_height = {fabric_height}")
  print("store ELFs and log files in the folder ", dirname)

  # NOTE: absolute, anchored to this file's own location -- see the matching
  # comment in original_spmv/run.py for why (container bind-mount only
  # covers the invocation cwd).
  code_csl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "layout_bool.csl")

  start = time.time()
  csl_compile_core(
      cslc,
      code_csl,
      dirname,
      fabric_width,
      fabric_height,
      core_fabric_offset_x,
      core_fabric_offset_y,
      args.run_only,
      args.arch,
      np_cols,
      np_rows,
      blk,
      max_local_nnz,
      max_local_nnz_cols,
      max_local_nnz_rows,
      channels,
      width_west_buf,
      width_east_buf,
  )
  end = time.time()
  compile_time = end - start
  print(f"Compilation done in {compile_time}s", flush=True)

  if args.compile_only:
    print("COMPILE ONLY: EXIT")
    return

  runner = SdkRuntime(dirname, cmaddr=args.cmaddr, suppress_simfab_trace=True)

  sym_x_bitmap = runner.get_id("x_bitmap")
  sym_y_bitmap_reduced = runner.get_id("y_bitmap_reduced")
  sym_visited_bitmap = runner.get_id("visited_bitmap")
  sym_rounds_completed = runner.get_id("rounds_completed")
  sym_parent_local_buf = runner.get_id("parent_local_buf")
  sym_mat_rows_buf = runner.get_id("mat_rows_buf")
  sym_mat_col_idx_buf = runner.get_id("mat_col_idx_buf")
  sym_mat_col_loc_buf = runner.get_id("mat_col_loc_buf")
  sym_mat_col_len_buf = runner.get_id("mat_col_len_buf")
  sym_local_nnz = runner.get_id("local_nnz")
  sym_local_nnz_cols = runner.get_id("local_nnz_cols")
  sym_local_nnz_rows = runner.get_id("local_nnz_rows")

  start = time.time()
  runner.load()
  end = time.time()
  print(f"*** Load done in {end-start}s")

  runner.run()

  print("step 1: copy the structure of A to the device (once, shared by both runs below)")

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
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  def seed_x(x_hwl):
    x_bitmap_hwl = pack_dense_to_bitmap(height, width, blk, x_hwl)
    x_bitmap_1d = hwl_to_oned_colmajor(height, width, bitmap_words, x_bitmap_hwl, np.uint32)
    runner.memcpy_h2d(sym_x_bitmap, x_bitmap_1d, 0, 0, width, height, bitmap_words,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)

  def read_bitmap(sym):
    # x_bitmap/visited_bitmap/y_bitmap_reduced are all packed the same way
    # (bitmap_words u32 words per PE) -- read at native size, then unpack to
    # the dense float shape callers already expect (see extract_diag_result).
    buf_1d = np.zeros(height * width * bitmap_words, np.uint32)
    runner.memcpy_d2h(buf_1d, sym, 0, 0, width, height, bitmap_words,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    bitmap_hwl = np.reshape(buf_1d, (height, width, bitmap_words), order="F")
    return unpack_bitmap_to_dense(height, width, blk, bitmap_hwl)

  def read_rounds_completed():
    # every PE increments its own copy in lockstep (the 4-phase relay makes
    # them all agree each round before any of them decides to continue), so
    # any single PE's value is the global answer -- just read (0, 0). u16
    # readback mirrors the h2d convention used for mat_rows_buf/local_nnz
    # above: MEMCPY_16BIT wire format, uint32-typed host buffer.
    buf_1d = np.zeros(height * width, np.uint32)
    runner.memcpy_d2h(buf_1d, sym_rounds_completed, 0, 0, width, height, 1,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    return int(np.reshape(buf_1d, (height, width, 1), order="F")[(0, 0, 0)])

  def read_parent_local_buf():
    # unlike x_bitmap/y_bitmap_reduced/visited_bitmap, parent_local_buf is meaningful at
    # EVERY PE (not just the diagonal) -- read back the full rectangle,
    # same u16-over-u32-wire convention as read_rounds_completed above.
    buf_1d = np.zeros(height * width * blk, np.uint32)
    runner.memcpy_d2h(buf_1d, sym_parent_local_buf, 0, 0, width, height, blk,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    return np.reshape(buf_1d, (height, width, blk), order="F")

  print("step 2: on-device iterative -- one f_spmv_iter launch, runs until the "
        "on-device termination relay stops it (see module docstring)")
  t0 = time.time()
  seed_x(x_hwl0)
  runner.launch("f_spmv_iter", nonblock=False)
  # x_bitmap ends the call holding the LAST round's masked new-discoveries (see
  # reduce_done() in bool_pe.csl -- it runs the mask/visited update on every
  # round, including the last, so x_bitmap is never stale); visited_bitmap holds
  # everything discovered across however many rounds actually ran. Getting
  # here at all (runner.launch above returned) already proves the device
  # terminated -- nonblock=False blocks until the device unblocks the cmd
  # stream, which only happens once the relay's nz_total goes to zero.
  #
  # NOTE: device_new_last (masked x_bitmap) is NOT compared against the host
  # below -- now that there's no round cap, the loop *only* stops once a
  # round's masked output is all-zero, so that vector is trivially zero on
  # BOTH sides by construction of the stop condition itself, regardless of
  # whether anything upstream is even correct. It's still useful as an
  # invariant check (the relay's stop decision better actually agree with
  # x_bitmap's content), just not as a host comparison. The real comparison
  # uses y_bitmap_reduced -- the RAW, unmasked SpMV result for the
  # terminating round, which is genuinely data-dependent (it can easily be
  # nonzero, full of entries that all happen to already be in visited_bitmap).
  device_new_last = extract_diag_result(n, blk, P, read_bitmap(sym_x_bitmap))
  device_last_raw = extract_diag_result(n, blk, P, read_bitmap(sym_y_bitmap_reduced))
  device_visited = extract_diag_result(n, blk, P, read_bitmap(sym_visited_bitmap))
  device_rounds_run = read_rounds_completed()
  device_parent = extract_parent_result(n, blk, P, read_parent_local_buf())
  t_iter = time.time() - t0

  print("step 3: host-driven baseline -- sequential f_spmv launches, host applies "
        "the same visited-mask and stops the same way (capped at n rounds purely "
        "as a safety net for this script, not a device limit -- see module docstring)")
  t0 = time.time()
  x_hwl = x_hwl0
  frontier_bool = x_bool0  # this round's active input, for update_parent_reference
  visited = x_bool0.copy()  # f_spmv_iter seeds visited_bitmap from the initial x_bitmap too
  host_parent = np.full(n, -1, dtype=np.int64)
  host_last_raw = None  # the terminating round's raw y_bitmap_reduced -- the real comparison target
  host_new_last = None
  per_round_popcount = []
  host_rounds_run = 0
  for _ in range(n):
    update_parent_reference(host_parent, A_csr, frontier_bool, visited)
    seed_x(x_hwl)
    runner.launch("f_spmv", nonblock=False)
    host_last_raw = extract_diag_result(n, blk, P, read_bitmap(sym_y_bitmap_reduced))
    host_new_last = host_last_raw & ~visited
    visited |= host_new_last
    host_rounds_run += 1
    per_round_popcount.append(int(np.sum(host_new_last)))
    if not host_new_last.any():
      break  # matches the device's real termination check -- stop as soon
             # as a round finds nothing new
    frontier_bool = host_new_last
    x_hwl = dist_x_to_diag_hwl(n, host_new_last.astype(np.float32), blk, P)
  else:
    raise RuntimeError(f"host-driven baseline did not converge within {n} rounds -- "
                        "this should be impossible (bounded by node count); likely a bug")
  t_recursive = time.time() - t0

  runner.stop()

  print(f"on-device f_spmv_iter:  visited {int(np.sum(device_visited))}/{n}, "
        f"last-round raw {int(np.sum(device_last_raw))}, "
        f"{device_rounds_run} rounds run, {t_iter*1e3:.2f} ms")
  print(f"host-driven f_spmv:     visited {int(np.sum(visited))}/{n}, "
        f"last-round raw {int(np.sum(host_last_raw))}, {host_rounds_run} rounds run, "
        f"{t_recursive*1e3:.2f} ms, per-round popcount {per_round_popcount}")

  n_mismatch_visited = int(np.sum(device_visited != visited))
  n_mismatch_last_raw = int(np.sum(device_last_raw != host_last_raw))
  n_mismatch_parent = int(np.sum(device_parent != host_parent))
  rounds_match = device_rounds_run == host_rounds_run
  invariant_ok = not device_new_last.any()  # masked x_bitmap must be all-zero when stopped
  passed = ((n_mismatch_visited == 0) and (n_mismatch_last_raw == 0)
            and (n_mismatch_parent == 0) and rounds_match and invariant_ok)
  n_with_parent = int(np.sum(device_parent >= 0))
  print(f"[[ visited mismatches: {n_mismatch_visited} / {n} ]]")
  print(f"[[ last-round RAW (unmasked y_bitmap_reduced) mismatches: {n_mismatch_last_raw} / {n} ]]")
  print(f"[[ parent mismatches: {n_mismatch_parent} / {n} ({n_with_parent} assigned a parent) ]]")
  print(f"[[ rounds completed: device={device_rounds_run}, host={host_rounds_run} "
        f"({'match' if rounds_match else 'MISMATCH'}) ]]")
  print(f"[[ stop invariant (masked x_bitmap all-zero): "
        f"{'OK' if invariant_ok else 'VIOLATED -- ' + str(int(np.sum(device_new_last)))} ]]")
  print(f"[[ Result: {'PASS' if passed else 'FAIL'} ]]")
  if not passed:
    idx = np.where(device_visited != visited)[0]
    print(f"visited mismatch indices: {idx[:20].tolist()}{' ...' if len(idx) > 20 else ''}")
    idx = np.where(device_last_raw != host_last_raw)[0]
    print(f"last-round raw mismatch indices: {idx[:20].tolist()}{' ...' if len(idx) > 20 else ''}")
    idx = np.where(device_parent != host_parent)[0]
    print(f"parent mismatch indices: {idx[:20].tolist()}{' ...' if len(idx) > 20 else ''}")
    if idx.size:
      shown = idx[:10]
      print(f"  device_parent{shown.tolist()} = {device_parent[shown].tolist()}")
      print(f"  host_parent{shown.tolist()}   = {host_parent[shown].tolist()}")

  log_run({
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "infile_mtx": infile_mtx,
      "n": n,
      "nnz": nnz,
      "pe_grid": f"{np_cols}x{np_rows}",
      "compile_time_s": round(compile_time, 6),
      "device_iter_s": round(t_iter, 6),
      "host_recursive_s": round(t_recursive, 6),
      "per_round_popcount": per_round_popcount,
      "device_rounds_run": device_rounds_run,
      "host_rounds_run": host_rounds_run,
      "verify_pass": passed,
      "mismatches_visited": n_mismatch_visited,
      "mismatches_last_round_raw": n_mismatch_last_raw,
      "mismatches_parent": n_mismatch_parent,
      "stop_invariant_ok": invariant_ok,
  })


if __name__ == "__main__":
  main()
