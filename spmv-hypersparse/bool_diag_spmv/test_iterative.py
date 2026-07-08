#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args
""" verify bool_diag_spmv's on-device iterative SpMV (f_spmv_iter) against
  the same number of sequential single-shot f_spmv launches, host-driven.

  bool_pe.csl's f_spmv_iter runs MAX_ITERS rounds of
  (broadcast -> local boolean multiply -> reduce-to-diagonal) on-device,
  copying each round's y back into x (at the diagonal PEs) before
  broadcasting again -- see the TODOs in bool_pe.csl's reduce_done() task:
  there is no masking yet (no visited-bitmap) and no real "y is globally
  empty" termination check, just a fixed MAX_ITERS loop.

  This script checks that the loop actually computes what it's supposed to:
  running f_spmv_iter once should give bit-identical results to calling the
  single-shot f_spmv MAX_ITERS times from the host, manually feeding each
  round's y back in as the next round's x -- exactly what the device does
  internally. Both entrypoints are exported from the SAME compiled kernel,
  so this only needs one compile + one SdkRuntime session.

  How to compile and run
     python test_iterative.py --arch=wse3 --num_pe_cols=4 --num_pe_rows=4
        --channels=1 --driver=<path to cslc> --infile_mtx=<path to mtx file>
"""

import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
from cmd_parser import parse_args
from preprocess_bool import preprocess
from run_bool import (csl_compile_core, dist_x_to_diag_hwl, extract_diag_result,
                       hwl_to_oned_colmajor, oned_to_hwl_colmajor)
from scipy.io import mmread

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)

# Must match bool_pe.csl's MAX_ITERS -- there's no host-settable param for
# this yet (see the TODO there: it's a fixed-count stub, not a real
# termination check), so the two sides are kept in sync by hand for now.
MAX_ITERS = 5

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iterative_results.jsonl")


def log_run(record):
  with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
  print(f"[test_iterative] appended run record to {LOG_FILE}")


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
  y_rows_init_buf = matrix_info["y_rows_init_buf"]
  local_nnz = matrix_info["local_nnz"]
  local_nnz_cols = matrix_info["local_nnz_cols"]
  local_nnz_rows = matrix_info["local_nnz_rows"]

  # blk = per-PE dense vector chunk size = ceil(n / P), same on both axes
  # since the matrix and grid are both square.
  blk = math.ceil(n / P)

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

  runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

  sym_x_buf = runner.get_id("x_buf")
  sym_y_buf = runner.get_id("y_buf")
  sym_mat_rows_buf = runner.get_id("mat_rows_buf")
  sym_mat_col_idx_buf = runner.get_id("mat_col_idx_buf")
  sym_mat_col_loc_buf = runner.get_id("mat_col_loc_buf")
  sym_mat_col_len_buf = runner.get_id("mat_col_len_buf")
  sym_y_rows_init_buf = runner.get_id("y_rows_init_buf")
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
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  def seed_x(x_hwl):
    x_buf_1d = hwl_to_oned_colmajor(height, width, blk, x_hwl, np.float32)
    runner.memcpy_h2d(sym_x_buf, x_buf_1d, 0, 0, width, height, blk,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)

  def read_y():
    y_1d = np.zeros(height * width * blk, np.float32)
    runner.memcpy_d2h(y_1d, sym_y_buf, 0, 0, width, height, blk,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    return oned_to_hwl_colmajor(height, width, blk, y_1d, np.float32)

  print(f"step 2: on-device iterative -- one f_spmv_iter launch, {MAX_ITERS} internal rounds")
  t0 = time.time()
  seed_x(x_hwl0)
  runner.launch("f_spmv_iter", nonblock=False)
  y_device_iter = extract_diag_result(n, blk, P, read_y())
  t_iter = time.time() - t0

  print(f"step 3: host-recursive baseline -- {MAX_ITERS}x sequential f_spmv launches, "
        "host copies y back into x between rounds")
  t0 = time.time()
  x_hwl = x_hwl0
  y_host_recursive = None
  per_round_popcount = []
  for _ in range(MAX_ITERS):
    seed_x(x_hwl)
    runner.launch("f_spmv", nonblock=False)
    y_host_recursive = extract_diag_result(n, blk, P, read_y())
    per_round_popcount.append(int(np.sum(y_host_recursive)))
    x_hwl = dist_x_to_diag_hwl(n, y_host_recursive, blk, P)
  t_recursive = time.time() - t0

  runner.stop()

  print(f"on-device f_spmv_iter:  {int(np.sum(y_device_iter))}/{n} reachable, "
        f"{t_iter*1e3:.2f} ms")
  print(f"host-recursive f_spmv:  {int(np.sum(y_host_recursive))}/{n} reachable, "
        f"{t_recursive*1e3:.2f} ms, per-round popcount {per_round_popcount}")

  n_mismatch = int(np.sum(y_device_iter != y_host_recursive))
  passed = n_mismatch == 0
  print(f"[[ mismatches between on-device iterative and host-recursive: {n_mismatch} / {n} ]]")
  print(f"[[ Result: {'PASS' if passed else 'FAIL'} ]]")
  if not passed:
    idx = np.where(y_device_iter != y_host_recursive)[0]
    shown = idx[:20].tolist()
    print(f"mismatched indices: {shown}{' ...' if len(idx) > 20 else ''}")

  log_run({
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "infile_mtx": infile_mtx,
      "n": n,
      "nnz": nnz,
      "pe_grid": f"{np_cols}x{np_rows}",
      "max_iters": MAX_ITERS,
      "compile_time_s": round(compile_time, 6),
      "device_iter_s": round(t_iter, 6),
      "host_recursive_s": round(t_recursive, 6),
      "per_round_popcount": per_round_popcount,
      "verify_pass": passed,
      "mismatches": n_mismatch,
  })


if __name__ == "__main__":
  main()
