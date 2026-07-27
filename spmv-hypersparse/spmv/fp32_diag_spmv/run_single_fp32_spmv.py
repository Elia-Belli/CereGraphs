#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args
""" test fp32 (real-valued) diagonal-broadcast sparse matrix-vector
  multiplication, one round.

  Forked from bool_diag_spmv/run_single_spmv.py for the fp32 variant (see
  src/fp32_pe.csl / src/layout_fp32.csl):
    - A is a square real-valued matrix,
    - y = A @ x (genuine fp32 sum, not a boolean OR-via-add surrogate),
    - the PE grid must be square so every row/column has a diagonal PE.

  This is Workstream A's correctness gate for the device-only-vs-host-driven
  comparison (see the plan): before any timing numbers from that comparison
  can be trusted, this script must confirm a single f_spmv round produces
  the same result as a dense NumPy/SciPy reference.

  How to compile and run
     python run_single_fp32_spmv.py --arch=wse2 --num_pe_cols=4 --num_pe_rows=4 --channels=1
        --driver=<path to cslc> --compile-only --infile_mtx=<path to mtx file>
     python run_single_fp32_spmv.py --arch=wse2 --num_pe_cols=4 --num_pe_rows=4 --channels=1
        --run-only --infile_mtx=<path to mtx file>
"""

import math
import os
import time

import numpy as np
from cmd_parser import parse_args
from device_io import (csl_compile_core, dist_x_to_diag_hwl, extract_diag_result,
                        hwl_to_oned_colmajor)
from graph_loader import load_graph
from preprocess_fp32 import preprocess

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)


def verify_result(ref, res, rtol=1e-3, atol=1e-4):
  print("Comparing fp32 result with dense reference...")
  close = np.isclose(ref, res, rtol=rtol, atol=atol)
  n_mismatch = np.sum(~close)
  print(f"[[ Mismatches: {n_mismatch} / {len(ref)} (rtol={rtol}, atol={atol}) ]]")
  result = "PASS" if n_mismatch == 0 else "FAIL"
  print(f"[[ Result: {result} ]]")
  if n_mismatch != 0:
    idx = np.where(~close)[0][:20]
    print(f"first mismatched indices: {idx}")
    print(f"  ref: {ref[idx]}")
    print(f"  res: {res[idx]}")
  return n_mismatch == 0


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

  A_coo = load_graph(infile_mtx)
  A_csr = A_coo.tocsr(copy=True)
  A_csr = A_csr.sorted_indices()
  assert A_csr.has_sorted_indices == 1, "Error: A is not sorted"

  [nrows, ncols] = A_csr.shape
  assert nrows == ncols, "diagonal-reduce SpMV requires a square matrix"
  n = nrows
  nnz = A_csr.nnz

  # Assign real (non-structural) edge weights unless --is_weight_one is
  # given -- same default sdk-hypersparse-spmv/run.py uses (np.random.seed(123)),
  # so the fp32 comparison in Workstream B exercises the same value
  # distribution as the SDK baseline it's measured against.
  A_csr = A_csr.astype(np.float32)
  if not args.is_weight_one:
    np.random.seed(123)
    A_csr.data[0:nnz] = np.random.rand(nnz).astype(np.float32)
  else:
    A_csr.data[0:nnz] = np.float32(1.0)

  print(f"Load matrix A, {nrows}-by-{ncols} with {nnz} nonzeros (fp32)")

  csrRowPtr = A_csr.indptr
  csrColInd = A_csr.indices

  A_csc = A_csr.tocsc(copy=True)
  A_csc = A_csc.sorted_indices()
  assert A_csc.has_sorted_indices == 1, "Error: A is not sorted"

  cscColPtr = A_csc.indptr
  cscRowInd = A_csc.indices
  cscVal = A_csc.data

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
      cscVal,
  )
  end = time.time()
  print(f"prepare the structure for spmv kernel: {end-start}s", flush=True)

  max_local_nnz = matrix_info["max_local_nnz"]
  max_local_nnz_cols = matrix_info["max_local_nnz_cols"]
  mat_rows_buf = matrix_info["mat_rows_buf"]
  mat_vals_buf = matrix_info["mat_vals_buf"]
  mat_col_idx_buf = matrix_info["mat_col_idx_buf"]
  mat_col_loc_buf = matrix_info["mat_col_loc_buf"]
  mat_col_len_buf = matrix_info["mat_col_len_buf"]
  local_nnz = matrix_info["local_nnz"]
  local_nnz_cols = matrix_info["local_nnz_cols"]

  # blk = per-PE dense vector chunk size = ceil(n / P), same on both axes
  # since the matrix and grid are both square.
  blk = math.ceil(n / P)

  np.random.seed(0)
  x = np.random.rand(n).astype(np.float32)

  print("Generating dense reference y = A @ x ...")
  y_ref = A_csr.dot(x)

  x_hwl = dist_x_to_diag_hwl(n, x, blk, P)

  # fabric-offsets = 1,1
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
  # comment in sdk-hypersparse-spmv/run.py for why (container bind-mount only
  # covers the invocation cwd).
  code_csl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "layout_fp32.csl")

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
      channels,
      width_west_buf,
      width_east_buf,
  )
  end = time.time()
  print(f"Compilation done in {end-start}s", flush=True)

  if args.compile_only:
    print("COMPILE ONLY: EXIT")
    return

  runner = SdkRuntime(dirname, cmaddr=args.cmaddr, suppress_simfab_trace=True)

  sym_x_buf = runner.get_id("x_buf")
  sym_y_buf = runner.get_id("y_buf")
  sym_mat_rows_buf = runner.get_id("mat_rows_buf")
  sym_mat_vals_buf = runner.get_id("mat_vals_buf")
  sym_mat_col_idx_buf = runner.get_id("mat_col_idx_buf")
  sym_mat_col_loc_buf = runner.get_id("mat_col_loc_buf")
  sym_mat_col_len_buf = runner.get_id("mat_col_len_buf")
  sym_local_nnz = runner.get_id("local_nnz")
  sym_local_nnz_cols = runner.get_id("local_nnz_cols")

  start = time.time()
  runner.load()
  end = time.time()
  print(f"*** Load done in {end-start}s")

  start = time.time()
  runner.run()

  print("step 1: copy the structure+values of A and diagonal-seeded x to the device")

  mat_rows_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz, mat_rows_buf, np.uint32)
  runner.memcpy_h2d(sym_mat_rows_buf, mat_rows_buf_1d, 0, 0, width, height, max_local_nnz,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  mat_vals_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz, mat_vals_buf, np.float32)
  runner.memcpy_h2d(sym_mat_vals_buf, mat_vals_buf_1d, 0, 0, width, height, max_local_nnz,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
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

  x_buf_1d = hwl_to_oned_colmajor(height, width, blk, x_hwl, np.float32)
  runner.memcpy_h2d(sym_x_buf, x_buf_1d, 0, 0, width, height, blk,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  print("step 2: spmv")
  runner.launch("f_spmv", nonblock=False)

  print("step 3: fetch the output y_buf (meaningful only at the diagonal PEs)")
  y_buf_1d = np.zeros(height * width * blk, np.float32)
  runner.memcpy_d2h(y_buf_1d, sym_y_buf, 0, 0, width, height, blk,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)

  runner.stop()

  end = time.time()
  print(f"*** Run done in {end-start}s")

  y_hwl = np.reshape(y_buf_1d, (height, width, blk), order="F")
  y_wse = extract_diag_result(n, blk, P, y_hwl)

  ok = verify_result(y_ref, y_wse)
  if not ok:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
