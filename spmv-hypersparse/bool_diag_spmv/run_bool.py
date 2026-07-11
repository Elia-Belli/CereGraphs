#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args
""" test boolean-semiring sparse matrix-vector multiplication, one iteration

  Forked from run.py for the boolean/diagonal-reduce variant (see src/bool_pe.csl / src/layout_bool.csl): 
    - A is a square boolean adjacency matrix, 
    - y = OR_j (A[i,j] AND x[j]),
    - the PE grid must be square so every row/column has a diagonal PE.

  The input vector x is seeded only at the diagonal PEs (host memcpy); 
  1. Phase 1: (<collectives_2d> mpi_y.broadcast) distributes each column's x-block from
    its diagonal PE to the rest of the column. 
  2. Phase 2: (mpi_x.reduce_fadds) reduces every row's local boolean contributions to that row's diagonal PE,
    which ends up holding the row's final result. 
  The host reads back the full PE rectangle and keeps only the diagonal entries.

  How to compile and run
     python run_bool.py --arch=wse2 --num_pe_cols=4 --num_pe_rows=4 --channels=1
        --driver=<path to cslc> --compile-only --infile_mtx=<path to mtx file>
     python run_bool.py --arch=wse2 --num_pe_cols=4 --num_pe_rows=4 --channels=1
        --run-only --infile_mtx=<path to mtx file>
"""

import math
import os
import subprocess
import time
from typing import Optional

import numpy as np
from cmd_parser import parse_args
from preprocess_bool import preprocess
from scipy.io import mmread

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)


def hwl_to_oned_colmajor(height: int, width: int, pe_length: int, A_hwl: np.ndarray, dtype):
  """
    Given a 3-D tensor A[height][width][pe_length], transform it to
    1D array by column-major
    """
  if A_hwl.dtype == np.float32:
    A_1d = np.zeros(height * width * pe_length, dtype)
    idx = 0
    for l in range(pe_length):
      for w in range(width):
        for h in range(height):
          A_1d[idx] = A_hwl[(h, w, l)]
          idx = idx + 1
  elif A_hwl.dtype == np.uint16:
    assert dtype == np.uint32, "only support dtype = u32 if A is u16"
    A_1d = np.zeros(height * width * pe_length, dtype)
    idx = 0
    for l in range(pe_length):
      for w in range(width):
        for h in range(height):
          x = A_hwl[(h, w, l)]
          A_1d[idx] = np.uint32(x)
          idx = idx + 1
  else:
    raise RuntimeError(f"{A_hwl.dtype} is not supported")

  return A_1d


def oned_to_hwl_colmajor(height: int, width: int, pe_length: int, A_1d: np.ndarray, dtype):
  """
    Given a 1-D tensor A_1d[height*width*pe_length], transform it to
    3-D tensor A[height][width][pe_length] by column-major
    """
  assert dtype == np.float32, "only support f32 readback for this kernel"
  assert A_1d.dtype == np.float32, "only support f32 to f32"
  return np.reshape(A_1d, (height, width, pe_length), order="F")


# x is boolean, length n. Only the diagonal PE of each column (py == px)
# gets a real slice; every other PE starts at zero and receives the
# broadcast from phase 1. This replaces hypersparse_spmv's dist_x_to_hwl,
# which spread x across every PE in a column.
def dist_x_to_diag_hwl(n, x_bool, blk, P):
  x_pad = np.zeros(P * blk, dtype=np.float32)
  x_pad[0:n] = x_bool.astype(np.float32)

  x_hwl = np.zeros((P, P, blk), dtype=np.float32)
  for p in range(P):
    x_hwl[(p, p)] = x_pad[p * blk:(p + 1) * blk]
  return x_hwl


# Extract the diagonal PEs' y_buf (the only ones holding a meaningful final
# result) and reassemble into the length-n boolean output vector.
def extract_diag_result(n, blk, P, y_hwl):
  parts = [y_hwl[(p, p)] for p in range(P)]
  y_pad = np.concatenate(parts)
  return y_pad[0:n] > 0.0


def generate_boolean_reference(nrows, ncols, csrRowPtr, csrColInd, x_bool):
  from scipy import sparse
  ones = np.ones(len(csrColInd), dtype=np.float32)
  mat = sparse.csr_matrix((ones, csrColInd, csrRowPtr), shape=(nrows, ncols))
  y = mat.dot(x_bool.astype(np.float32))
  return y > 0.0


def verify_result(ref, res):
  print("Comparing boolean result with reference...")
  n_mismatch = np.sum(ref != res)
  print(f"reference[{len(ref)}]: \n{ref.astype(int)}")
  print(f"result   [{len(res)}]: \n{res.astype(int)}")
  print(f"[[ Mismatches: {n_mismatch} / {len(ref)} ]]")
  result = "PASS" if n_mismatch == 0 else "FAIL"
  print(f"[[ Result: {result} ]]")
  if n_mismatch != 0:
    idx = np.where(ref != res)[0]
    print(f"mismatched indices: {idx}")


def csl_compile_core(
    cslc: str,
    file_config: str,
    elf_dir: str,
    fabric_width: int,
    fabric_height: int,
    core_fabric_offset_x: int,
    core_fabric_offset_y: int,
    use_precompile: bool,
    arch: Optional[str],
    np_cols: int,
    np_rows: int,
    blk: int,
    max_local_nnz: int,
    max_local_nnz_cols: int,
    max_local_nnz_rows: int,
    channels: int,
    width_west_buf: int,
    width_east_buf: int,
    max_rounds: Optional[int] = None,
):
  comp_dir = elf_dir

  if not use_precompile:
    args = []
    args.append(cslc)
    args.append(file_config)
    args.append(f"--fabric-dims={fabric_width},{fabric_height}")
    args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
    args.append(f"--params=pcols:{np_cols}")
    args.append(f"--params=prows:{np_rows}")
    args.append(f"--params=blk:{blk}")
    args.append(f"--params=max_local_nnz:{max_local_nnz}")
    args.append(f"--params=max_local_nnz_cols:{max_local_nnz_cols}")
    args.append(f"--params=max_local_nnz_rows:{max_local_nnz_rows}")
    # left at layout_bool.csl's own default (32) unless a caller (see
    # bench_timing.py) needs per-round timing over a deeper BFS.
    if max_rounds is not None:
      args.append(f"--params=max_rounds:{max_rounds}")

    args.append(f"-o={comp_dir}")
    if arch is not None:
      args.append(f"--arch={arch}")
    args.append("--memcpy")
    args.append(f"--channels={channels}")
    args.append(f"--width-west-buf={width_west_buf}")
    args.append(f"--width-east-buf={width_east_buf}")

    print(f"subprocess.check_call(args = {args}")
    subprocess.check_call(args)
  else:
    print("[csl_compile_core] use pre-compile ELFs")


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
  x_bool = (np.random.rand(n) < 0.5)

  print("Generating boolean reference y = OR_j (A[i,j] AND x[j]) ...")
  y_ref = generate_boolean_reference(nrows, ncols, csrRowPtr, csrColInd, x_bool)

  x_hwl = dist_x_to_diag_hwl(n, x_bool, blk, P)

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
  # comment in original_spmv/run.py for why (container bind-mount only overs the invocation cwd).
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
  print(f"Compilation done in {end-start}s", flush=True)

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

  start = time.time()
  runner.run()

  print("step 1: copy the structure of A and diagonal-seeded x to the device")

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

  x_buf_1d = hwl_to_oned_colmajor(height, width, blk, x_hwl, np.float32)
  runner.memcpy_h2d(sym_x_buf, x_buf_1d, 0, 0, width, height, blk,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  print("step 2: spmv")
  runner.launch("f_spmv", nonblock=False)

  print("step 3: fetch the output vector y (f32, meaningful only at the diagonal PEs)")
  y_1d = np.zeros(height * width * blk, np.float32)
  runner.memcpy_d2h(y_1d, sym_y_buf, 0, 0, width, height, blk,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)

  runner.stop()

  end = time.time()
  print(f"*** Run done in {end-start}s")

  y_hwl = oned_to_hwl_colmajor(height, width, blk, y_1d, np.float32)
  y_wse = extract_diag_result(n, blk, P, y_hwl)

  verify_result(y_ref, y_wse)


if __name__ == "__main__":
  main()
