#!/usr/bin/env cs_python
"""Raw (no clock-sync) on-device timing for bool_pe.csl (diagonal-reduce,
boolean, <collectives_2d>): memcpy excluded, measures broadcast+compute+reduce
only. Companion to bench_orig_timing.py -- run separately because the
simulator cannot be instantiated twice in one process.

Prints a final line "CYCLES=<n>" for easy parsing by a caller.
"""

import argparse
import math
import os
import subprocess
import sys

import numpy as np
from scipy.io import mmread

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import BOOL_DIAG_SPMV_DIR, fabric_dims, hwl_to_oned_colmajor, make_u48
from bench_log import log_result

sys.path.insert(0, BOOL_DIAG_SPMV_DIR)
from preprocess_bool import preprocess as preprocess_bool  # noqa: E402
from run_bool import dist_x_to_diag_hwl  # noqa: E402

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--infile_mtx", required=True)
  p.add_argument("--num_pe_cols", type=int, required=True)
  p.add_argument("--num_pe_rows", type=int, required=True)
  p.add_argument("--driver", default="cslc")
  p.add_argument("--arch", default="wse2")
  p.add_argument("--channels", type=int, default=1)
  p.add_argument("--cmaddr")
  p.add_argument("--out_dir", default="out_bench_bool")
  p.add_argument("--skip_compile", action="store_true")
  return p.parse_args()


def main():
  args = parse_args()
  P = args.num_pe_cols
  assert args.num_pe_cols == args.num_pe_rows

  A_coo = mmread(args.infile_mtx)
  A_csr = A_coo.tocsr(copy=True).sorted_indices()
  n, ncols = A_csr.shape
  assert n == ncols
  A_csc = A_csr.tocsc(copy=True).sorted_indices()
  nnz = A_csr.nnz
  print(f"matrix: {n}x{n}, nnz={nnz}, grid={P}x{P}")

  info = preprocess_bool(n, n, nnz, P, P, A_csr.indptr, A_csr.indices,
                          A_csc.indptr, A_csc.indices)
  blk = math.ceil(n / P)

  fabric_w, fabric_h, core_x, core_y = fabric_dims(P, P)
  layout_bool_csl = os.path.join(BOOL_DIAG_SPMV_DIR, "src", "layout_bool.csl")
  compile_args = [
      args.driver, layout_bool_csl, f"--fabric-dims={fabric_w},{fabric_h}",
      f"--fabric-offsets={core_x},{core_y}",
      f"--params=pcols:{P}", f"--params=prows:{P}", f"--params=blk:{blk}",
      f"--params=max_local_nnz:{info['max_local_nnz']}",
      f"--params=max_local_nnz_cols:{info['max_local_nnz_cols']}",
      f"--params=max_local_nnz_rows:{info['max_local_nnz_rows']}",
      f"-o={args.out_dir}", f"--arch={args.arch}", "--memcpy", f"--channels={args.channels}",
      "--width-west-buf=0", "--width-east-buf=0",
  ]
  if args.skip_compile:
    print(f"skipping compile, reusing existing {args.out_dir}")
  else:
    print(f"$ {' '.join(compile_args)}")
    subprocess.check_call(compile_args)

  runner = SdkRuntime(args.out_dir, cmaddr=args.cmaddr)
  sym = {name: runner.get_id(name) for name in [
      "x_buf", "mat_rows_buf", "mat_col_idx_buf", "mat_col_loc_buf", "mat_col_len_buf",
      "y_rows_init_buf", "local_nnz", "local_nnz_cols", "local_nnz_rows",
      "tsc_start_buffer", "tsc_end_buffer",
  ]}

  runner.load()
  runner.run()

  np.random.seed(0)
  x_bool = (np.random.rand(n) < 0.5)
  x_hwl = dist_x_to_diag_hwl(n, x_bool, blk, P)

  def h2d(name, arr_hwl, length, dtype, mdtype):
    arr_1d = hwl_to_oned_colmajor(P, P, length, arr_hwl, dtype)
    runner.memcpy_h2d(sym[name], arr_1d, 0, 0, P, P, length, streaming=False,
                       data_type=mdtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

  h2d("mat_rows_buf", info["mat_rows_buf"], info["max_local_nnz"], np.uint32,
      MemcpyDataType.MEMCPY_16BIT)
  h2d("mat_col_idx_buf", info["mat_col_idx_buf"], info["max_local_nnz_cols"], np.uint32,
      MemcpyDataType.MEMCPY_16BIT)
  h2d("mat_col_loc_buf", info["mat_col_loc_buf"], info["max_local_nnz_cols"], np.uint32,
      MemcpyDataType.MEMCPY_16BIT)
  h2d("mat_col_len_buf", info["mat_col_len_buf"], info["max_local_nnz_cols"], np.uint32,
      MemcpyDataType.MEMCPY_16BIT)
  h2d("y_rows_init_buf", info["y_rows_init_buf"], info["max_local_nnz_rows"], np.uint32,
      MemcpyDataType.MEMCPY_16BIT)
  h2d("local_nnz", info["local_nnz"], 1, np.uint32, MemcpyDataType.MEMCPY_16BIT)
  h2d("local_nnz_cols", info["local_nnz_cols"], 1, np.uint32, MemcpyDataType.MEMCPY_16BIT)
  h2d("local_nnz_rows", info["local_nnz_rows"], 1, np.uint32, MemcpyDataType.MEMCPY_16BIT)
  h2d("x_buf", x_hwl, blk, np.float32, MemcpyDataType.MEMCPY_32BIT)

  runner.launch("f_enable_tsc", nonblock=True)
  runner.launch("f_tic", nonblock=True)
  runner.launch("f_spmv", nonblock=False)
  runner.launch("f_toc", nonblock=False)

  start_1d = np.zeros(P * P * 3, np.uint32)
  runner.memcpy_d2h(start_1d, sym["tsc_start_buffer"], 0, 0, P, P, 3, streaming=False,
                     data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.COL_MAJOR,
                     nonblock=False)
  end_1d = np.zeros(P * P * 3, np.uint32)
  runner.memcpy_d2h(end_1d, sym["tsc_end_buffer"], 0, 0, P, P, 3, streaming=False,
                     data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.COL_MAJOR,
                     nonblock=False)
  runner.stop()

  start_hwl = np.reshape(start_1d, (P, P, 3), order="F").astype(np.uint16)
  end_hwl = np.reshape(end_1d, (P, P, 3), order="F").astype(np.uint16)
  time_start = np.zeros((P, P), dtype=np.uint64)
  time_end = np.zeros((P, P), dtype=np.uint64)
  for py in range(P):
    for px in range(P):
      time_start[py, px] = make_u48(start_hwl[py, px, :])
      time_end[py, px] = make_u48(end_hwl[py, px, :])

  cycles = int(time_end.max()) - int(time_start.min())
  print(f"raw cycles = {cycles}")
  log_result("bool_pe_diagonal_reduce", args.infile_mtx, n, nnz, P, cycles)
  print(f"CYCLES={cycles}")


if __name__ == "__main__":
  main()
