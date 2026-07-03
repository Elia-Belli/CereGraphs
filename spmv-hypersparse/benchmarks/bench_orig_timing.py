#!/usr/bin/env cs_python
"""Raw (no clock-sync) on-device timing for the original hypersparse_spmv
kernel: memcpy excluded, measures broadcast+compute+reduce only. See
bench_common.py for shared helpers and bench_bool_timing.py for the
boolean-kernel counterpart -- run separately because the simulator cannot be
instantiated twice in one process.

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
from bench_common import ORIGINAL_SPMV_DIR, fabric_dims, hwl_to_oned_colmajor, make_u48
from bench_log import log_result

sys.path.insert(0, ORIGINAL_SPMV_DIR)
from preprocess import preprocess as preprocess_orig  # noqa: E402

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
  p.add_argument("--out_dir", default="out_bench_orig")
  p.add_argument("--skip_compile", action="store_true")
  return p.parse_args()


def main():
  args = parse_args()
  P = args.num_pe_cols
  assert args.num_pe_cols == args.num_pe_rows

  A_coo = mmread(args.infile_mtx)
  A_csr = A_coo.tocsr(copy=True).sorted_indices().astype(np.float32)
  n, ncols = A_csr.shape
  assert n == ncols
  A_csc = A_csr.tocsc(copy=True).sorted_indices().astype(np.float32)
  nnz = A_csr.nnz
  print(f"matrix: {n}x{n}, nnz={nnz}, grid={P}x{P}")

  info = preprocess_orig(n, n, nnz, P, P, A_csr.indptr, A_csr.indices,
                          A_csc.indptr, A_csc.indices, A_csc.data)

  local_vec_sz = math.ceil(math.ceil(n / P) / P)
  local_out_vec_sz = local_vec_sz
  out_pad_start_idx = math.ceil(n / P)

  fabric_w, fabric_h, core_x, core_y = fabric_dims(P, P)
  layout_csl = os.path.join(ORIGINAL_SPMV_DIR, "src", "layout.csl")
  compile_args = [
      args.driver, layout_csl, f"--fabric-dims={fabric_w},{fabric_h}",
      f"--fabric-offsets={core_x},{core_y}",
      f"--params=ncols:{n}", f"--params=nrows:{n}", f"--params=pcols:{P}",
      f"--params=prows:{P}", f"--params=max_local_nnz:{info['max_local_nnz']}",
      f"--params=max_local_nnz_cols:{info['max_local_nnz_cols']}",
      f"--params=max_local_nnz_rows:{info['max_local_nnz_rows']}",
      f"--params=local_vec_sz:{local_vec_sz}", f"--params=local_out_vec_sz:{local_out_vec_sz}",
      f"--params=y_pad_start_row_idx:{out_pad_start_idx}",
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
      "mat_vals_buf", "x_tx_buf", "mat_rows_buf", "mat_col_idx_buf", "mat_col_loc_buf",
      "mat_col_len_buf", "y_rows_init_buf", "local_nnz", "local_nnz_cols", "local_nnz_rows",
      "time_buf_u16",
  ]}

  runner.load()
  runner.run()

  # x = all ones; values don't affect timing (wavelet counts are structural)
  vec_len_per_pe_col = math.ceil(n / P)
  pad_len = vec_len_per_pe_col * P - n
  x_ref = np.ones(n, dtype=np.float32)
  invec = np.append(x_ref, np.ones(pad_len, dtype=np.float32)) if pad_len > 0 else x_ref
  pad_len_per_pe_col = local_vec_sz * P - vec_len_per_pe_col
  x_hwl = np.zeros((P, P, local_vec_sz), np.float32)
  for col in range(P):
    invec_col = invec[col * vec_len_per_pe_col:(col + 1) * vec_len_per_pe_col]
    if pad_len_per_pe_col > 0:
      invec_col = np.append(invec_col, np.ones(pad_len_per_pe_col, dtype=np.float32))
    for row in range(P):
      x_hwl[(row, col)] = invec_col[row * local_vec_sz:(row + 1) * local_vec_sz]

  def h2d(name, arr_hwl, length, dtype, mdtype):
    arr_1d = hwl_to_oned_colmajor(P, P, length, arr_hwl, dtype)
    runner.memcpy_h2d(sym[name], arr_1d, 0, 0, P, P, length, streaming=False,
                       data_type=mdtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

  h2d("mat_vals_buf", info["mat_vals_buf"], info["max_local_nnz"], np.float32,
      MemcpyDataType.MEMCPY_32BIT)
  h2d("x_tx_buf", x_hwl, local_vec_sz, np.float32, MemcpyDataType.MEMCPY_32BIT)
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

  # deliberately skip f_sync/f_reference_timestamps -- raw timing only
  runner.launch("f_enable_tsc", nonblock=True)
  runner.launch("f_tic", nonblock=True)
  runner.launch("f_spmv", nonblock=False)
  runner.launch("f_toc", nonblock=False)
  runner.launch("f_memcpy_timestamps", nonblock=False)

  time_1d = np.zeros(P * P * 6, np.uint32)
  runner.memcpy_d2h(time_1d, sym["time_buf_u16"], 0, 0, P, P, 6, streaming=False,
                     data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.COL_MAJOR,
                     nonblock=False)
  runner.stop()

  time_hwl = np.reshape(time_1d, (P, P, 6), order="F").astype(np.uint16)
  time_start = np.zeros((P, P), dtype=np.uint64)
  time_end = np.zeros((P, P), dtype=np.uint64)
  for py in range(P):
    for px in range(P):
      time_start[py, px] = make_u48(time_hwl[py, px, 0:3])
      time_end[py, px] = make_u48(time_hwl[py, px, 3:6])

  cycles = int(time_end.max()) - int(time_start.min())
  print(f"raw cycles = {cycles}")
  log_result("original_hypersparse_spmv_f32", args.infile_mtx, n, nnz, P, cycles)
  print(f"CYCLES={cycles}")


if __name__ == "__main__":
  main()
