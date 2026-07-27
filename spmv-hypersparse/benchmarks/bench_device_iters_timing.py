#!/usr/bin/env cs_python
"""fp32_diag_spmv, device-only, --iters rounds: y = A*y run entirely
on-fabric via a single f_spmv_iter launch, no host round trip between
rounds. Counterpart to bench_sdk_iters_timing.py -- see that script's module
docstring for why this is a simulator-only comparison (sdk-hypersparse-spmv
is WSE-2 only, and only WSE-3 appliance hardware is available) and for the
shared h2d/d2h measurement technique: bracket each memcpy block with
on-device f_tic/f_toc (reliable in the simulator, since it reads the
device's own tsc counter, unlike host wall-clock which is dominated by
simulation-stepping overhead).

Only one h2d bracket (matrix + seed x + num_iters, all once) and one d2h
bracket (the final round's y) exist here -- there's no per-round host round
trip to time, which is exactly the point of this comparison. Compute cycles
come from a bracket around the one f_spmv_iter launch (all --iters rounds
in one continuous on-fabric span, not summed per-round launches like the
SDK side).

Prints the same three parseable final lines: H2D_CYCLES=<n>
COMPUTE_CYCLES=<n> D2H_CYCLES=<n>.
"""

import argparse
import math
import os
import subprocess
import sys

import numpy as np
from scipy.io import mmread

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import FP32_DIAG_SPMV_DIR, fabric_dims, make_u48  # noqa: E402

sys.path.insert(0, FP32_DIAG_SPMV_DIR)
from device_io import dist_x_to_diag_hwl, extract_diag_result, hwl_to_oned_colmajor  # noqa: E402
from preprocess_fp32 import preprocess as preprocess_fp32  # noqa: E402

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
  p.add_argument("--out_dir", default="out_bench_device_iters")
  p.add_argument("--skip_compile", action="store_true")
  p.add_argument("--iters", type=int, default=10)
  p.add_argument("--dump_y", help="if set, np.save the final round's y vector to this path")
  return p.parse_args()


def main():
  args = parse_args()
  P = args.num_pe_cols
  assert args.num_pe_cols == args.num_pe_rows

  A_coo = mmread(args.infile_mtx)
  A_csr = A_coo.tocsr(copy=True).sorted_indices().astype(np.float32)
  n, ncols = A_csr.shape
  assert n == ncols, "diagonal-reduce SpMV requires a square matrix"
  nnz = A_csr.nnz

  # Same value convention as run_single_fp32_spmv.py: random fp32 weights
  # (seed 123), matching the SDK side's default (--is_weight_one off).
  np.random.seed(123)
  A_csr.data[0:nnz] = np.random.rand(nnz).astype(np.float32)

  A_csc = A_csr.tocsc(copy=True).sorted_indices().astype(np.float32)
  print(f"matrix: {n}x{n}, nnz={nnz}, grid={P}x{P}, iters={args.iters}")

  os.makedirs(args.out_dir, exist_ok=True)

  info = preprocess_fp32(n, n, nnz, P, P, A_csr.indptr, A_csr.indices,
                          A_csc.indptr, A_csc.indices, A_csc.data)

  blk = math.ceil(n / P)

  fabric_w, fabric_h, core_x, core_y = fabric_dims(P, P)
  layout_csl = os.path.join(FP32_DIAG_SPMV_DIR, "src", "layout_fp32.csl")
  compile_args = [
      args.driver, layout_csl, f"--fabric-dims={fabric_w},{fabric_h}",
      f"--fabric-offsets={core_x},{core_y}",
      f"--params=pcols:{P}", f"--params=prows:{P}", f"--params=blk:{blk}",
      f"--params=max_local_nnz:{info['max_local_nnz']}",
      f"--params=max_local_nnz_cols:{info['max_local_nnz_cols']}",
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
      "mat_rows_buf", "mat_vals_buf", "mat_col_idx_buf", "mat_col_loc_buf", "mat_col_len_buf",
      "local_nnz", "local_nnz_cols", "x_buf", "y_buf", "num_iters", "tsc_start_buffer",
      "tsc_end_buffer",
  ]}

  runner.load()
  runner.run()
  runner.launch("f_enable_tsc", nonblock=True)

  def bracket_cycles(fn):
    """Runs fn() (a block of memcpy_h2d/memcpy_d2h calls) bracketed by
    on-device f_tic/f_toc, returns the elapsed on-device cycles -- see
    bench_sdk_iters_timing.py's module docstring for why this (not host
    wall-clock) is the simulator-reliable way to measure this."""
    runner.launch("f_tic", nonblock=True)
    fn()
    runner.launch("f_toc", nonblock=False)  # blocks -> every memcpy in fn() above is done
    tsc_start_1d = np.zeros(P * P * 3, np.uint32)
    runner.memcpy_d2h(tsc_start_1d, sym["tsc_start_buffer"], 0, 0, P, P, 3, streaming=False,
                       data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.COL_MAJOR,
                       nonblock=False)
    tsc_end_1d = np.zeros(P * P * 3, np.uint32)
    runner.memcpy_d2h(tsc_end_1d, sym["tsc_end_buffer"], 0, 0, P, P, 3, streaming=False,
                       data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.COL_MAJOR,
                       nonblock=False)
    tsc_start_hwl = np.reshape(tsc_start_1d, (P, P, 3), order="F").astype(np.uint16)
    tsc_end_hwl = np.reshape(tsc_end_1d, (P, P, 3), order="F").astype(np.uint16)
    time_start = np.zeros((P, P), dtype=np.uint64)
    time_end = np.zeros((P, P), dtype=np.uint64)
    for py in range(P):
      for px in range(P):
        time_start[py, px] = make_u48(tsc_start_hwl[py, px, 0:3])
        time_end[py, px] = make_u48(tsc_end_hwl[py, px, 0:3])
    return int(time_end.max()) - int(time_start.min())

  def h2d(name, arr_1d, length, mdtype):
    runner.memcpy_h2d(sym[name], arr_1d, 0, 0, P, P, length, streaming=False,
                       data_type=mdtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

  h2d_cycles = bracket_cycles(lambda: (
      h2d("mat_rows_buf", hwl_to_oned_colmajor(P, P, info["max_local_nnz"], info["mat_rows_buf"],
          np.uint32), info["max_local_nnz"], MemcpyDataType.MEMCPY_16BIT),
      h2d("mat_vals_buf", hwl_to_oned_colmajor(P, P, info["max_local_nnz"], info["mat_vals_buf"],
          np.float32), info["max_local_nnz"], MemcpyDataType.MEMCPY_32BIT),
      h2d("mat_col_idx_buf", hwl_to_oned_colmajor(P, P, info["max_local_nnz_cols"],
          info["mat_col_idx_buf"], np.uint32), info["max_local_nnz_cols"],
          MemcpyDataType.MEMCPY_16BIT),
      h2d("mat_col_loc_buf", hwl_to_oned_colmajor(P, P, info["max_local_nnz_cols"],
          info["mat_col_loc_buf"], np.uint32), info["max_local_nnz_cols"],
          MemcpyDataType.MEMCPY_16BIT),
      h2d("mat_col_len_buf", hwl_to_oned_colmajor(P, P, info["max_local_nnz_cols"],
          info["mat_col_len_buf"], np.uint32), info["max_local_nnz_cols"],
          MemcpyDataType.MEMCPY_16BIT),
      h2d("local_nnz", hwl_to_oned_colmajor(P, P, 1, info["local_nnz"], np.uint32), 1,
          MemcpyDataType.MEMCPY_16BIT),
      h2d("local_nnz_cols", hwl_to_oned_colmajor(P, P, 1, info["local_nnz_cols"], np.uint32), 1,
          MemcpyDataType.MEMCPY_16BIT),
  ))

  np.random.seed(0)
  x_vec = np.random.rand(n).astype(np.float32)
  x_hwl = dist_x_to_diag_hwl(n, x_vec, blk, P)
  num_iters_hwl = np.full((P, P, 1), args.iters, dtype=np.uint16)

  h2d_cycles += bracket_cycles(lambda: (
      h2d("x_buf", hwl_to_oned_colmajor(P, P, blk, x_hwl, np.float32), blk,
          MemcpyDataType.MEMCPY_32BIT),
      h2d("num_iters", hwl_to_oned_colmajor(P, P, 1, num_iters_hwl, np.uint32), 1,
          MemcpyDataType.MEMCPY_16BIT),
  ))

  compute_cycles = bracket_cycles(lambda: runner.launch("f_spmv_iter", nonblock=False))

  y_1d = np.zeros(P * P * blk, np.float32)

  def read_y():
    runner.memcpy_d2h(y_1d, sym["y_buf"], 0, 0, P, P, blk, streaming=False,
                       data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.COL_MAJOR,
                       nonblock=False)

  d2h_cycles = bracket_cycles(read_y)

  runner.stop()

  if args.dump_y:
    y_hwl = np.reshape(y_1d, (P, P, blk), order="F")
    y_vec = extract_diag_result(n, blk, P, y_hwl)
    np.save(args.dump_y, y_vec)
    print(f"dumped final y to {args.dump_y}")

  print(f"H2D_CYCLES={h2d_cycles}")
  print(f"COMPUTE_CYCLES={compute_cycles}")
  print(f"D2H_CYCLES={d2h_cycles}")


if __name__ == "__main__":
  main()
