#!/usr/bin/env cs_python
"""SDK hypersparse-spmv (sdk-hypersparse-spmv), host-driven, --iters rounds:
y = A*x, y fed back as the next round's x via the host (memcpy_d2h -> host
redistributes -> memcpy_h2d), exactly the `spmv(x,y); spmv(y,x)` ping-pong
sdk-hypersparse-spmv/README.md notes is "never actually exercised anywhere in
this repo" -- this script is what exercises it, for timing purposes only.

The kernel itself (kernel.csl/layout.csl) is NOT modified -- this reuses
sdk-hypersparse-spmv/run.py's own preprocess()/dist_x_to_hwl()/
unpad_3d_to_1d() unchanged, imported directly rather than re-derived, since
x and y live in genuinely different per-PE distributions (see run.py's own
module comment) and getting that redistribution wrong would silently corrupt
every round after the first.

sdk-hypersparse-spmv is WSE-2 only (no WSE-3 SDK example exists to fork),
and only WSE-3 appliance hardware is available for this comparison -- so
this comparison is simulator-only, on both sides, by necessity. Given that,
h2d/d2h are measured the SAME way run_bfs.py already does (see its own
module docstring): bracket each memcpy block with on-device f_tic/f_toc
(nonblock=True before, nonblock=False -- which blocks until every queued
memcpy in between has actually completed -- after), then decode the
elapsed cycles from the kernel's existing time_buf_u16/f_memcpy_timestamps
mechanism, same as compute already was. This is reliable in the simulator
because it reads the DEVICE's own tsc counter, which the simulator advances
according to its cycle-accurate model of the transfer -- unlike a host-side
time.time() bracket, which is dominated by simulation-stepping overhead
that has nothing to do with real transfer cost (confirmed: an earlier,
now-replaced version of this script using time.time() showed d2h_seconds
~100-1000x h2d_seconds on a toy matrix, an artifact, not a real signal).

The one-time matrix upload (mat_vals_buf & friends, done once before round
0) is counted toward the h2d total, since it's real cost paid to get
--iters rounds of results out of this approach -- not excluded as pure
"setup."

Prints three parseable final lines: H2D_CYCLES=<n> COMPUTE_CYCLES=<n>
D2H_CYCLES=<n>. See bench_device_iters_timing.py for the device-only
counterpart and bench_device_vs_host.py for the orchestrator that runs both
as separate subprocesses (the simulator can't be instantiated twice in one
process, per bench_common.py's own convention) and combines their output.
"""

import argparse
import math
import os
import subprocess
import sys

import numpy as np
from scipy.io import mmread

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import ORIGINAL_SPMV_DIR, fabric_dims, make_u48  # noqa: E402

sys.path.insert(0, ORIGINAL_SPMV_DIR)
import run as sdk_run  # noqa: E402 -- reuses dist_x_to_hwl/unpad_3d_to_1d/hwl_to_oned_colmajor unchanged
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
  p.add_argument("--out_dir", default="out_bench_sdk_iters")
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
  assert n == ncols, "iterative power-iteration round-trip requires a square matrix"
  nnz = A_csr.nnz

  # Same value convention as bench_device_iters_timing.py / run_single_fp32_spmv.py
  # (seed 123) -- without this, this script silently used whatever raw weights
  # mmread() returned (all-1.0 structural, for these RMAT files), which would
  # make a final-y comparison against the device-only script meaningless (different
  # A -> different y, regardless of whether either kernel is implemented correctly).
  np.random.seed(123)
  A_csr.data[0:nnz] = np.random.rand(nnz).astype(np.float32)

  A_csc = A_csr.tocsc(copy=True).sorted_indices().astype(np.float32)
  print(f"matrix: {n}x{n}, nnz={nnz}, grid={P}x{P}, iters={args.iters}")

  os.makedirs(args.out_dir, exist_ok=True)

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
      "mat_vals_buf", "x_tx_buf", "y_local_buf", "mat_rows_buf", "mat_col_idx_buf",
      "mat_col_loc_buf", "mat_col_len_buf", "y_rows_init_buf", "local_nnz", "local_nnz_cols",
      "local_nnz_rows", "time_buf_u16",
  ]}

  runner.load()
  runner.run()
  runner.launch("f_enable_tsc", nonblock=True)

  def bracket_cycles(fn):
    """Runs fn() (a block of memcpy_h2d/memcpy_d2h calls) bracketed by
    on-device f_tic/f_toc, returns the elapsed on-device cycles -- same
    technique run_bfs.py uses for its own h2d_seed/d2h brackets (see this
    module's own docstring for why this is simulator-reliable and host
    wall-clock isn't)."""
    runner.launch("f_tic", nonblock=True)
    fn()
    runner.launch("f_toc", nonblock=False)  # blocks -> every memcpy in fn() above is done
    runner.launch("f_memcpy_timestamps", nonblock=False)
    time_1d = np.zeros(P * P * 6, np.uint32)
    runner.memcpy_d2h(time_1d, sym["time_buf_u16"], 0, 0, P, P, 6, streaming=False,
                       data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.COL_MAJOR,
                       nonblock=False)
    time_hwl = np.reshape(time_1d, (P, P, 6), order="F").astype(np.uint16)
    time_start = np.zeros((P, P), dtype=np.uint64)
    time_end = np.zeros((P, P), dtype=np.uint64)
    for py in range(P):
      for px in range(P):
        time_start[py, px] = make_u48(time_hwl[py, px, 0:3])
        time_end[py, px] = make_u48(time_hwl[py, px, 3:6])
    return int(time_end.max()) - int(time_start.min())

  def h2d(name, arr_hwl, length, dtype, mdtype):
    arr_1d = sdk_run.hwl_to_oned_colmajor(P, P, length, arr_hwl, dtype)
    runner.memcpy_h2d(sym[name], arr_1d, 0, 0, P, P, length, streaming=False,
                       data_type=mdtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

  # One-time matrix upload -- real cost paid to get `iters` rounds of results,
  # counted toward h2d_cycles (see module docstring).
  h2d_cycles = bracket_cycles(lambda: (
      h2d("mat_vals_buf", info["mat_vals_buf"], info["max_local_nnz"], np.float32,
          MemcpyDataType.MEMCPY_32BIT),
      h2d("mat_rows_buf", info["mat_rows_buf"], info["max_local_nnz"], np.uint32,
          MemcpyDataType.MEMCPY_16BIT),
      h2d("mat_col_idx_buf", info["mat_col_idx_buf"], info["max_local_nnz_cols"], np.uint32,
          MemcpyDataType.MEMCPY_16BIT),
      h2d("mat_col_loc_buf", info["mat_col_loc_buf"], info["max_local_nnz_cols"], np.uint32,
          MemcpyDataType.MEMCPY_16BIT),
      h2d("mat_col_len_buf", info["mat_col_len_buf"], info["max_local_nnz_cols"], np.uint32,
          MemcpyDataType.MEMCPY_16BIT),
      h2d("y_rows_init_buf", info["y_rows_init_buf"], info["max_local_nnz_rows"], np.uint32,
          MemcpyDataType.MEMCPY_16BIT),
      h2d("local_nnz", info["local_nnz"], 1, np.uint32, MemcpyDataType.MEMCPY_16BIT),
      h2d("local_nnz_cols", info["local_nnz_cols"], 1, np.uint32, MemcpyDataType.MEMCPY_16BIT),
      h2d("local_nnz_rows", info["local_nnz_rows"], 1, np.uint32, MemcpyDataType.MEMCPY_16BIT),
  ))

  compute_cycles = 0
  d2h_cycles = 0

  # Seed x for round 0 (random, matching fp32_diag_spmv's own seed=0 --
  # values don't affect the timing story, only structure/degree does, but
  # keeping them comparable avoids a spurious "different workload" objection).
  np.random.seed(0)
  x_vec = np.random.rand(n).astype(np.float32)

  for r in range(args.iters):
    x_hwl = sdk_run.dist_x_to_hwl(n, x_vec, local_vec_sz, P, P)
    h2d_cycles += bracket_cycles(
        lambda: h2d("x_tx_buf", x_hwl, local_vec_sz, np.float32, MemcpyDataType.MEMCPY_32BIT))

    compute_cycles += bracket_cycles(lambda: runner.launch("f_spmv", nonblock=False))

    y_1d = np.zeros(P * P * local_out_vec_sz, np.float32)

    def read_y():
      runner.memcpy_d2h(y_1d, sym["y_local_buf"], 0, 0, P, P, local_out_vec_sz, streaming=False,
                         data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.COL_MAJOR,
                         nonblock=False)

    d2h_cycles += bracket_cycles(read_y)

    y_hwl = sdk_run.oned_to_hwl_colmajor(P, P, local_out_vec_sz, y_1d, np.float32)
    y_vec = sdk_run.unpad_3d_to_1d(n, y_hwl)[0:n]
    x_vec = y_vec  # feed this round's result back as the next round's input

  runner.stop()

  if args.dump_y:
    np.save(args.dump_y, x_vec)  # x_vec holds the final round's y after the last feed-back
    print(f"dumped final y to {args.dump_y}")

  print(f"H2D_CYCLES={h2d_cycles}")
  print(f"COMPUTE_CYCLES={compute_cycles}")
  print(f"D2H_CYCLES={d2h_cycles}")


if __name__ == "__main__":
  main()
