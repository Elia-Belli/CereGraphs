#!/usr/bin/env cs_python
""" Standalone, isolated correctness test for collectives_2d/pe.csl's new
  reduce_or() (bitwise OR-reduce) -- stage 2a of the bitmap work (see
  docs/GRAPH500_BENCHMARK.md / the bitmap-branch plan). No matrix/BFS logic at
  all: every PE seeds a random [count]u32 buffer, one mpi_x.reduce_or() call
  per compile OR-combines each row toward a compile-time `root` column, and
  the result is compared against a plain numpy bitwise-OR reduction across
  each row.

  Tests one root position per process (the simulator backend can't be
  instantiated twice in the same process -- confirmed empirically, a second
  compile+run in one process crashes with a Simfabric::Create assertion
  unrelated to kernel correctness). With no --root given, re-invokes itself
  as a subprocess once per root position (0, middle, NUM_PES-1) instead --
  since transfer_data_reduce_or()'s branching depends on where root sits
  relative to the grid's extremes, exactly the cases transfer_data_reduce()
  itself special-cases, per collectives_2d/pe.csl's own comments.

  How to run
     cs_python run_reduce_or_test.py --arch=wse3 --num_pe_cols=4 --num_pe_rows=4 \
        --driver=<path to cslc> --count=8
     cs_python run_reduce_or_test.py --arch=wse3 --num_pe_cols=4 --num_pe_rows=4 \
        --count=8 --root=2   # single root position, one process
"""

import argparse
import os
import sys
import time

import numpy as np


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_pe_cols", type=int, required=True, help="width of the core rectangle")
  parser.add_argument("--num_pe_rows", type=int, required=True, help="height of the core rectangle")
  parser.add_argument("--count", type=int, default=8,
                       help="number of u32 words per PE (mirrors bool_pe.csl's BITMAP_WORDS; "
                            "default 8 matches the s12/16x16 scale's blk=256 -> BITMAP_WORDS=8)")
  parser.add_argument("--root", type=int, default=None,
                       help="single root column to test (default: re-invoke self once per "
                            "root in {0, P//2, P-1}, one subprocess each)")
  parser.add_argument("--fabric-dims", help="Fabric dimension, i.e. <W>,<H>")
  parser.add_argument("--compile-only", action="store_true")
  parser.add_argument("--run-only", action="store_true")
  parser.add_argument("--width-west-buf", default=0, type=int)
  parser.add_argument("--width-east-buf", default=0, type=int)
  parser.add_argument("--channels", default=1, type=int)
  parser.add_argument("-d", "--driver", help="path to the CSL compiler")
  parser.add_argument("--cmaddr", help="CM address and port, i.e. <IP>:<port>")
  parser.add_argument("--arch", help="wse2 or wse3")
  parser.add_argument("--latestlink", default="out/reduce_or_test",
                       help="folder to contain the log files")
  return parser.parse_args()


def compile_and_run(cslc, code_csl, dirname, args, P, count, root, send_hwl):
  """One compile+run for a single root position. Returns recv_hwl, shape
  (P, P, count) uint32 -- only row p's own recv_hwl[p, root, :] is
  meaningful (every other column in that row is a non-root participant)."""
  fabric_offset_x = 1
  fabric_offset_y = 1
  core_fabric_offset_x = fabric_offset_x + 3 + args.width_west_buf
  core_fabric_offset_y = fabric_offset_y
  min_fabric_width = core_fabric_offset_x + P + 2 + 1 + args.width_east_buf
  min_fabric_height = core_fabric_offset_y + P + 1

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

  if not args.run_only:
    cmd = [
        cslc, code_csl,
        f"--fabric-dims={fabric_width},{fabric_height}",
        f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}",
        f"--params=pcols:{P}", f"--params=prows:{P}",
        f"--params=count:{count}", f"--params=root:{root}",
        f"-o={dirname}",
    ]
    if args.arch is not None:
      cmd.append(f"--arch={args.arch}")
    cmd.append("--memcpy")
    cmd.append(f"--channels={args.channels}")
    cmd.append(f"--width-west-buf={args.width_west_buf}")
    cmd.append(f"--width-east-buf={args.width_east_buf}")
    print(f"subprocess.check_call(args = {cmd}")
    start = time.time()
    import subprocess
    subprocess.check_call(cmd)
    print(f"Compilation done in {time.time()-start}s", flush=True)

  if args.compile_only:
    return None

  # local import: only needed for the actual run, not the orchestrating
  # (no --root) invocation that just spawns one subprocess per root.
  from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
      MemcpyDataType, MemcpyOrder, SdkRuntime,
  )

  runner = SdkRuntime(dirname, cmaddr=args.cmaddr, suppress_simfab_trace=True)
  sym_send_buf = runner.get_id("send_buf")
  sym_recv_buf = runner.get_id("recv_buf")

  runner.load()
  runner.run()

  # send_hwl: (P, P, count) uint32, column-major per bool_diag_spmv's own
  # host<->device convention (see device_io.hwl_to_oned_colmajor).
  send_1d = np.zeros(P * P * count, np.uint32)
  idx = 0
  for l in range(count):
    for w in range(P):
      for h in range(P):
        send_1d[idx] = send_hwl[h, w, l]
        idx += 1
  runner.memcpy_h2d(sym_send_buf, send_1d, 0, 0, P, P, count,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)

  runner.launch("f_run", nonblock=False)

  recv_1d = np.zeros(P * P * count, np.uint32)
  runner.memcpy_d2h(recv_1d, sym_recv_buf, 0, 0, P, P, count,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)
  runner.stop()

  recv_hwl = np.reshape(recv_1d, (P, P, count), order="F")
  return recv_hwl


def main():
  args = parse_args()
  P = args.num_pe_cols
  assert P == args.num_pe_rows, "reduce_or is a row-wise op here -- keep the test grid square"

  if args.root is None:
    # Orchestrator: one fresh subprocess per root position (the simulator
    # can't be instantiated twice in the same process -- see module
    # docstring). Deterministic seed means every subprocess independently
    # regenerates identical test data, so nothing needs passing between them.
    roots_to_test = sorted(set([0, P // 2, P - 1]))
    overall_pass = True
    for root in roots_to_test:
      print(f"\n=== testing root={root} (subprocess) ===")
      cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + [f"--root={root}"]
      import subprocess
      result = subprocess.run(cmd, check=False)
      if result.returncode != 0:
        overall_pass = False
        print(f"[[ root={root}: subprocess exited {result.returncode} ]]")
    if not args.compile_only:
      print(f"\n[[ Overall result: {'PASS' if overall_pass else 'FAIL'} ]]")
    sys.exit(0 if overall_pass else 1)

  cslc = args.driver or "cslc"
  count = args.count
  root = args.root

  code_csl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "src", "layout_reduce_or_test.csl")

  # random per-PE test data, one word per count -- exercises every bit
  # position across whatever [count]u32 words the OR-reduce combines, not
  # just a hand-picked one-hot pattern. Fixed seed: deterministic and
  # identical across the orchestrator's separate per-root subprocesses.
  rng = np.random.default_rng(0)
  send_hwl = rng.integers(0, 2**32, size=(P, P, count), dtype=np.uint64).astype(np.uint32)

  # expected: bitwise OR across each row (axis=1, the column/width axis) --
  # matches mpi_x's row-wise reduce convention (x-dimension = varies by
  # column, root is a column index within the row).
  expected = np.bitwise_or.reduce(send_hwl, axis=1)  # shape (P, count)

  dirname = f"{args.latestlink}_root{root}"
  recv_hwl = compile_and_run(cslc, code_csl, dirname, args, P, count, root, send_hwl)
  if recv_hwl is None:  # --compile-only
    return
  device_result = recv_hwl[:, root, :]  # (P, count) -- root column's copy, one per row
  n_mismatch = int(np.sum(device_result != expected))
  print(f"expected[{P}, {count}]:\n{expected}")
  print(f"device  [{P}, {count}]:\n{device_result}")
  print(f"[[ root={root}: mismatches: {n_mismatch} / {P * count} ]]")
  if n_mismatch != 0:
    idx = np.where(device_result != expected)
    print(f"mismatched (row, word) indices: {list(zip(*idx))}")
    sys.exit(1)


if __name__ == "__main__":
  main()
