#!/usr/bin/env cs_python
""" Standalone, isolated correctness test for collectives_2d/pe.csl's
  reduce_select_any() and its explicit-index counterpart,
  reduce_select_any_indexed() -- see docs/ERRORS.md #16/#17 for the real
  d2h gRPC message-size ceiling the dense collective originally fixed, and
  for the design history (a first sparse bitmap+compact-values attempt was
  built, measured ~7.6-8x SLOWER on real hardware, and has since been
  removed entirely -- #16). No matrix/BFS logic at all: every PE seeds a
  [count]u32 buffer (a controlled mix of all-sentinel rows,
  exactly-one-real-value rows, and -- the actual point of this collective,
  since a naive OR-style reduce would silently corrupt this case -- rows
  where MULTIPLE PEs hold DIFFERENT real values), one mpi_x.
  reduce_select_any()/reduce_select_any_indexed() call per compile
  resolves each row toward a compile-time `root` column, and the result is
  checked against the correctness property: non-sentinel iff the row had
  at least one real candidate, and whenever non-sentinel, a genuine MEMBER
  of that row's set of real candidates -- never a specific expected value
  (which one wins among several valid candidates is deliberately
  unspecified, same tie-break-is-not-a-bug convention this repo already
  uses for BFS parent selection) and never a corrupted bitwise mix.

  --variant=dense (default) tests reduce_select_any(); --variant=indexed
  tests reduce_select_any_indexed() -- same ground truth, same checking
  logic, only the CSL kernel compiled differs (pe_reduce_select_test.csl
  vs pe_reduce_select_indexed_test.csl), since the two collectives share
  an identical caller-visible dense [count]u32/sentinel contract.

  --occupancy controls what fraction of (row, word) cells get a real value
  at all (the rest are all-sentinel) -- low occupancy is the regime
  indexed's O(popcount) merge is actually for; high/near-dense occupancy
  is the regime where its bandwidth win shrinks toward zero (still must
  stay CORRECT there, just not necessarily cheaper). Real hardware
  measured <1.2% mean occupancy at RMAT s17/s19 scale -- --occupancy=0.01
  (or lower) reproduces that regime; --occupancy=0.95 stresses the
  near-dense regime.

  Tests one root position per process (the simulator backend can't be
  instantiated twice in the same process -- same constraint
  run_reduce_or_test.py documents). With no --root given, re-invokes
  itself as a subprocess once per root position (0, middle, NUM_PES-1).

  How to run
     cs_python run_reduce_select_test.py --arch=wse3 --num_pe_cols=4 --num_pe_rows=4 \
        --driver=<path to cslc> --count=8
     cs_python run_reduce_select_test.py --arch=wse3 --num_pe_cols=2 --num_pe_rows=2 \
        --count=1 --root=0   # single root position, one process, smallest P
     cs_python run_reduce_select_test.py --arch=wse3 --num_pe_cols=8 --num_pe_rows=8 \
        --count=64 --variant=indexed --occupancy=0.01   # low-occupancy regime
"""

import argparse
import os
import sys
import time

import numpy as np

SENTINEL = 4294967295  # must match bool_pe.csl's PARENT_NONE / device_io.py's PARENT_NONE_GLOBAL


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_pe_cols", type=int, required=True, help="width of the core rectangle")
  parser.add_argument("--num_pe_rows", type=int, required=True, help="height of the core rectangle")
  parser.add_argument("--count", type=int, default=8,
                       help="number of u32 words per PE")
  parser.add_argument("--root", type=int, default=None,
                       help="single root column to test (default: re-invoke self once per "
                            "root in {0, P//2, P-1}, one subprocess each)")
  parser.add_argument("--variant", choices=["dense", "indexed"], default="dense",
                       help="dense = reduce_select_any(), indexed = reduce_select_any_indexed() "
                            "(explicit row-index+value pairs instead of position-implicit "
                            "compaction; measured 4.5-8.9x FASTER than dense on real hardware "
                            "at grids where it compiles, see docs/ERRORS.md #17). A first "
                            "sparse bitmap+compact-values attempt was measured ~7.6-8x SLOWER "
                            "and has since been removed entirely (#16).")
  parser.add_argument("--occupancy", type=float, default=2.0 / 3.0,
                       help="fraction of (row, word) cells assigned a real value; the rest "
                            "stay all-sentinel. Default 2/3 matches this test's original "
                            "fixed category-0/1/2 cycle. Use a low value (e.g. 0.01) to "
                            "reproduce the low-occupancy regime real hardware measured at "
                            "RMAT s17/s19 scale, or a high value (e.g. 0.95) to stress the "
                            "near-dense regime where indexed's bandwidth win shrinks toward "
                            "zero.")
  parser.add_argument("--fabric-dims", help="Fabric dimension, i.e. <W>,<H>")
  parser.add_argument("--compile-only", action="store_true")
  parser.add_argument("--run-only", action="store_true")
  parser.add_argument("--width-west-buf", default=0, type=int)
  parser.add_argument("--width-east-buf", default=0, type=int)
  parser.add_argument("--channels", default=1, type=int)
  parser.add_argument("-d", "--driver", help="path to the CSL compiler")
  parser.add_argument("--cmaddr", help="CM address and port, i.e. <IP>:<port>")
  parser.add_argument("--arch", help="wse2 or wse3")
  parser.add_argument("--latestlink", default="out/reduce_select_test",
                       help="folder to contain the log files")
  return parser.parse_args()


def make_test_data(P, count, seed=0, occupancy=2.0 / 3.0):
  """send_hwl: (P, P, count) uint32, SENTINEL everywhere except a
  deliberately-mixed set of row categories at each (h, l) position:
    category 0: all-sentinel (no candidate at all)
    category 1: exactly one column has a real value
    category 2: every column has a DIFFERENT real value -- the actual
      collision case reduce_or's fused OR-combine cannot handle correctly
      (see this module's own docstring); with P<2 this degenerates to
      category 1, still valid but not exercising the real collision path.
  `occupancy` is the fraction of (h, l) cells assigned category 1 or 2
  (split evenly between them); the rest stay category 0. Low occupancy
  reproduces the regime reduce_select_any_indexed()'s O(popcount) merge is
  actually for (real hardware measured <1.2% mean occupancy at RMAT
  s17/s19 scale); high occupancy stresses the regime where its bandwidth
  win shrinks toward zero -- both must stay correct, so both are worth
  testing.
  Returns (send_hwl, real_sets) where real_sets[h][l] is the python set of
  real (non-sentinel) values placed in row h, word l -- the ground truth
  the device result is checked against (membership, not equality)."""
  rng = np.random.default_rng(seed)
  send_hwl = np.full((P, P, count), SENTINEL, dtype=np.uint32)
  real_sets = [[set() for _ in range(count)] for _ in range(P)]

  def real_value():
    v = int(rng.integers(0, 2**32 - 1, dtype=np.uint64))
    return v + 1 if v == SENTINEL else v  # never emit the sentinel itself

  for h in range(P):
    for l in range(count):
      if rng.random() >= occupancy:
        continue  # category 0: stays all-sentinel
      # deterministic (given the occupancy draw above already consumed
      # randomness) alternation between category 1 and 2, same 50/50 split
      # the original fixed 3-way cycle gave the two real categories.
      category = 1 if (h * count + l) % 2 == 0 else 2
      if category == 1 or P < 2:
        w0 = int(rng.integers(0, P))
        v = real_value()
        send_hwl[h, w0, l] = v
        real_sets[h][l].add(v)
      else:  # category == 2: every column gets its own distinct real value
        seen = set()
        for w in range(P):
          v = real_value()
          while v in seen:
            v = real_value()
          seen.add(v)
          send_hwl[h, w, l] = v
        real_sets[h][l] = seen

  return send_hwl, real_sets


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
        f"--params=sentinel:{SENTINEL}",
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

  from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
      MemcpyDataType, MemcpyOrder, SdkRuntime,
  )

  runner = SdkRuntime(dirname, cmaddr=args.cmaddr, suppress_simfab_trace=True)
  sym_send_buf = runner.get_id("send_buf")
  sym_recv_buf = runner.get_id("recv_buf")

  runner.load()
  runner.run()

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
  assert P == args.num_pe_rows, "reduce_select_any is a row-wise op here -- keep the test grid square"

  if args.root is None:
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

  layout_name = {
      "dense": "layout_reduce_select_test.csl",
      "indexed": "layout_reduce_select_indexed_test.csl",
  }[args.variant]
  code_csl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "src", layout_name)

  send_hwl, real_sets = make_test_data(P, count, seed=0, occupancy=args.occupancy)

  # Host-side evidence of how sparse this run's data actually is -- the
  # ground truth already fully determines per-PE occupancy, so no extra
  # device-side byte-count instrumentation is needed to see whether a test
  # run is actually exercising the low-occupancy regime
  # reduce_select_any_indexed() is for, versus the near-dense regime where
  # its bandwidth win (not its correctness) shrinks toward zero.
  real_count_total = sum(1 for h in range(P) for l in range(count)
                          for w in range(P) if send_hwl[h, w, l] != SENTINEL)
  measured_occupancy = real_count_total / (P * P * count)
  print(f"[[ root={root}: variant={args.variant} requested_occupancy={args.occupancy:.4f} "
        f"measured_occupancy={measured_occupancy:.4f} ]]")

  dirname = f"{args.latestlink}_{args.variant}_root{root}"
  recv_hwl = compile_and_run(cslc, code_csl, dirname, args, P, count, root, send_hwl)
  if recv_hwl is None:  # --compile-only
    return
  device_result = recv_hwl[:, root, :]  # (P, count) -- root column's copy, one per row

  n_mismatch = 0
  n_collision_rows_checked = 0
  bad = []
  for h in range(P):
    for l in range(count):
      expected_set = real_sets[h][l]
      got = int(device_result[h, l])
      if len(expected_set) == 0:
        ok = (got == SENTINEL)
      else:
        ok = (got in expected_set)
        if len(expected_set) > 1:
          n_collision_rows_checked += 1
      if not ok:
        n_mismatch += 1
        bad.append((h, l, got, sorted(expected_set)))

  print(f"[[ root={root}: variant={args.variant} P={P} count={count} "
        f"collision-rows-checked={n_collision_rows_checked} ]]")
  print(f"[[ root={root}: mismatches: {n_mismatch} / {P * count} ]]")
  if n_mismatch != 0:
    print(f"mismatched (row, word, device_got, expected_set): {bad[:20]}"
          + (" ... (truncated)" if len(bad) > 20 else ""))
    sys.exit(1)


if __name__ == "__main__":
  main()
