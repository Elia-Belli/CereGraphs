#!/usr/bin/env python3
"""Orchestrator for the device-only-vs-SDK-host-driven timing comparison
(Workstream B of the poster benchmark plan): for each (matrix, grid) test
case in TEST_CASES below, runs bench_sdk_iters_timing.py and
bench_device_iters_timing.py as SEPARATE subprocesses (the simulator can't
be instantiated twice in one process, per bench_common.py's own convention),
parses each one's H2D_CYCLES/COMPUTE_CYCLES/D2H_CYCLES output, and appends
one CSV row per (test case, version) to results/device_vs_host_timing.csv.

Edit TEST_CASES below to whatever (matrix, grid) pairs you actually want to
run -- the list here is just a small smoke-test default. Run this from the
repo root (not from inside benchmarks/) so the container's cwd-only bind
mount can see the matrix files and the sibling spmv/ directories:

  cs_python benchmarks/bench_device_vs_host.py [--driver=cslc] [--arch=wse2] [--iters=10]

See bench_sdk_iters_timing.py's module docstring for the h2d/d2h
simulator-vs-appliance caveat -- this orchestrator just plumbs whatever
those two scripts report, unmodified.
"""

import argparse
import csv
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                           "device_vs_host_timing.csv")

# Each entry: (matrix path relative to repo root, PE grid size P for a PxP square grid).
# Edit freely -- these are just small smoke-test defaults.
TEST_CASES = [
    ("data/rmat_s6_e4.mtx", 8),
    ("data/rmat_s8_e4.mtx", 8),
]

CYCLE_RE = re.compile(r"^(H2D|COMPUTE|D2H)_CYCLES=(\d+)$")


def run_one(script, infile_mtx, grid, driver, arch, iters, out_dir):
  args = [
      "cs_python", script,
      f"--infile_mtx={infile_mtx}",
      f"--num_pe_cols={grid}",
      f"--num_pe_rows={grid}",
      f"--driver={driver}",
      f"--arch={arch}",
      f"--iters={iters}",
      f"--out_dir={out_dir}",
  ]
  print(f"$ {' '.join(args)}", flush=True)
  proc = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
  print(proc.stdout)
  if proc.returncode != 0:
    print(proc.stderr, file=sys.stderr)
    raise RuntimeError(f"{script} failed on {infile_mtx} (grid {grid}x{grid})")

  cycles = {}
  for line in proc.stdout.splitlines():
    m = CYCLE_RE.match(line.strip())
    if m:
      cycles[m.group(1).lower() + "_cycles"] = int(m.group(2))
  for key in ("h2d_cycles", "compute_cycles", "d2h_cycles"):
    if key not in cycles:
      raise RuntimeError(f"{script} did not print {key.upper()} for {infile_mtx}")
  return cycles


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--driver", default="cslc")
  p.add_argument("--arch", default="wse2")
  p.add_argument("--iters", type=int, default=10)
  p.add_argument("--out_root", default="/tmp/bench_device_vs_host")
  args = p.parse_args()

  os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
  write_header = not os.path.exists(RESULTS_CSV)

  with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if write_header:
      writer.writerow(["matrix", "pe_grid", "version", "h2d_cycles", "compute_cycles",
                        "d2h_cycles", "total_cycles"])

    for infile_mtx, grid in TEST_CASES:
      stem = os.path.splitext(os.path.basename(infile_mtx))[0]
      pe_grid = f"{grid}x{grid}"
      print(f"\n=== {infile_mtx} @ {pe_grid} ===")

      sdk_out = os.path.join(args.out_root, f"{stem}_{pe_grid}", "sdk")
      sdk_cycles = run_one("benchmarks/bench_sdk_iters_timing.py", infile_mtx, grid,
                            args.driver, args.arch, args.iters, sdk_out)
      sdk_total = sdk_cycles["h2d_cycles"] + sdk_cycles["compute_cycles"] + sdk_cycles["d2h_cycles"]
      writer.writerow([infile_mtx, pe_grid, "sdk-hypersparse-spmv (host-driven)",
                        sdk_cycles["h2d_cycles"], sdk_cycles["compute_cycles"],
                        sdk_cycles["d2h_cycles"], sdk_total])
      f.flush()

      device_out = os.path.join(args.out_root, f"{stem}_{pe_grid}", "device")
      device_cycles = run_one("benchmarks/bench_device_iters_timing.py", infile_mtx, grid,
                               args.driver, args.arch, args.iters, device_out)
      device_total = (device_cycles["h2d_cycles"] + device_cycles["compute_cycles"] +
                       device_cycles["d2h_cycles"])
      writer.writerow([infile_mtx, pe_grid, "fp32_diag_spmv (device-only)",
                        device_cycles["h2d_cycles"], device_cycles["compute_cycles"],
                        device_cycles["d2h_cycles"], device_total])
      f.flush()

  print(f"\nResults appended to {RESULTS_CSV}")


if __name__ == "__main__":
  main()
