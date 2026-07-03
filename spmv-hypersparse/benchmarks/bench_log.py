"""Shared helper to persist benchmark timing results (with input parameters)
to a durable file instead of only printing to stdout, per project policy."""

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import cycles_to_us  # noqa: E402

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.jsonl")


def log_result(kernel, infile_mtx, n, nnz, P, cycles):
  record = {
      "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
      "kernel": kernel,
      "infile_mtx": infile_mtx,
      "n": n,
      "nnz": nnz,
      "pe_grid": f"{P}x{P}",
      "raw_cycles": cycles,
      "time_us": round(cycles_to_us(cycles), 3),
      "note": "memcpy excluded; raw min/max tsc, no cross-PE clock-skew correction",
  }
  with open(RESULTS_FILE, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
  print(f"[bench_log] appended to {RESULTS_FILE}: {record}")
  return record
