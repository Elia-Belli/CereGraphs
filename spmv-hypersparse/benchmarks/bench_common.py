"""Shared, kernel-agnostic helpers for the bench_*_timing.py scripts.

Path resolution is anchored to this file's location (not the invocation cwd)
so the benchmark scripts work regardless of where they're launched from.
"""

import os

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BENCH_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
ORIGINAL_SPMV_DIR = os.path.join(REPO_ROOT, "original_spmv")
BOOL_DIAG_SPMV_DIR = os.path.join(REPO_ROOT, "bool_diag_spmv")

CLOCK_GHZ = 0.85  # 850MHz, matches hypersparse_spmv/run.py's cycles->us conversion


def cycles_to_us(cycles):
  return (cycles / CLOCK_GHZ) * 1e-3


def make_u48(words):
  return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


def fabric_dims(width, height):
  fabric_offset_x = 1
  fabric_offset_y = 1
  core_x = fabric_offset_x + 3
  core_y = fabric_offset_y
  min_w = core_x + width + 2 + 1
  min_h = core_y + height + 1
  return min_w, min_h, core_x, core_y


def hwl_to_oned_colmajor(height, width, pe_length, A_hwl, dtype):
  """Given a 3-D tensor A[height][width][pe_length], transform it to a 1D
  array by column-major order (matches MemcpyOrder.COL_MAJOR)."""
  import numpy as np
  if A_hwl.dtype == np.float32:
    A_1d = np.zeros(height * width * pe_length, dtype)
    idx = 0
    for l in range(pe_length):
      for w in range(width):
        for h in range(height):
          A_1d[idx] = A_hwl[(h, w, l)]
          idx += 1
  elif A_hwl.dtype == np.uint16:
    assert dtype == np.uint32, "only support dtype = u32 if A is u16"
    A_1d = np.zeros(height * width * pe_length, dtype)
    idx = 0
    for l in range(pe_length):
      for w in range(width):
        for h in range(height):
          A_1d[idx] = np.uint32(A_hwl[(h, w, l)])
          idx += 1
  else:
    raise RuntimeError(f"{A_hwl.dtype} is not supported")
  return A_1d


def oned_to_hwl_colmajor(height, width, pe_length, A_1d, dtype):
  """Inverse of hwl_to_oned_colmajor."""
  import numpy as np
  assert dtype == np.float32, "only support f32 readback"
  assert A_1d.dtype == np.float32, "only support f32 to f32"
  return np.reshape(A_1d, (height, width, pe_length), order="F")
