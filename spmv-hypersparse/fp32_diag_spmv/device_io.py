""" Shared host<->device data-marshaling helpers for fp32_diag_spmv's scripts --
  forked from bool_diag_spmv/device_io.py, dropping the bitmap-packing helpers
  (this kernel uses dense f32 buffers directly, no boolean bitmap packing)
  and adapting csl_compile_core to fp32_pe.csl/layout_fp32.csl's simpler
  param list (no max_local_nnz_rows, no BFS-only max_rounds/tau_switch_count).
"""

import subprocess
from typing import Optional

import numpy as np


def hwl_to_oned_colmajor(height: int, width: int, pe_length: int, A_hwl: np.ndarray, dtype):
  """Given a 3-D tensor A[height][width][pe_length], transform it to a 1D
  array by column-major order."""
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
          A_1d[idx] = np.uint32(A_hwl[(h, w, l)])
          idx = idx + 1
  else:
    raise RuntimeError(f"{A_hwl.dtype} is not supported")

  return A_1d


def oned_to_hwl_colmajor(height: int, width: int, pe_length: int, A_1d: np.ndarray, dtype):
  """Given a 1-D tensor A_1d[height*width*pe_length], transform it to a 3-D
  tensor A[height][width][pe_length] by column-major order."""
  assert dtype == np.float32, "only support f32 readback for this kernel"
  assert A_1d.dtype == np.float32, "only support f32 to f32"
  return np.reshape(A_1d, (height, width, pe_length), order="F")


def dist_x_to_diag_hwl(n, x, blk, P):
  """x is real-valued, length n. Only the diagonal PE of each column
  (py == px) gets a real slice; every other PE starts at zero and receives
  the broadcast from phase 1."""
  x_pad = np.zeros(P * blk, dtype=np.float32)
  x_pad[0:n] = x.astype(np.float32)

  x_hwl = np.zeros((P, P, blk), dtype=np.float32)
  for p in range(P):
    x_hwl[(p, p)] = x_pad[p * blk:(p + 1) * blk]
  return x_hwl


def extract_diag_result(n, blk, P, y_hwl):
  """Extract the diagonal PEs' y_buf (the only ones holding a meaningful
  final result) and reassemble into the length-n real output vector."""
  parts = [y_hwl[(p, p)] for p in range(P)]
  y_pad = np.concatenate(parts)
  return y_pad[0:n]


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
    channels: int,
    width_west_buf: int,
    width_east_buf: int,
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
