""" Shared host<->device data-marshaling helpers for bool_diag_spmv's scripts
  (run_single_spmv.py, run_host_driven_bfs.py, run_bfs.py) -- the low-level
  hwl<->1d layout conversions, diagonal/parent result extraction, and the
  cslc invocation, none of which are specific to any one script's purpose.
"""

import subprocess
from typing import Optional

import numpy as np


def hwl_to_oned_colmajor(height: int, width: int, pe_length: int, A_hwl: np.ndarray, dtype):
  """
    Given a 3-D tensor A[height][width][pe_length], transform it to
    1D array by column-major
    """
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
          x = A_hwl[(h, w, l)]
          A_1d[idx] = np.uint32(x)
          idx = idx + 1
  else:
    raise RuntimeError(f"{A_hwl.dtype} is not supported")

  return A_1d


def oned_to_hwl_colmajor(height: int, width: int, pe_length: int, A_1d: np.ndarray, dtype):
  """
    Given a 1-D tensor A_1d[height*width*pe_length], transform it to
    3-D tensor A[height][width][pe_length] by column-major
    """
  assert dtype == np.float32, "only support f32 readback for this kernel"
  assert A_1d.dtype == np.float32, "only support f32 to f32"
  return np.reshape(A_1d, (height, width, pe_length), order="F")


# x is boolean, length n. Only the diagonal PE of each column (py == px)
# gets a real slice; every other PE starts at zero and receives the
# broadcast from phase 1. This replaces hypersparse_spmv's dist_x_to_hwl,
# which spread x across every PE in a column.
def dist_x_to_diag_hwl(n, x_bool, blk, P):
  x_pad = np.zeros(P * blk, dtype=np.float32)
  x_pad[0:n] = x_bool.astype(np.float32)

  x_hwl = np.zeros((P, P, blk), dtype=np.float32)
  for p in range(P):
    x_hwl[(p, p)] = x_pad[p * blk:(p + 1) * blk]
  return x_hwl


def single_source_seed_pe(source, blk, P):
  """For a genuinely single-source BFS seed (exactly one true bit, at
  `source`), return (px, py, local_x) for the ONE diagonal PE that owns
  it -- px=py=source//blk (diagonal: column==row), local_x a length-blk
  float32 array with a single 1.0 at local index source%blk. Pair with a
  1x1-region memcpy_h2d instead of dist_x_to_diag_hwl's full (P,P,blk)
  rectangle: every OTHER diagonal PE's x_buf is provably already zero
  (bool_pe.csl's reduce_done() sets x_buf[i] = newly every round including
  the last, and the loop's own termination condition, nz_total == 0, is a
  non-negative sum over every diagonal PE's own nz_local flag -- which is
  1.0 iff that PE's own x_buf had any nonzero entry -- so nz_total == 0
  provably means every diagonal PE's x_buf is all-zero at the moment
  f_spmv_iter() returns; combined with x_buf's zero state at kernel load,
  this holds for the very first search too), and non-diagonal PEs never
  need a host write at all regardless (every PE's x_buf is unconditionally
  overwritten by that round's own column-broadcast, in
  visited_bcast_done(), before compute() ever reads it).

  Only valid for this single-source, f_spmv_iter case -- NOT for
  run_host_driven_bfs.py's multi-source frontier (several diagonal PEs can
  be genuinely live at once there) or run_single_spmv.py's one-shot
  f_spmv (which never touches x_buf itself, so has no such self-zeroing
  invariant)."""
  p = source // blk
  local_x = np.zeros(blk, dtype=np.float32)
  local_x[source % blk] = 1.0
  return p, p, local_x


# Extract the diagonal PEs' y_bitmap_reduced (the only ones holding a
# meaningful final result) and reassemble into the length-n boolean output
# vector.
def extract_diag_result(n, blk, P, y_hwl):
  parts = [y_hwl[(p, p)] for p in range(P)]
  y_pad = np.concatenate(parts)
  return y_pad[0:n] > 0.0


def unpack_bitmap_to_dense(height, width, blk, bitmap_hwl):
  """bitmap_hwl: (height, width, bitmap_words) uint32, bit k of word (k>>5)
  at bit position (k&31) -- bool_pe.csl's y_bitmap/y_bitmap_reduced layout
  exactly (see its own declaration comment there). Returns a dense
  (height, width, blk) float32 array (0.0/1.0), matching the dtype
  extract_diag_result already assumes, so callers can feed the result
  straight into extract_diag_result unchanged."""
  dense = np.zeros((height, width, blk), dtype=np.float32)
  for k in range(blk):
    dense[:, :, k] = ((bitmap_hwl[:, :, k >> 5] >> (k & 31)) & 1).astype(np.float32)
  return dense


def extract_parent_result(n, blk, P, parent_hwl):
  """Assemble the length-n parent vector from the full (not diagonal-only)
  parent_local_buf rectangle. parent_hwl has shape (height=P, width=P, blk):
  for row-block p, every column-PE parent_hwl[p, :, :] independently
  computed a candidate parent for that row-block's blk local positions (see
  bool_pe.csl's module docstring) -- take the min across the P column-PEs
  (the row/column-min-reduce <collectives_2d> can't do, per the TODO there),
  same list-then-concatenate-then-truncate shape extract_diag_result above
  uses for the diagonal case."""
  parts = [parent_hwl[p, :, :].min(axis=0) for p in range(P)]
  parent = np.concatenate(parts).astype(np.int64)[0:n]
  parent[parent >= n] = -1  # normalize the device's PARENT_NONE (65535) sentinel
  return parent


def derive_visited_from_parent(n, parent, source):
  """visited[v] == True iff parent[v] >= 0 or v is the source itself.

  This is provably equivalent to reading back bool_pe.csl's visited_buf
  directly, not an approximation: compute() only ever records a parent
  candidate for row v in the SAME round v's row-reduce first turns
  visited_buf[v] nonzero (the visited_buf[dense_idx] == 0.0 gate in
  compute() -- see its own comment -- means every PE that contributes a hit
  to v's row-reduce in v's true discovery round also attempts to record a
  parent candidate that round), so parent_local_buf's row-min is non-
  PARENT_NONE exactly when v was ever discovered. The source is seeded
  directly into visited_buf in start_spmv(), never through compute(), so it
  never gets a parent recorded there -- callers already patch
  parent[source] = source after extract_parent_result(), which this
  function's `v is the source` check also covers.

  This is also Graph500's own convention: the reference implementation's
  Kernel 2 output is exactly the predecessor array (pred[v] = -1 for
  unreached, pred[root] = root) -- there's no separate "visited" output at
  all, so recovering it from parent instead of reading back visited_buf
  separately is not just an optimization, it's the more spec-faithful
  representation of "the output"."""
  visited = parent >= 0
  visited[source] = True
  return visited


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
    max_local_nnz_rows: int,
    channels: int,
    width_west_buf: int,
    width_east_buf: int,
    max_rounds: Optional[int] = None,
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
    args.append(f"--params=max_local_nnz_rows:{max_local_nnz_rows}")
    # left at layout_bool.csl's own default (32) unless a caller (see
    # run_bfs.py) needs per-round timing over a deeper BFS.
    if max_rounds is not None:
      args.append(f"--params=max_rounds:{max_rounds}")

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
