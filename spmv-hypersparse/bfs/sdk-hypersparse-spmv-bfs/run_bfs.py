#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args
""" host-orchestrated BFS on top of the (unmodified) hypersparse SpMV kernel

  This is deliberately the "naive" way to build BFS out of an SpMV primitive:
  the device kernel only ever computes one y = A*x per launch, and every
  piece of BFS state (visited set, parent pointers, frontier masking,
  termination) lives on the host. Each hop costs one full host<->device
  round trip (repack x, memcpy_h2d, launch, memcpy_d2h, repack y) on top of
  the device compute itself -- that overhead, plus the two limitations
  documented below, is the point: this version exists to be compared against
  a smarter, fabric-resident BFS later. See README.md for the full writeup.

  Design notes (see README.md for the long version):

  1. Boolean semiring via real arithmetic. The kernel only knows how to
     compute a real-valued y = A*x (multiply-add). We never touch the CSL
     source: A's structural nonzeros carry value 1.0 and the frontier is
     0.0/1.0, so y[i] becomes the *count* of frontier predecessors that
     reach i -- still exactly zero iff i is unreached this hop. The host
     thresholds y > 0 to recover the boolean OR a real boolean semiring
     would have computed directly. Wasteful (a float multiply-add per edge
     instead of one bit), but it needs zero kernel changes.

  2. Directed graphs via a free transpose. The kernel computes y = A*x
     (row i dotted with x). Under the standard adjacency convention used
     here -- A[i, j] != 0 means a directed edge i -> j (row = source,
     col = destination) -- one hop of forward BFS expansion is
     y = A^T @ frontier, not A @ frontier. Transposing a CSR gives back a
     CSC of the same matrix (and vice versa), so this costs nothing: we
     just feed preprocess() A's CSC as its "CSR" argument and A's CSR as
     its "CSC" argument. For an undirected/symmetric matrix A == A^T, so
     this is a no-op relabeling -- the directed and undirected cases are
     the same code path.

  3. Parent reconstruction is a host-side afterthought, not a kernel
     capability. OR-reduction tells you a node became reachable, not which
     frontier member reached it first -- that needs a "select a witness"
     semiring (e.g. min-index), which this AND-OR kernel doesn't implement.
     So for every newly-discovered node we separately scan its incoming
     edges (host-side, against the *original* A, not the transposed one fed
     to the device) and pick whichever predecessor happens to be in the
     previous frontier. This is a real limitation of bolting BFS onto a
     bare boolean SpMV: correct visited/level information falls out for
     free, but parent pointers require this separate host-side pass.

  Ping-pong requirement (see the comment on `y_local_buf` in
  hypersparse_spmv/pe.csl): local_vec_sz must equal local_out_vec_sz for the
  kernel to compile with an eye toward feeding y back as the next x. Since A
  is square (nrows == ncols) and we require a square PE grid (pcols ==
  prows), local_vec_sz == local_out_vec_sz automatically. NOTE this does NOT
  mean y's on-device layout can be copied directly back into x_tx_buf: x is
  distributed column-block-then-row-block while y is distributed
  row-block-then-column-block (see dist_x_to_hwl / unpad_3d_to_1d below), so
  every hop still needs a full host-side repack -- which is exactly where
  the visited-mask also gets applied, so it isn't wasted work.

  How to compile and run
     python run_bfs.py --arch=wse2 --num_pe_cols=4 --num_pe_rows=4 --channels=1
        --driver=<path to cslc> --infile_mtx=<path to mtx file> --source=0
"""

import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from cmd_parser import parse_args
from memory_usage import memory_per_pe
from preprocess import preprocess
from scipy.io import mmread
from scipy.sparse.csgraph import breadth_first_order

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bfs_results.jsonl")


def hwl_to_oned_colmajor(height: int, width: int, pe_length: int, A_hwl: np.ndarray, dtype):
  """Given a 3-D tensor A[height][width][pe_length], transform it to
  1D array by column-major (matches MemcpyOrder.COL_MAJOR)."""
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


def oned_to_hwl_colmajor(height: int, width: int, pe_length: int, A_1d: np.ndarray, dtype):
  """Inverse of hwl_to_oned_colmajor. Only the f32 direction is needed here
  (y readback); matrix structure upload is h2d-only."""
  assert dtype == np.float32, "only support f32 readback for this kernel"
  assert A_1d.dtype == np.float32, "only support f32 to f32"
  return np.reshape(A_1d, (height, width, pe_length), order="F")


# Same distribution sdk-hypersparse-spmv/run.py uses for x: columns first, then rows
# within a column. See that file for the worked example.
def dist_x_to_hwl(ncols, x, local_vec_sz, np_cols, np_rows):
  vec_len = ncols
  vec_len_per_pe_col = math.ceil(vec_len / np_cols)
  vec_len_per_pe = math.ceil(vec_len_per_pe_col / np_rows)
  assert vec_len_per_pe == local_vec_sz

  pad_len_per_pe_col = (vec_len_per_pe * np_rows) - vec_len_per_pe_col
  pad_len = (vec_len_per_pe_col * np_cols) - vec_len
  invec = np.copy(x)
  if pad_len > 0:
    invec = np.append(invec, np.zeros(pad_len, dtype=x.dtype))

  x_hwl = np.zeros((np_rows, np_cols, vec_len_per_pe), x.dtype)
  for col in range(np_cols):
    invec_col = invec[col * vec_len_per_pe_col:(col + 1) * vec_len_per_pe_col]
    if pad_len_per_pe_col > 0:
      invec_col = np.append(invec_col, np.zeros(pad_len_per_pe_col, dtype=x.dtype))
    for row in range(np_rows):
      data = invec_col[row * vec_len_per_pe:(row + 1) * vec_len_per_pe]
      x_hwl[(row, col)] = data

  return x_hwl


# Same distribution sdk-hypersparse-spmv/run.py uses for y: rows first, then
# columns within a row -- deliberately NOT the same layout as x above, which
# is why ping-ponging y back into x still requires a host round trip.
def unpad_3d_to_1d(out_vec_sz, out_vec):
  assert out_vec.ndim == 3, "y must be a 3-d tensor of the form h-by-w-by-l"
  (height, width, local_out_vec_sz) = out_vec.shape
  np_rows = height
  np_cols = width

  vec_len_per_pe_row = math.ceil(out_vec_sz / np_rows)
  vec_len_per_pe = math.ceil(vec_len_per_pe_row / np_cols)
  assert vec_len_per_pe == local_out_vec_sz

  result = np.zeros(vec_len_per_pe_row * np_rows, dtype=np.float32)
  tmp_buf = np.empty(vec_len_per_pe * np_cols, dtype=np.float32)
  for row in range(np_rows):
    low_idx = row * vec_len_per_pe_row
    high_idx = low_idx + vec_len_per_pe_row
    for col in range(np_cols):
      start = col * vec_len_per_pe
      end = start + vec_len_per_pe
      tmp_buf[start:end] = out_vec[(row, col)]
    result[low_idx:high_idx] = tmp_buf[0:vec_len_per_pe_row]
  return result


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
    ncols: int,
    nrows: int,
    np_cols: int,
    np_rows: int,
    max_local_nnz: int,
    max_local_nnz_cols: int,
    max_local_nnz_rows: int,
    local_vec_sz: int,
    local_out_vec_sz: int,
    out_pad_start_idx: int,
    channels: int,
    width_west_buf: int,
    width_east_buf: int,
):
  if not use_precompile:
    args = [
        cslc,
        file_config,
        f"--fabric-dims={fabric_width},{fabric_height}",
        f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}",
        f"--params=ncols:{ncols}",
        f"--params=nrows:{nrows}",
        f"--params=pcols:{np_cols}",
        f"--params=prows:{np_rows}",
        f"--params=max_local_nnz:{max_local_nnz}",
        f"--params=max_local_nnz_cols:{max_local_nnz_cols}",
        f"--params=max_local_nnz_rows:{max_local_nnz_rows}",
        f"--params=local_vec_sz:{local_vec_sz}",
        f"--params=local_out_vec_sz:{local_out_vec_sz}",
        f"--params=y_pad_start_row_idx:{out_pad_start_idx}",
        f"-o={elf_dir}",
    ]
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


def find_parents(A_csc_indptr, A_csc_indices, new_nodes, frontier_set, parent):
  """For each newly-discovered node v, pick whichever of its incoming-edge
  predecessors (A_csc column v, under the row=src/col=dst convention) is in
  the previous frontier. Any such predecessor is a valid BFS parent (all
  frontier members are at the same, correct, distance) -- see module
  docstring point 3 for why this can't come from the device kernel."""
  for v in new_nodes:
    start, end = A_csc_indptr[v], A_csc_indptr[v + 1]
    for u in A_csc_indices[start:end]:
      if u in frontier_set:
        parent[v] = int(u)
        break


def verify_bfs(A_csr, n, source, visited, parent):
  print("Comparing BFS result with scipy reference...")
  order = breadth_first_order(A_csr, source, directed=True, return_predecessors=False)
  ref_visited = np.zeros(n, dtype=bool)
  ref_visited[order] = True

  n_mismatch = int(np.sum(ref_visited != visited))
  print(f"[[ visited-set mismatches vs reference: {n_mismatch} / {n} ]]")

  # Parent pointers are not unique (several frontier members may reach the
  # same node in the same hop), so we check *validity* rather than an exact
  # match against scipy's own (equally arbitrary) parent choice: every
  # non-source visited node's parent must be visited and must be a real
  # predecessor edge in A.
  bad_parents = []
  for v in range(n):
    if v == source or not visited[v]:
      continue
    u = parent[v]
    if u < 0 or not visited[u] or A_csr[u, v] == 0:
      bad_parents.append(v)
  print(f"[[ invalid parent pointers: {len(bad_parents)} / {int(np.sum(visited)) - 1} ]]")
  if bad_parents:
    shown = bad_parents[:20]
    print(f"nodes with invalid parent: {shown}{' ...' if len(bad_parents) > 20 else ''}")

  passed = (n_mismatch == 0) and (not bad_parents)
  print(f"[[ Result: {'PASS' if passed else 'FAIL'} ]]")
  return passed, n_mismatch, len(bad_parents)


def log_run(record):
  with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
  print(f"[run_bfs] appended run record to {LOG_FILE}")


def main():
  """Main method to run the example code."""
  args = parse_args()

  cslc = "cslc"
  if args.driver is not None:
    cslc = args.driver

  width_west_buf = args.width_west_buf
  width_east_buf = args.width_east_buf
  channels = args.channels
  assert channels <= 16, "only support up to 16 I/O channels"
  assert channels >= 1, "number of I/O channels must be at least 1"

  dirname = args.latestlink

  np_cols = args.num_pe_cols
  np_rows = args.num_pe_rows
  assert np_cols == np_rows, "square PE grid required (see module docstring: ping-pong needs local_vec_sz == local_out_vec_sz)"
  assert np_rows >= 4, "kernel requires prows >= 4 (see hypersparse_spmv/layout.csl)"

  width = np_cols
  height = np_rows

  infile_mtx = args.infile_mtx
  print(f"infile_mtx = {infile_mtx}")

  A_coo = mmread(infile_mtx)
  A_csr = A_coo.tocsr(copy=True).astype(np.float32)
  A_csr.sum_duplicates()
  A_csr.data[:] = 1.0  # structural/boolean adjacency only, no edge weights
  A_csr = A_csr.sorted_indices()
  assert A_csr.has_sorted_indices == 1, "Error: A is not sorted"

  [nrows, ncols] = A_csr.shape
  assert nrows == ncols, "BFS requires a square adjacency matrix"
  n = nrows
  nnz = A_csr.nnz
  source = args.source
  assert 0 <= source < n, f"--source={source} out of range [0, {n})"

  print(f"Load matrix A, {nrows}-by-{ncols} with {nnz} nonzeros (structural, boolean)")

  A_csc = A_csr.tocsc(copy=True)
  A_csc = A_csc.sorted_indices()
  assert A_csc.has_sorted_indices == 1, "Error: A is not sorted"

  # Feed preprocess() A^T's structure (see module docstring point 2): a
  # transpose of a CSR is a CSC of the same matrix and vice versa, so this
  # is just an argument swap, not a matrix rebuild.
  start = time.time()
  matrix_info = preprocess(
      nrows,
      ncols,
      nnz,
      np_cols,
      np_rows,
      A_csc.indptr,
      A_csc.indices,
      A_csr.indptr,
      A_csr.indices,
      A_csr.data,
  )
  end = time.time()
  print(f"prepare the structure for spmv kernel: {end-start}s", flush=True)

  max_local_nnz = matrix_info["max_local_nnz"]
  max_local_nnz_cols = matrix_info["max_local_nnz_cols"]
  max_local_nnz_rows = matrix_info["max_local_nnz_rows"]
  mat_vals_buf = matrix_info["mat_vals_buf"]
  mat_rows_buf = matrix_info["mat_rows_buf"]
  mat_col_idx_buf = matrix_info["mat_col_idx_buf"]
  mat_col_loc_buf = matrix_info["mat_col_loc_buf"]
  mat_col_len_buf = matrix_info["mat_col_len_buf"]
  y_rows_init_buf = matrix_info["y_rows_init_buf"]
  local_nnz = matrix_info["local_nnz"]
  local_nnz_cols = matrix_info["local_nnz_cols"]
  local_nnz_rows = matrix_info["local_nnz_rows"]

  # square matrix + square grid => local_vec_sz == local_out_vec_sz (see
  # module docstring on the ping-pong requirement)
  local_vec_sz = math.ceil(math.ceil(ncols / np_cols) / np_rows)
  local_out_vec_sz = math.ceil(math.ceil(nrows / np_rows) / np_cols)
  assert local_vec_sz == local_out_vec_sz

  mem_use_per_pe = memory_per_pe(
      max_local_nnz,
      max_local_nnz_cols,
      max_local_nnz_rows,
      local_vec_sz,
      local_out_vec_sz,
  )
  print(f"Total memory use per PE = {mem_use_per_pe} bytes = {mem_use_per_pe / 1024} KB",
        flush=True)
  assert (mem_use_per_pe < 46 * 1024), "exceed maximum memory capacity, increase the core rectangle"

  fabric_offset_x = 1
  fabric_offset_y = 1
  core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
  core_fabric_offset_y = fabric_offset_y
  min_fabric_width = core_fabric_offset_x + width + 2 + 1 + width_east_buf
  min_fabric_height = core_fabric_offset_y + height + 1

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

  print(f"fabric_width = {fabric_width}, fabric_height = {fabric_height}")
  print("store ELFs and log files in the folder ", dirname)

  # NOTE: absolute, anchored to this file's own location -- see the matching
  # comment in sdk-hypersparse-spmv/run.py for why (container bind-mount only
  # covers the invocation cwd).
  #
  # src_wse3/ carries WSE-3-specific kernel changes (queue remapping, no
  # allreduce2R1E-based sync -- see ../sdk-hypersparse-spmv/README.md and
  # github.com/Cerebras/sdk-examples/pull/23) that are NOT backwards
  # compatible with WSE-2, so it's a separate source tree rather than a
  # conditional inside src/.
  src_dir = "src_wse3" if args.arch == "wse3" else "src"
  code_csl = os.path.join(os.path.dirname(os.path.abspath(__file__)), src_dir, "layout.csl")

  out_vec_len_per_pe_row = math.ceil(nrows / np_rows)
  out_pad_start_idx = out_vec_len_per_pe_row

  start = time.time()
  csl_compile_core(
      cslc,
      code_csl,
      dirname,
      fabric_width,
      fabric_height,
      core_fabric_offset_x,
      core_fabric_offset_y,
      args.run_only,
      args.arch,
      ncols,
      nrows,
      np_cols,
      np_rows,
      max_local_nnz,
      max_local_nnz_cols,
      max_local_nnz_rows,
      local_vec_sz,
      local_out_vec_sz,
      out_pad_start_idx,
      channels,
      width_west_buf,
      width_east_buf,
  )
  end = time.time()
  compile_time = end - start
  print(f"Compilation done in {compile_time}s", flush=True)

  if args.compile_only:
    print("COMPILE ONLY: EXIT")
    return

  runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

  sym_mat_vals_buf = runner.get_id("mat_vals_buf")
  sym_x_tx_buf = runner.get_id("x_tx_buf")
  sym_y_local_buf = runner.get_id("y_local_buf")
  sym_mat_rows_buf = runner.get_id("mat_rows_buf")
  sym_mat_col_idx_buf = runner.get_id("mat_col_idx_buf")
  sym_mat_col_loc_buf = runner.get_id("mat_col_loc_buf")
  sym_mat_col_len_buf = runner.get_id("mat_col_len_buf")
  sym_y_rows_init_buf = runner.get_id("y_rows_init_buf")
  sym_local_nnz = runner.get_id("local_nnz")
  sym_local_nnz_cols = runner.get_id("local_nnz_cols")
  sym_local_nnz_rows = runner.get_id("local_nnz_rows")

  start = time.time()
  runner.load()
  end = time.time()
  print(f"*** Load done in {end-start}s")

  runner.run()

  print("step 1: copy the (transposed) matrix structure to the device -- once, outside the BFS loop")

  mat_vals_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz, mat_vals_buf, np.float32)
  runner.memcpy_h2d(sym_mat_vals_buf, mat_vals_buf_1d, 0, 0, width, height, max_local_nnz,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  mat_rows_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz, mat_rows_buf, np.uint32)
  runner.memcpy_h2d(sym_mat_rows_buf, mat_rows_buf_1d, 0, 0, width, height, max_local_nnz,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  mat_col_idx_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_cols, mat_col_idx_buf,
                                            np.uint32)
  runner.memcpy_h2d(sym_mat_col_idx_buf, mat_col_idx_buf_1d, 0, 0, width, height,
                     max_local_nnz_cols, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  mat_col_loc_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_cols, mat_col_loc_buf,
                                            np.uint32)
  runner.memcpy_h2d(sym_mat_col_loc_buf, mat_col_loc_buf_1d, 0, 0, width, height,
                     max_local_nnz_cols, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  mat_col_len_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_cols, mat_col_len_buf,
                                            np.uint32)
  runner.memcpy_h2d(sym_mat_col_len_buf, mat_col_len_buf_1d, 0, 0, width, height,
                     max_local_nnz_cols, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  y_rows_init_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_rows, y_rows_init_buf,
                                            np.uint32)
  runner.memcpy_h2d(sym_y_rows_init_buf, y_rows_init_buf_1d, 0, 0, width, height,
                     max_local_nnz_rows, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  local_nnz_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz, np.uint32)
  runner.memcpy_h2d(sym_local_nnz, local_nnz_1d, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  local_nnz_cols_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz_cols, np.uint32)
  runner.memcpy_h2d(sym_local_nnz_cols, local_nnz_cols_1d, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  local_nnz_rows_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz_rows, np.uint32)
  runner.memcpy_h2d(sym_local_nnz_rows, local_nnz_rows_1d, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  print("step 2: host-orchestrated BFS loop -- one f_spmv launch per hop")

  visited = np.zeros(n, dtype=bool)
  parent = np.full(n, -1, dtype=np.int64)
  frontier = np.zeros(ncols, dtype=np.float32)

  visited[source] = True
  parent[source] = source
  frontier[source] = 1.0

  iteration_log = []
  iteration = 0
  bfs_start = time.time()
  while True:
    t0 = time.time()

    x_tx_hwl = dist_x_to_hwl(ncols, frontier, local_vec_sz, np_cols, np_rows)
    x_tx_buf_1d = hwl_to_oned_colmajor(height, width, local_vec_sz, x_tx_hwl, np.float32)
    runner.memcpy_h2d(sym_x_tx_buf, x_tx_buf_1d, 0, 0, width, height, local_vec_sz,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)

    runner.launch("f_spmv", nonblock=False)

    y_1d = np.zeros(height * width * local_out_vec_sz, np.float32)
    runner.memcpy_d2h(y_1d, sym_y_local_buf, 0, 0, width, height, local_out_vec_sz,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)

    t_device = time.time() - t0

    y_hwl = oned_to_hwl_colmajor(height, width, local_out_vec_sz, y_1d, np.float32)
    y_full = unpad_3d_to_1d(nrows, y_hwl)[0:n]

    # boolean semiring adjustment: any positive sum means "reached this hop"
    candidate = y_full > 0.5
    new_mask = candidate & ~visited
    new_nodes = np.nonzero(new_mask)[0]
    n_new = int(new_nodes.size)

    if n_new == 0:  # popcount(new_mask) == 0 -> no more reachable nodes
      break

    frontier_set = set(np.nonzero(frontier)[0].tolist())
    find_parents(A_csc.indptr, A_csc.indices, new_nodes, frontier_set, parent)

    visited |= new_mask
    frontier = new_mask.astype(np.float32)

    iteration += 1
    t_iter = time.time() - t0
    iteration_log.append({
        "hop": iteration,
        "new_nodes": n_new,
        "device_roundtrip_s": round(t_device, 6),
        "total_s": round(t_iter, 6),
    })
    print(f"[bfs] hop {iteration}: +{n_new} nodes "
          f"(device roundtrip {t_device*1e3:.2f} ms, total {t_iter*1e3:.2f} ms)")

  bfs_total = time.time() - bfs_start
  runner.stop()

  print(f"*** BFS done in {bfs_total}s over {iteration} hops, "
        f"{int(np.sum(visited))}/{n} nodes visited")

  passed, n_mismatch, n_bad_parents = verify_bfs(A_csr, n, source, visited, parent)

  log_run({
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "infile_mtx": infile_mtx,
      "n": n,
      "nnz": nnz,
      "pe_grid": f"{np_cols}x{np_rows}",
      "source": source,
      "hops": iteration,
      "nodes_visited": int(np.sum(visited)),
      "compile_time_s": round(compile_time, 6),
      "bfs_total_s": round(bfs_total, 6),
      "per_hop": iteration_log,
      "verify_pass": passed,
      "visited_mismatches": n_mismatch,
      "invalid_parents": n_bad_parents,
  })


if __name__ == "__main__":
  main()
