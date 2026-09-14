""" Shared host<->device data-marshaling helpers for bool_diag_spmv's scripts
  (run_bfs.py, run_bfs.appliance.py, run_graph500.py) -- the low-level
  hwl<->1d layout conversions, parent result extraction, and the
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
  elif A_hwl.dtype == np.uint32:
    assert dtype == np.uint32, "only support dtype = u32 if A is u32"
    A_1d = np.reshape(A_hwl, height * width * pe_length, order="F")
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


# The SDK's real wire payload is always 4 bytes/element (memcpy_h2d hard-
# asserts src.dtype.itemsize == 4 internally), regardless of the data_type
# kwarg -- that only controls device-side unpacking. This is also
# grpc.max_send_message_length's value; the SDK's own internal chunker
# leaves ~zero headroom for its message envelope and off-by-a-few-dozen-
# bytes overflows right at this boundary (confirmed against the installed
# cerebras/sdk/client/sdk_appliance_client.py -- see docs/ERRORS.md #4).
_H2D_WIRE_ITEMSIZE = 4
_H2D_MAX_MESSAGE_LENGTH = (1024**3 * 2) - 1024


def prepare_h2d_chunked(A_hwl: np.ndarray, height: int, width: int, elt_per_pe: int, dtype,
                        max_bytes_per_call: int = 1_610_612_736):
  """Pure-marshaling half of the h2d-chunking split (paired with
  send_h2d_chunked below): does every hwl_to_oned_colmajor call (including
  the chunked path's per-row-band slicing) up front and performs no device
  I/O, so this can run entirely outside a bfs_timing.timed_transfer
  bracket -- keeping host-side reshape cost out of a transfer-time
  measurement. Splits large transfers into multiple PE-row-band chunks,
  each safely under the ~2GiB gRPC message-size ceiling -- works around a
  real ceiling this repo hit uploading mat_rows_buf for graphs with high
  per-PE nonzero skew (berkstan, orkut; see docs/ERRORS.md #4).
  max_bytes_per_call (default 1.5GiB, real margin under the 2,147,482,624-
  byte ceiling -- not the vendor chunker's near-zero margin). Chunks along
  the PE-row (`height`) axis, not `elt_per_pe` -- x/y are documented,
  precedented origin offsets in this codebase (the reduce_select_any d2h
  fix already narrows `w`); a nonzero starting offset into elt_per_pe has
  no precedent here and is unverified, so this is the lower-risk axis to
  split on. Returns a list of (y0, h, chunk_1d) tuples consumed by
  send_h2d_chunked below; for the common single-chunk case this is a
  1-element list with y0=0, h=height."""
  total_bytes = _H2D_WIRE_ITEMSIZE * height * width * elt_per_pe
  if total_bytes <= max_bytes_per_call:
    return [(0, height, hwl_to_oned_colmajor(height, width, elt_per_pe, A_hwl, dtype))]

  num_chunks = -(-total_bytes // max_bytes_per_call)  # ceil div
  row_chunk = -(-height // num_chunks)  # ceil div
  assert _H2D_WIRE_ITEMSIZE * row_chunk * width * elt_per_pe <= max_bytes_per_call
  chunks = []
  y0 = 0
  while y0 < height:
    h = min(row_chunk, height - y0)
    chunks.append((y0, h, hwl_to_oned_colmajor(h, width, elt_per_pe, A_hwl[y0:y0 + h], dtype)))
    y0 += h
  return chunks


def send_h2d_chunked(runner, dest_sym, prepared, width: int, elt_per_pe: int, data_type, order,
                     nonblock: bool):
  """Pure-transfer half of the h2d-chunking split: issues runner.memcpy_h2d for
  each (y0, h, chunk_1d) produced by prepare_h2d_chunked above, with no numpy
  work -- safe to call from inside a bfs_timing.timed_transfer closure.
  Multi-chunk bands are always nonblock=True except the last, which takes
  the caller's `nonblock` -- so a caller wanting a blocking finish (e.g.
  need_timing=True, "the last transfer in the bracket must block so f_toc
  reads an accurate timestamp") still gets it even when the matrix was large
  enough to need chunking."""
  last = len(prepared) - 1
  for i, (y0, h, chunk_1d) in enumerate(prepared):
    runner.memcpy_h2d(dest_sym, chunk_1d, 0, y0, width, h, elt_per_pe,
                       streaming=False, data_type=data_type, order=order,
                       nonblock=nonblock if i == last else True)


def oned_to_hwl_colmajor(height: int, width: int, pe_length: int, A_1d: np.ndarray, dtype):
  """
    Given a 1-D tensor A_1d[height*width*pe_length], transform it to
    3-D tensor A[height][width][pe_length] by column-major
    """
  assert dtype == np.float32, "only support f32 readback for this kernel"
  assert A_1d.dtype == np.float32, "only support f32 to f32"
  return np.reshape(A_1d, (height, width, pe_length), order="F")


def single_source_seed_pe(source, blk, P):
  """For a genuinely single-source BFS seed (exactly one true bit, at
  `source`), return (px, py, local_x) for the ONE diagonal PE that owns
  it -- px=py=source//blk (diagonal: column==row), local_x a length-
  bitmap_words uint32 array with a single bit set at local index
  source%blk (word (source%blk)>>5, bit position (source%blk)&31 --
  bool_pe.csl's x_bitmap layout exactly). Pair with a 1x1-region
  memcpy_h2d instead of dist_x_to_diag_hwl's full (P,P,blk) rectangle:
  every OTHER diagonal PE's x_bitmap is provably already zero
  (bool_pe.csl's reduce_done() sets x_bitmap[w] = newly_word every round
  including the last, and the loop's own termination condition,
  nz_total == 0, is a non-negative sum over every diagonal PE's own
  nz_local flag -- which is 1.0 iff that PE's own x_bitmap had any nonzero
  word -- so nz_total == 0 provably means every diagonal PE's x_bitmap is
  all-zero at the moment f_spmv_iter() returns; combined with x_bitmap's
  zero state at kernel load, this holds for the very first search too),
  and non-diagonal PEs never need a host write at all regardless (every
  PE's x_bitmap is unconditionally overwritten by that round's own
  column-broadcast, in visited_bcast_done(), before compute() ever reads
  it).

  Only valid for this single-source case -- a multi-source frontier (several
  diagonal PEs genuinely live at once) would need a different seeding
  approach entirely."""
  p = source // blk
  bitmap_words = (blk + 31) // 32
  local_x = np.zeros(bitmap_words, dtype=np.uint32)
  local_idx = source % blk
  local_x[local_idx >> 5] = np.uint32(1) << (local_idx & 31)
  return p, p, local_x


# Must match bool_pe.csl's PARENT_NONE exactly -- #26 moved parent_values
# from storing the FULL global vertex id (u32) to just the LOCAL column
# offset within the producing PE's own column range (u16), since #24 had
# already made the global-id reconstruction (pcol_id*blk + local column) a
# host-side-only need -- no reason left to spend on-device compute or wire
# bytes on it. Local-column values span [0, blk), which
# preprocess_bool.py's own `by < u16::max` assert already guarantees fits,
# so u16::max is always a safe, unambiguous sentinel. Keep this numerically
# in sync with bool_pe.csl's own PARENT_NONE constant by hand; there is no
# single shared source of truth for the two languages.
PARENT_NONE_LOCAL = 65535


def extract_parent_result(n, blk, mat_row_idx_buf_hwl, local_nnz_rows_hwl, parent_values_hwl):
  """Host-side per-row combine, replacing the on-device parent-resolution
  relay removed in docs/ERRORS.md #24. Each PE keeps its own sparse,
  position-aligned candidates -- parent_values_hwl[py,px,i] is the LOCAL
  column (see #26; PARENT_NONE_LOCAL's own comment) that produced a
  candidate for GLOBAL row py*blk + mat_row_idx_buf_hwl[py,px,i], for
  i in [0, local_nnz_rows_hwl[py,px,0]) -- instead of every row's P
  per-PE candidates being resolved down to a single winner ON-DEVICE
  (the old `reduce_select_any_indexed_precompacted()` relay, #19-#22).
  That relay needed blk-sized wire buffers on every PE regardless of how
  sparse any single PE's own data was (the relay's cross-PE union, not
  local storage, was the actual dominant cost -- see #24's own writeup),
  so this combine moved back to the host instead, at the cost of a much
  larger d2h transfer (every PE's own array, not one post-relay column).
  #26 then shrank that transfer back down (u16 local column instead of u32
  global id) by pushing the GLOBAL id reconstruction (pcol_id*blk + local
  column) into this function too -- the PE's own column index (px) is
  already known here the same way its row index (py) already was.

  Multiple (py,px) can legally claim the same row (different PEs' column
  ranges can each have a real structural edge into it) -- any one valid
  candidate winning is correct; there's no on-device tie-break requirement
  either (see bool_pe.csl's own module docstring), so a plain vectorized
  scatter-assign (last-duplicate-wins, in whatever order numpy processes
  it) is fine, same as every other tie-break convention in this kernel.

  mat_row_idx_buf_hwl/local_nnz_rows_hwl are the SAME host-side arrays
  preprocess() already produced for the h2d upload -- no need to read them
  back from the device, only parent_values_hwl is a new d2h read."""
  height, width, max_local_nnz_rows = parent_values_hwl.shape
  py_idx = np.arange(height, dtype=np.int64).reshape(height, 1, 1)
  px_idx = np.arange(width, dtype=np.int64).reshape(1, width, 1)
  global_rows = py_idx * blk + mat_row_idx_buf_hwl.astype(np.int64)
  slot_idx = np.arange(max_local_nnz_rows).reshape(1, 1, max_local_nnz_rows)
  valid_slot = slot_idx < local_nnz_rows_hwl  # broadcasts (1,1,K) vs (height,width,1)
  has_parent = parent_values_hwl != PARENT_NONE_LOCAL
  mask = valid_slot & has_parent
  # Reconstruct the global vertex id from each candidate's own PE column
  # (px) and its LOCAL column offset (#26) -- same reconstruction
  # global_c = pcol_id*blk + local_column used to do on-device, before #26
  # moved it here.
  global_parent = px_idx * blk + parent_values_hwl.astype(np.int64)
  parent = np.full(n, -1, dtype=np.int64)
  parent[global_rows[mask]] = global_parent[mask]
  return parent


def derive_visited_from_parent(n, parent, source):
  """visited[v] == True iff parent[v] >= 0 or v is the source itself.

  This is provably equivalent to reading back bool_pe.csl's visited_bitmap
  directly, not an approximation: compute() only ever records a parent
  candidate for row v in the SAME round v's row-reduce first turns
  visited_bitmap's bit v nonzero (the visited_bitmap bit-test gate in
  compute() -- see its own comment -- means every PE that contributes a hit
  to v's row-reduce in v's true discovery round also attempts to record a
  parent candidate that round), so v's combined parent_values entry is
  non-PARENT_NONE exactly when v was ever discovered. The source is seeded
  directly into visited_bitmap in start_spmv(), never through compute(), so
  it never gets a parent recorded there -- callers already patch
  parent[source] = source after extract_parent_result(), which this
  function's `v is the source` check also covers.

  This is also Graph500's own convention: the reference implementation's
  Kernel 2 output is exactly the predecessor array (pred[v] = -1 for
  unreached, pred[root] = root) -- there's no separate "visited" output at
  all, so recovering it from parent instead of reading back visited_bitmap
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


def csl_compile_core_appliance(
    csl_path: str,  # directory containing file_config (NOT joined into one path -- see below)
    file_config: str,  # bare filename, e.g. "layout_bool.csl", relative to csl_path
    elf_dir: str,
    fabric_width: int,
    fabric_height: int,
    core_fabric_offset_x: int,
    core_fabric_offset_y: int,
    arch: Optional[str],
    np_cols: int,
    np_rows: int,
    blk: int,
    max_local_nnz: int,
    max_local_nnz_rows: int,
    channels: int,
    width_west_buf: int,
    width_east_buf: int,
    max_rounds: Optional[int] = None,
):
  """Appliance-mode counterpart to csl_compile_core: same --params, but
  compiled via cerebras.sdk.client.SdkCompiler instead of a direct `cslc`
  subprocess -- the appliance's compile step runs as its own cluster job,
  separate from the run step (see run_bfs.appliance.py's own module
  docstring for why), so this returns an artifact path identifying the
  compiled result rather than compiling in place. Matches the exact call
  shape in ALCF's own docs (docs.alcf.anl.gov/ai-testbed/cerebras/csl/,
  v2.10.0 csl-examples), NOT sdk-hypersparse-spmv/run.appliance.py's
  3-argument version, which appears to be from a different/older SDK
  build than the one this was verified against.

  Local import: this module is also imported by run_bfs.py, which only
  ever runs against the simulator and doesn't have (or need)
  cerebras.sdk.client installed.

  fabric_width/fabric_height are passed through as given -- the CALLER
  decides whether these are minimal simulator-sized dims or the real
  WSE-3's full physical dims (762,1172), see run_bfs.appliance.py's own
  fabric-dims branch."""
  from cerebras.sdk.client import SdkCompiler  # pylint: disable=import-error,no-name-in-module

  args = []
  args.append(f"--fabric-dims={fabric_width},{fabric_height}")
  args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
  args.append(f"--params=pcols:{np_cols}")
  args.append(f"--params=prows:{np_rows}")
  args.append(f"--params=blk:{blk}")
  args.append(f"--params=max_local_nnz:{max_local_nnz}")
  args.append(f"--params=max_local_nnz_rows:{max_local_nnz_rows}")
  if max_rounds is not None:
    args.append(f"--params=max_rounds:{max_rounds}")

  args.append(f"-o={elf_dir}")
  if arch is not None:
    args.append(f"--arch={arch}")
  args.append("--memcpy")
  args.append(f"--channels={channels}")
  args.append(f"--width-west-buf={width_west_buf}")
  args.append(f"--width-east-buf={width_east_buf}")

  # disable_version_check: ALCF's own sample output warns of exactly this
  # ("client semantic version X is inconsistent with cluster server
  # semantic version Y, there's a risk job could fail due to inconsistent
  # setup") -- their own tutorial scripts pass this unconditionally.
  with SdkCompiler(disable_version_check=True) as compiler:
    args_str = " ".join(args)
    # CONFIRMED (2026-07-27, via inspect.signature/docstring against the
    # actually-installed cerebras.sdk.client on a real ALCF node): compile()
    # is (app_path, csl_main, options, out_path) -- out_path is genuinely
    # "the path where to place the compile artifact on the user's machine"
    # (a .tar.gz), not redundant with app_path. Reusing csl_path for both
    # just places the artifact next to the source, matching ALCF's own
    # gemv-01 tutorial's "." / "." pattern. Validated end-to-end against a
    # real appliance-sim run, not just this signature check.
    artifact_path = compiler.compile(csl_path, file_config, args_str, csl_path)
    print("compile artifact_path:", artifact_path)
    return artifact_path
