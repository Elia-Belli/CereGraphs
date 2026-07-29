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

  Only valid for this single-source, f_spmv_iter case -- NOT for
  run_host_driven_bfs.py's multi-source frontier (several diagonal PEs can
  be genuinely live at once there) or run_single_spmv.py's one-shot
  f_spmv (which never touches x_bitmap itself, so has no such self-zeroing
  invariant)."""
  p = source // blk
  bitmap_words = (blk + 31) // 32
  local_x = np.zeros(bitmap_words, dtype=np.uint32)
  local_idx = source % blk
  local_x[local_idx >> 5] = np.uint32(1) << (local_idx & 31)
  return p, p, local_x


def pack_dense_to_bitmap(height, width, blk, dense_hwl):
  """Inverse of unpack_bitmap_to_dense: dense_hwl is a (height, width, blk)
  float32/bool array (0.0/1.0 or False/True); returns a (height, width,
  bitmap_words) uint32 array packed the same way bool_pe.csl's
  x_bitmap/y_bitmap/visited_bitmap are (bit k of word k>>5, bit position
  k&31)."""
  bitmap_words = (blk + 31) // 32
  bitmap = np.zeros((height, width, bitmap_words), dtype=np.uint32)
  bits = dense_hwl != 0
  for k in range(blk):
    bitmap[:, :, k >> 5] |= bits[:, :, k].astype(np.uint32) << (k & 31)
  return bitmap


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


# Must match bool_pe.csl's PARENT_NONE exactly. Phase A of the on-device
# parent resolution plan: parent_local_buf now stores the FULL global
# vertex id (widened u16 local-index -> u32 global-id), so this sentinel is
# now a global-vertex-index one (u32::MAX), not a local-column-index one --
# keep this numerically in sync with bool_pe.csl's own PARENT_NONE constant
# by hand; there is no single shared source of truth for the two languages.
PARENT_NONE_GLOBAL = 4294967295


def extract_parent_result(n, blk, P, parent_hwl):
  """Assemble the length-n parent vector from parent_local_buf's PE-column-0
  slice. parent_hwl has shape (height=P, width=1, blk): Phase B of the
  on-device parent resolution plan resolves each row's P per-PE candidates
  down to a single winner ON-DEVICE (bool_pe.csl's term_col_bcast_done()
  calls mpi_x.reduce_select_any(root=0, ...) exactly once, at the very end
  of the BFS, right before host readback -- see its own comment), landing
  the result at a FIXED PE-column (0) for every row so the host can read
  back a plain narrow rectangle instead of the full P-wide grid this used
  to require (the fix for the real d2h gRPC ~2GiB message-size ceiling --
  see project memory / GRAPH500_BENCHMARK.md). No per-row combine needed
  here any more -- just decode column 0's global ids and map the sentinel
  to -1. `P` is accepted but unused (kept for call-site stability across
  this repo's four callers)."""
  del P  # unused in Phase B -- see docstring
  global_c = parent_hwl[:, 0, :].astype(np.int64)
  parent = np.where(global_c == PARENT_NONE_GLOBAL, -1, global_c).reshape(-1)[0:n]
  return parent


def derive_visited_from_parent(n, parent, source):
  """visited[v] == True iff parent[v] >= 0 or v is the source itself.

  This is provably equivalent to reading back bool_pe.csl's visited_bitmap
  directly, not an approximation: compute() only ever records a parent
  candidate for row v in the SAME round v's row-reduce first turns
  visited_bitmap's bit v nonzero (the visited_bitmap bit-test gate in
  compute() -- see its own comment -- means every PE that contributes a hit
  to v's row-reduce in v's true discovery round also attempts to record a
  parent candidate that round), so parent_local_buf's row-min is non-
  PARENT_NONE exactly when v was ever discovered. The source is seeded
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
    max_local_nnz_cols: int,
    max_local_nnz_rows: int,
    channels: int,
    width_west_buf: int,
    width_east_buf: int,
    max_rounds: Optional[int] = None,
    tau_switch_count: Optional[int] = None,
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
    # left at layout_bool.csl's own default (effectively unreachable, i.e.
    # always top-down) unless a caller (see run_bfs.py) opts into the
    # direction-optimizing switch with a real fraction of n.
    if tau_switch_count is not None:
      args.append(f"--params=tau_switch_count:{int(tau_switch_count)}")

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
    max_local_nnz_cols: int,
    max_local_nnz_rows: int,
    channels: int,
    width_west_buf: int,
    width_east_buf: int,
    max_rounds: Optional[int] = None,
    tau_switch_count: Optional[int] = None,
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

  Local import: this module is also imported by run_bfs.py/
  run_single_spmv.py/run_host_driven_bfs.py, which only ever run against
  the simulator and don't have (or need) cerebras.sdk.client installed.

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
  args.append(f"--params=max_local_nnz_cols:{max_local_nnz_cols}")
  args.append(f"--params=max_local_nnz_rows:{max_local_nnz_rows}")
  if max_rounds is not None:
    args.append(f"--params=max_rounds:{max_rounds}")
  if tau_switch_count is not None:
    args.append(f"--params=tau_switch_count:{int(tau_switch_count)}")

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
