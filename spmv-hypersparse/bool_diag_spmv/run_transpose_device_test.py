#!/usr/bin/env cs_python
""" Device-level cross-check for bool_pe.csl's transpose_structure() (Phase B
  of the direction-optimizing BFS work -- see the plan). Complements
  run_transpose_test.py (the pure-Python prototype, already validated
  against scipy): this script actually runs the on-device kernel, forces
  the one-way top-down -> bottom-up switch to fire immediately after round
  1 (tau_switch_count=0, so any nonzero discovery count triggers it), reads
  back every PE's mat_row_idx/loc/len_buf and (now-repurposed) mat_rows_buf,
  and compares each PE against cycle_leader_transpose_inplace() run on that
  SAME PE's ORIGINAL (pre-transpose) block -- the identical function
  run_transpose_test.py already validated against scipy, so a match here
  means the CSL port is transposing correctly, not just that the port
  compiles.

  compute() itself is NOT touched by this (Phase C isn't implemented yet --
  see the plan), so rounds after the switch fires produce semantically
  meaningless BFS results (compute() keeps reading mat_rows_buf as if it
  still held destination-row values, which it no longer does post-switch).
  That's expected and irrelevant here: this script only reads back the
  transposed STRUCTURE buffers, never the BFS output itself.

  How to run
     cs_python run_transpose_device_test.py --arch=wse3 --num_pe_cols=8 \
        --num_pe_rows=8 --driver=cslc --infile_mtx=data/rmat_s8_e4.mtx --source=0
"""

import argparse
import math
import os

import numpy as np
from device_io import csl_compile_core, hwl_to_oned_colmajor, single_source_seed_pe
from graph_loader import load_graph
from preprocess_bool import preprocess
from run_transpose_test import cycle_leader_transpose_inplace

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--infile_mtx", required=True)
  parser.add_argument("--num_pe_cols", type=int, required=True)
  parser.add_argument("--num_pe_rows", type=int, required=True)
  parser.add_argument("--fabric-dims")
  parser.add_argument("--width-west-buf", default=0, type=int)
  parser.add_argument("--width-east-buf", default=0, type=int)
  parser.add_argument("--channels", default=1, type=int)
  parser.add_argument("-d", "--driver", default="cslc")
  parser.add_argument("--cmaddr")
  parser.add_argument("--arch")
  parser.add_argument("--latestlink", default="out/latest")
  parser.add_argument("--source", type=int, default=0)
  return parser.parse_args()


def main():
  args = parse_args()
  P = args.num_pe_cols
  assert args.num_pe_cols == args.num_pe_rows
  width = height = P

  A_coo = load_graph(args.infile_mtx)
  A_csr = A_coo.tocsr(copy=True).sorted_indices()
  A_csc = A_csr.tocsc(copy=True).sorted_indices()
  n = A_csr.shape[0]
  nnz = A_csr.nnz
  assert 0 <= args.source < n

  matrix_info = preprocess(n, n, nnz, P, P, A_csr.indptr, A_csr.indices,
                            A_csc.indptr, A_csc.indices)
  max_local_nnz = matrix_info["max_local_nnz"]
  max_local_nnz_cols = matrix_info["max_local_nnz_cols"]
  max_local_nnz_rows = matrix_info["max_local_nnz_rows"]
  mat_rows_buf = matrix_info["mat_rows_buf"]
  mat_col_idx_buf = matrix_info["mat_col_idx_buf"]
  mat_col_loc_buf = matrix_info["mat_col_loc_buf"]
  mat_col_len_buf = matrix_info["mat_col_len_buf"]
  local_nnz = matrix_info["local_nnz"]
  local_nnz_cols = matrix_info["local_nnz_cols"]

  blk = math.ceil(n / P)
  bitmap_words = (blk + 31) // 32
  seed_px, seed_py, seed_local_x = single_source_seed_pe(args.source, blk, P)

  fabric_offset_x = 1
  fabric_offset_y = 1
  core_fabric_offset_x = fabric_offset_x + 3 + args.width_west_buf
  core_fabric_offset_y = fabric_offset_y
  min_fabric_width = core_fabric_offset_x + width + 2 + 1 + args.width_east_buf
  min_fabric_height = core_fabric_offset_y + height + 1
  fabric_width = fabric_height = 0
  if args.fabric_dims:
    w_str, h_str = args.fabric_dims.split(",")
    fabric_width, fabric_height = int(w_str), int(h_str)
  if fabric_width == 0 or fabric_height == 0:
    fabric_width, fabric_height = min_fabric_width, min_fabric_height

  code_csl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "layout_bool.csl")
  csl_compile_core(
      args.driver, code_csl, args.latestlink, fabric_width, fabric_height,
      core_fabric_offset_x, core_fabric_offset_y, False, args.arch,
      P, P, blk, max_local_nnz, max_local_nnz_cols, max_local_nnz_rows,
      args.channels, args.width_west_buf, args.width_east_buf,
      max_rounds=2,  # only round 0/1 matter; keep ts_buf tiny
      tau_switch_count=0,  # force the switch immediately after round 1
  )

  runner = SdkRuntime(args.latestlink, cmaddr=args.cmaddr, simfab_numthreads=64,
                       suppress_simfab_trace=True)
  sym_x_bitmap = runner.get_id("x_bitmap")
  sym_mat_rows_buf = runner.get_id("mat_rows_buf")
  sym_mat_col_idx_buf = runner.get_id("mat_col_idx_buf")
  sym_mat_col_loc_buf = runner.get_id("mat_col_loc_buf")
  sym_mat_col_len_buf = runner.get_id("mat_col_len_buf")
  sym_local_nnz = runner.get_id("local_nnz")
  sym_local_nnz_cols = runner.get_id("local_nnz_cols")
  sym_local_nnz_rows = runner.get_id("local_nnz_rows")
  sym_mat_row_idx_buf = runner.get_id("mat_row_idx_buf")
  sym_mat_row_loc_buf = runner.get_id("mat_row_loc_buf")
  sym_mat_row_len_buf = runner.get_id("mat_row_len_buf")
  sym_is_bottom_up_dbg = runner.get_id("is_bottom_up_dbg")

  runner.load()
  runner.run()

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
  local_nnz_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz, np.uint32)
  runner.memcpy_h2d(sym_local_nnz, local_nnz_1d, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  local_nnz_cols_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz_cols, np.uint32)
  runner.memcpy_h2d(sym_local_nnz_cols, local_nnz_cols_1d, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)

  runner.memcpy_h2d(sym_x_bitmap, seed_local_x, seed_px, seed_py, 1, 1, bitmap_words,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)

  print("running f_spmv_iter (tau_switch_count=0 -> switch fires right after round 1)...")
  runner.launch("f_spmv_iter", nonblock=False)

  is_bu_buf = np.zeros(height * width, np.uint32)
  runner.memcpy_d2h(is_bu_buf, sym_is_bottom_up_dbg, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)
  is_bu = np.reshape(is_bu_buf, (height, width, 1), order="F")[:, :, 0]
  assert np.all(is_bu == 1), "switch never fired on some PE -- tau_switch_count=0 should be immediate"
  print("confirmed: is_bottom_up fired on every PE")

  local_nnz_rows_1d = np.zeros(height * width, np.uint32)
  runner.memcpy_d2h(local_nnz_rows_1d, sym_local_nnz_rows, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)
  device_local_nnz_rows = np.reshape(local_nnz_rows_1d, (height, width, 1), order="F")[:, :, 0]

  mat_row_idx_1d = np.zeros(height * width * max_local_nnz_rows, np.uint32)
  runner.memcpy_d2h(mat_row_idx_1d, sym_mat_row_idx_buf, 0, 0, width, height, max_local_nnz_rows,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)
  device_row_idx = np.reshape(mat_row_idx_1d, (height, width, max_local_nnz_rows), order="F")

  mat_row_loc_1d = np.zeros(height * width * max_local_nnz_rows, np.uint32)
  runner.memcpy_d2h(mat_row_loc_1d, sym_mat_row_loc_buf, 0, 0, width, height, max_local_nnz_rows,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)
  device_row_loc = np.reshape(mat_row_loc_1d, (height, width, max_local_nnz_rows), order="F")

  mat_row_len_1d = np.zeros(height * width * max_local_nnz_rows, np.uint32)
  runner.memcpy_d2h(mat_row_len_1d, sym_mat_row_len_buf, 0, 0, width, height, max_local_nnz_rows,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)
  device_row_len = np.reshape(mat_row_len_1d, (height, width, max_local_nnz_rows), order="F")

  mat_rows_buf_after_1d = np.zeros(height * width * max_local_nnz, np.uint32)
  runner.memcpy_d2h(mat_rows_buf_after_1d, sym_mat_rows_buf, 0, 0, width, height, max_local_nnz,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=False)
  device_cols_buf = np.reshape(mat_rows_buf_after_1d, (height, width, max_local_nnz), order="F")

  runner.stop()

  def as_dict(idx, loc, ln, buf, nrows):
    return {int(idx[b]): sorted(int(x) for x in buf[loc[b]:loc[b] + ln[b]])
            for b in range(nrows)}

  n_ok = n_checked = 0
  for py in range(height):
    for px in range(width):
      nz = int(local_nnz[py, px, 0])
      if nz == 0:
        continue
      n_checked += 1
      nc = int(local_nnz_cols[py, px, 0])
      # Python reference: run the SAME already-validated prototype on this
      # PE's ORIGINAL (pre-transpose) block.
      row_idx_ref, row_loc_ref, row_len_ref, cols_buf_ref = cycle_leader_transpose_inplace(
          mat_col_idx_buf[py, px, :nc].copy(), mat_col_loc_buf[py, px, :nc].copy(),
          mat_col_len_buf[py, px, :nc].copy(), mat_rows_buf[py, px, :nz].copy(), blk)
      want = as_dict(row_idx_ref, row_loc_ref, row_len_ref, cols_buf_ref, len(row_idx_ref))

      nrows_dev = int(device_local_nnz_rows[py, px])
      got = as_dict(device_row_idx[py, px], device_row_loc[py, px], device_row_len[py, px],
                     device_cols_buf[py, px], nrows_dev)

      ok = got == want and nrows_dev == len(row_idx_ref)
      n_ok += int(ok)
      status = "OK" if ok else "MISMATCH"
      print(f"  PE({px},{py}) nnz={nz} nrows_dev={nrows_dev} nrows_ref={len(row_idx_ref)}: {status}")
      if not ok:
        bad = [r for r in set(got) | set(want) if got.get(r) != want.get(r)]
        print(f"    mismatched rows (up to 10): {bad[:10]}")

  print(f"{n_ok}/{n_checked} non-empty PE blocks matched the Python-prototype reference")
  assert n_ok == n_checked, "on-device transpose_structure() diverges from the validated prototype"
  print("PASS")


if __name__ == "__main__":
  main()
