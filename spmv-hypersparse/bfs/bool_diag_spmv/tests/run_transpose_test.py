#!/usr/bin/env cs_python
""" Pure-Python/NumPy prototype + validation of the Phase B in-place
  cycle-leader transpose (see the direction-optimizing-BFS plan). No device
  interaction at all -- this only exercises the ALGORITHM (transposing one
  PE's local hypersparse block from "grouped by column" (today's
  mat_col_idx/loc/len_buf + mat_rows_buf, CSC-by-source) into "grouped by
  row" (CSR-by-destination), reusing the SAME physical array instead of a
  second one) against every PE block preprocess_bool.py produces for a real
  fixture, cross-checked against scipy's own sparse .T of that same block.

  Deliberately kept out of any .csl file: iterating in Python first lets the
  cycle-following logic (the part genuinely at risk of an off-by-one/aliasing
  bug -- see the plan's own Phase B writeup) be validated with fast
  edit/run cycles, before it's ported to CSL where every bug costs a full
  cslc compile to see.

  How to run
     cs_python run_transpose_test.py --infile_mtx=data/rmat_s8_e4.mtx \
        --num_pe_cols=8 --num_pe_rows=8
"""

import argparse

import numpy as np
import scipy.sparse as sp
from graph_loader import load_graph
from preprocess_bool import preprocess


def cycle_leader_transpose_inplace(col_idx, col_loc, col_len, rows_buf, blk):
  """In-place transpose of one PE's local CSC-by-source block into a
  CSR-by-destination block, reusing rows_buf's own array object for the
  result (mirrors what bool_pe.csl's Phase B does to mat_rows_buf's actual
  device memory -- see the plan).

  Inputs, all trimmed to their real (not padded max_local_nnz*) length:
    col_idx[nnz_cols]: distinct nonzero column values, ascending
    col_loc[nnz_cols]: prefix-sum start offset into rows_buf for each column
    col_len[nnz_cols]: degree (nonzero-row count) of each column
    rows_buf[nnz]: destination row value per edge, grouped by column
                   (col_loc[i]..col_loc[i]+col_len[i]) -- MUTATED IN PLACE.
    blk: dense per-PE dimension (both row and column range are 0..blk-1)

  Returns (row_idx, row_loc, row_len, rows_buf) -- row_idx/row_loc/row_len
  mirror col_idx/col_loc/col_len's own shape/semantics but keyed by row;
  rows_buf is the same array object passed in, now holding source-column
  values grouped by row (row_loc[j]..row_loc[j]+row_len[j]) instead of
  destination-row values grouped by column.
  """
  nnz_cols = len(col_idx)
  nnz = len(rows_buf)

  # --- counting pass: row-degree histogram -> row_idx/row_loc/row_len ---
  row_degree = np.zeros(blk, dtype=np.int64)
  for p in range(nnz):
    row_degree[rows_buf[p]] += 1

  nz_rows = np.nonzero(row_degree)[0]
  nnz_rows = len(nz_rows)
  row_idx = nz_rows.astype(col_idx.dtype)
  row_loc = np.zeros(nnz_rows, dtype=col_loc.dtype)
  row_len = np.zeros(nnz_rows, dtype=col_len.dtype)
  row_to_bucket = np.full(blk, -1, dtype=np.int64)
  running = 0
  for b, r in enumerate(nz_rows):
    row_to_bucket[r] = b
    row_loc[b] = running
    row_len[b] = row_degree[r]
    running += row_degree[r]
  assert running == nnz

  # --- col_of(p): which column-group old position p belongs to. On-device
  # (see the plan) this is a binary search over col_loc during the
  # scattered cycle-jump access pattern below, to avoid an nnz-sized lookup
  # array; precomputing it directly here is functionally equivalent and
  # simpler for a correctness-focused prototype (memory cost only matters
  # on-device).
  col_of = np.zeros(nnz, dtype=col_idx.dtype)
  for i in range(nnz_cols):
    col_of[col_loc[i]:col_loc[i] + col_len[i]] = col_idx[i]

  # --- cycle-leader in-place scatter ---
  cursor = row_loc.copy()
  visited = np.zeros(nnz, dtype=bool)

  def target_of(r):
    b = row_to_bucket[r]
    t = cursor[b]
    cursor[b] += 1
    return t

  for p0 in range(nnz):
    if visited[p0]:
      continue
    cur = p0
    r_cur = rows_buf[cur]
    c_cur = col_of[cur]
    visited[cur] = True
    while True:
      target = target_of(r_cur)
      if visited[target]:
        # closes the cycle -- target must be p0 (the only unvisited-until-now
        # position pi could map back onto, since pi is a bijection and every
        # other position on this cycle was freshly marked visited above)
        rows_buf[target] = c_cur
        break
      r_next = rows_buf[target]
      c_next = col_of[target]
      rows_buf[target] = c_cur
      visited[target] = True
      r_cur, c_cur, cur = r_next, c_next, target

  assert visited.all()
  return row_idx, row_loc, row_len, rows_buf


def reference_transpose_block(col_idx, col_loc, col_len, rows_buf, blk):
  """Independent reference for the same local block, via scipy -- decode
  (col_idx/col_loc/col_len, rows_buf) into explicit (row, col) edge pairs,
  build a scipy CSR from the TRANSPOSE (col, row) directly, and read back
  its own indptr/indices as the expected row_idx/row_loc/row_len/cols_buf.
  Deliberately not sharing any code with cycle_leader_transpose_inplace --
  the whole point is an implementation with zero shared logic to catch bugs
  the prototype's own reasoning could otherwise miss."""
  nnz_cols = len(col_idx)
  cols = np.concatenate([np.full(col_len[i], col_idx[i]) for i in range(nnz_cols)])
  rows = rows_buf.copy()
  # (rows, cols) is exactly this block's edge list (row=destination,
  # col=source -- this repo's own A[r,c]!=0 convention), in the SAME order
  # cycle_leader_transpose_inplace consumes it (grouped by column). Building
  # a COO/CSR straight from (row=dest, col=source) and reading it back
  # row-major (NOT its transpose -- row is already "destination" here) is
  # exactly the CSR-by-destination structure wanted; no .T needed, since
  # this matrix was never "by-source" to begin with, just column-*ordered*.
  mat = sp.coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(blk, blk)).tocsr()
  mat.sort_indices()
  row_idx_ref = np.nonzero(np.diff(mat.indptr) > 0)[0]
  row_loc_ref = mat.indptr[row_idx_ref]
  row_len_ref = np.diff(mat.indptr)[row_idx_ref]
  cols_buf_ref = mat.indices
  return row_idx_ref, row_loc_ref, row_len_ref, cols_buf_ref


def check_one_block(col_idx, col_loc, col_len, rows_buf, blk, label):
  row_idx, row_loc, row_len, cols_buf = cycle_leader_transpose_inplace(
      col_idx.copy(), col_loc.copy(), col_len.copy(), rows_buf.copy(), blk)
  row_idx_ref, row_loc_ref, row_len_ref, cols_buf_ref = reference_transpose_block(
      col_idx, col_loc, col_len, rows_buf, blk)

  # row_idx/row_loc/row_len only need to describe the SAME set of
  # (row -> [cols]) buckets, not necessarily in the same bucket order or
  # with cols sorted within a bucket the same way -- rebuild both into
  # {row: sorted(cols)} dicts before comparing, which is the only property
  # bool_pe.csl's compute_bottomup() (Phase C) will actually rely on.
  def as_dict(idx, loc, ln, buf):
    return {int(idx[b]): sorted(int(x) for x in buf[loc[b]:loc[b] + ln[b]])
            for b in range(len(idx))}

  got = as_dict(row_idx, row_loc, row_len, cols_buf)
  want = as_dict(row_idx_ref, row_loc_ref, row_len_ref, cols_buf_ref)
  ok = got == want
  print(f"  [{label}] rows={len(row_idx)} nnz={len(rows_buf)} blk={blk}: "
        f"{'OK' if ok else 'MISMATCH'}")
  if not ok:
    bad_rows = [r for r in set(got) | set(want) if got.get(r) != want.get(r)]
    print(f"    mismatched rows (up to 10): {bad_rows[:10]}")
  return ok


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--infile_mtx", required=True)
  parser.add_argument("--num_pe_cols", type=int, required=True)
  parser.add_argument("--num_pe_rows", type=int, required=True)
  args = parser.parse_args()

  A_coo = load_graph(args.infile_mtx)
  A_csr = A_coo.tocsr(copy=True)
  A_csr = A_csr.sorted_indices()
  A_csc = A_csr.tocsc(copy=True)
  A_csc = A_csc.sorted_indices()
  [nrows, ncols] = A_csr.shape
  nnz = A_csr.nnz

  matrix_info = preprocess(
      nrows, ncols, nnz, args.num_pe_cols, args.num_pe_rows,
      A_csr.indptr, A_csr.indices, A_csc.indptr, A_csc.indices,
  )
  blk = -(-nrows // args.num_pe_cols)  # ceil(n/P), matches run_bfs.py's own blk

  local_nnz = matrix_info["local_nnz"]
  local_nnz_cols = matrix_info["local_nnz_cols"]
  mat_rows_buf = matrix_info["mat_rows_buf"]
  mat_col_idx_buf = matrix_info["mat_col_idx_buf"]
  mat_col_loc_buf = matrix_info["mat_col_loc_buf"]
  mat_col_len_buf = matrix_info["mat_col_len_buf"]

  print(f"infile_mtx={args.infile_mtx} n={nrows} nnz={nnz} "
        f"grid={args.num_pe_cols}x{args.num_pe_rows} blk={blk}")

  n_ok = 0
  n_checked = 0
  for py in range(args.num_pe_rows):
    for px in range(args.num_pe_cols):
      nc = int(local_nnz_cols[py, px, 0])
      nz = int(local_nnz[py, px, 0])
      if nz == 0:
        continue  # empty block -- nothing to transpose, trivially fine
      n_checked += 1
      ok = check_one_block(
          mat_col_idx_buf[py, px, :nc], mat_col_loc_buf[py, px, :nc],
          mat_col_len_buf[py, px, :nc], mat_rows_buf[py, px, :nz], blk,
          label=f"PE({px},{py})")
      n_ok += int(ok)

  print(f"{n_ok}/{n_checked} non-empty PE blocks matched the scipy reference")
  assert n_ok == n_checked, "cycle-leader transpose prototype has a bug -- see mismatches above"
  print("PASS")


if __name__ == "__main__":
  main()
