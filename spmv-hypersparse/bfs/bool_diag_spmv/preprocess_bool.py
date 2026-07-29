import numpy as np


# Boolean-semiring variant of preprocess.py: structural-only, no A_vals/mat_vals_buf.
#
# name mapping between spmv kernel and this C code
#   C code           spmv kernel
# ----------------------------------
#  local_nzcols     local_nnzcols
#  local_nzrows     local_nnzrows
#  local_nnz        local_nnz
#  A_colloc         mat_col_loc_buf
#  A_collen         mat_col_len_buf
#  A_colidx         mat_col_idx_buf
#  A_rows           mat_rows_buf
#
# Vectorized with numpy (no per-nonzero Python loop) -- the original
# implementation did three full O(nnz) passes in plain Python (`for col in
# range(ncols): for colidx in range(start, end): ...`), which for graphs
# with tens of millions of nonzeros (e.g. SNAP-scale pokec/topcats) cost
# multiple CPU-minutes of local, single-threaded, unvectorized work before
# ever reaching the actual remote CSL compile -- easy to mistake for "the
# cluster is slow" when it was actually this client-side step. Every
# per-nonzero decision below is instead expressed as a numpy
# sort/unique/bincount/cumsum call over the full nonzero array at once, so
# it scales the same O(nnz) but at C/vectorized speed, not interpreted
# per-element speed. See the module's own git history / project memory for
# the timing investigation that motivated this rewrite.
def preprocess(
    # A is nrows-by-ncols with nnz nonzeros
    nrows: int,
    ncols: int,
    nnz: int,
    # core rectangle of spmv is fabx-by-faby
    fabx: int,
    faby: int,
    # (csrRowPtr, csrColInd) is the CSR representation (structural, no values)
    csrRowPtr: np.ndarray,
    csrColInd: np.ndarray,
    # (cscColPtr, cscRowInd) is the CSC representation (structural, no values)
    cscColPtr: np.ndarray,
    cscRowInd: np.ndarray,
):
  """
    Given a sparse boolean matrix A of dimension nrows-by-ncols with nnz nonzeros
    and the dimension of core rectangle fabx-by-faby, parition the matrix
    A such that PE(px=j, py=i) contains the submatrix Aij with the
    following quantities:

    local_nzrows: number of nonzero rows
    local_nzcols: number of nonzero columns
    local_nnz: number of nonzero elements
    A_colloc[local_nzcols]: prefix sum of A_collen, used to point to A_rows
    A_collen[local_nzcols]: A_collen[j] is number of nonzeros of j-th nonzero columns
    A_colidx[local_nzcols]: column index of nonzero columns
    A_rows[local_nnz]: dense block-local row index of nonzeros (row_l)

    """
  assert csrRowPtr[0] == 0, "CSR must be base-0"
  assert cscColPtr[0] == 0, "CSC must be base-0"
  assert csrRowPtr[nrows] == nnz, "CSR has wrong nnz"
  assert cscColPtr[ncols] == nnz, "CSC has wrong nnz"

  bx = int((ncols + fabx - 1) / fabx)  # number of columns of a block
  by = int((nrows + faby - 1) / faby)  # number of rows of a block

  # ---- per-nonzero column/row arrays, CSC order (col-major: col ascending,
  # row ascending within each col -- REQUIRES sorted CSC row indices, same
  # invariant the original loop's own "Remark" comment relied on; callers
  # pass scipy's own .tocsc(), whose indices are sorted by construction
  # here, so no extra .sort_indices() call is needed -- verified against
  # graph_loader.py's own CSR/CSC construction). ----
  col_per_nz = np.repeat(np.arange(ncols, dtype=np.int64), np.diff(cscColPtr))
  row_per_nz = cscRowInd.astype(np.int64)
  row_b_per_nz = row_per_nz // by
  col_b_per_nz = col_per_nz // bx
  row_l_per_nz = row_per_nz - row_b_per_nz * by
  col_l_per_nz = col_per_nz - col_b_per_nz * bx

  max_grid_dim = max(faby, fabx)
  del max_grid_dim  # unused now -- was the pure-Python loop's `counted[]` scratch size

  # step 1 (local_nnz): a nonzero's block is fully determined by (row_b,
  # col_b) -- a straight histogram over the flat block id, no per-nonzero
  # branching needed at all.
  block_id_per_nz = row_b_per_nz * fabx + col_b_per_nz
  local_nnz = np.bincount(block_id_per_nz, minlength=faby * fabx).reshape(faby, fabx, 1)

  # step 1 (local_nzcols): count of DISTINCT (row_b, col) combinations per
  # block -- the original loop's `counted[row_b] != check_token(=col)` gate
  # counts each (row_b, col) pair at most once per col, i.e. exactly once
  # per distinct (row_b, col_b, col) triple (col_b follows deterministically
  # from col). `col < ncols` always, so `row_b * ncols + col` is a safe
  # unique key -- np.unique's own sort does in one vectorized pass what the
  # scalar `counted[]` scratch array did one element at a time.
  rowb_col_key = row_b_per_nz * np.int64(ncols) + col_per_nz
  unique_rc_key, unique_rc_inverse, unique_rc_count = np.unique(
      rowb_col_key, return_inverse=True, return_counts=True)
  u_row_b = unique_rc_key // ncols
  u_col = unique_rc_key % ncols
  u_col_b = u_col // bx
  u_col_l = u_col - u_col_b * bx
  u_block_id = u_row_b * fabx + u_col_b
  local_nzcols = np.bincount(u_block_id, minlength=faby * fabx).reshape(faby, fabx, 1)

  # step 2 (local_nzrows): symmetric with step 1's local_nzcols, but over
  # CSR (distinct (col_b, row) combinations per block, i.e. exactly once
  # per distinct (row_b, col_b, row) triple, row_b determined by row).
  row_per_nz_csr = np.repeat(np.arange(nrows, dtype=np.int64), np.diff(csrRowPtr))
  col_per_nz_csr = csrColInd.astype(np.int64)
  colb_row_key = (col_per_nz_csr // bx) * np.int64(nrows) + row_per_nz_csr
  unique_cr_key = np.unique(colb_row_key)
  u2_col_b = unique_cr_key // nrows
  u2_row = unique_cr_key % nrows
  u2_row_b = u2_row // by
  u2_block_id = u2_row_b * fabx + u2_col_b
  local_nzrows = np.bincount(u2_block_id, minlength=faby * fabx).reshape(faby, fabx, 1)

  # step 3: compute maximum dimension of Aij
  max_local_nnz = int(local_nnz.max())
  max_local_nnz_cols = int(local_nzcols.max())
  max_local_nnz_rows = int(local_nzrows.max())

  assert (max_local_nnz < np.iinfo(
      np.uint16).max), "LOCAL NUMBER OF NONZEROS WILL OVERFLOW, TRY USING A LARGER FABRIC"
  assert (max_local_nnz_cols < np.iinfo(
      np.uint16).max), "LOCAL NUMBER OF NZCOLS WILL OVERFLOW, TRY USING A LARGER FABRIC"
  assert (max_local_nnz_rows < np.iinfo(
      np.uint16).max), "LOCAL NUMBER OF NZROWS WILL OVERFLOW, TRY USING A LARGER FABRIC"
  # mat_rows_buf now stores direct dense row-block indices (row_l, see step
  # 5 below) instead of a compact position, so the real bound on its values
  # is `by` (the per-PE dense row-block size, i.e. the kernel's `blk`), not
  # max_local_nnz_rows -- assert that explicitly (previously implicitly
  # covered, since compact indices were always <= max_local_nnz_rows <= by).
  assert (by < np.iinfo(
      np.uint16).max), "PER-PE ROW BLOCK SIZE (by) WILL OVERFLOW, TRY USING A LARGER FABRIC"
  # no data overflows u16, we can convert the data to u16
  local_nnz = local_nnz.astype(np.uint16)
  local_nzrows = local_nzrows.astype(np.uint16)
  local_nzcols = local_nzcols.astype(np.uint16)

  #     spmv kernel                      actual storage in preprocess
  # ------------------------------------------------------------------
  # mat_rows_buf[max_local_nnz]           A_rows[local_nnz]
  # mat_col_loc_buf[max_local_nnz_cols]   A_colloc[local_nzcols]
  # mat_col_len_buf[max_local_nnz_cols]   A_collen[local_nzcols]
  # mat_col_idx_buf[max_local_nnz_cols]   A_colidx[local_nzcols]
  #
  # To prepare the data for spmv, each PE allocates the maximum dimension
  # max_local_nnz, max_local_nnz_cols or max_local_nnz_rows
  A_rows = np.zeros((faby, fabx, max_local_nnz), dtype=np.uint16)
  A_colloc = np.zeros((faby, fabx, max_local_nnz_cols), dtype=np.uint16)
  A_collen = np.zeros((faby, fabx, max_local_nnz_cols), dtype=np.uint16)
  A_colidx = np.zeros((faby, fabx, max_local_nnz_cols), dtype=np.uint16)

  # step 4 (formerly step 5): compute A_colloc, A_colidx, A_colen and A_rows.
  # A_rows now stores row_l directly (the dense block-local row index) --
  # no compact "position in y_rows" indirection any more (the compact
  # y_rows scheme this comment used to describe was dropped: the on-device
  # kernel now keeps a dense per-PE bitmap over `by`/`blk` rows instead of a
  # compact scratch array, so there is no compact index space to map into).
  #
  # Vectorized restatement of the original per-nonzero loop: `unique_rc_key`
  # above (sorted ascending by row_b then col, since it was built as
  # row_b*ncols+col) already enumerates every distinct (row_b, col_b, col)
  # triple this block needs an A_colidx/A_colloc/A_collen slot for, in
  # EXACTLY the order the original loop assigned increasing `pos` values --
  # because col_b = col // bx is non-decreasing as col increases within a
  # fixed row_b, entries belonging to the same (row_b, col_b) block form a
  # contiguous run in this sorted key array. `pos` is therefore just each
  # entry's 0-indexed rank within its own contiguous same-block run.
  same_block_as_prev = np.empty(u_block_id.shape[0], dtype=bool)
  same_block_as_prev[0] = False
  same_block_as_prev[1:] = u_block_id[1:] == u_block_id[:-1]
  # cumulative count reset to 0 at every run boundary -- a standard
  # "position within contiguous group" trick: subtract, from each index,
  # the index where its own run started.
  run_start_idx = np.where(~same_block_as_prev)[0]
  run_len = np.diff(np.append(run_start_idx, u_block_id.shape[0]))
  pos_in_block = np.arange(u_block_id.shape[0]) - np.repeat(run_start_idx, run_len)

  A_colidx[(u_row_b, u_col_b, pos_in_block)] = u_col_l.astype(np.uint16)
  A_collen[(u_row_b, u_col_b, pos_in_block)] = unique_rc_count.astype(np.uint16)
  # A_colloc[pos] = exclusive prefix sum of A_collen WITHIN this (row_b,
  # col_b) block, i.e. a cumsum reset to 0 at every block boundary -- same
  # run-start-subtraction trick, applied to the cumulative sum instead of a
  # plain index.
  cumsum_excl = np.concatenate(([0], np.cumsum(unique_rc_count)))[:-1]
  A_colloc[(u_row_b, u_col_b,
            pos_in_block)] = (cumsum_excl - cumsum_excl[np.repeat(run_start_idx, run_len)]).astype(np.uint16)

  # Now place every ORIGINAL nonzero's row_l into A_rows at its correct
  # flat position: this block's A_colloc[pos] (looked up via each
  # original's own group) plus that nonzero's own rank within its
  # (row_b, col) group (0-indexed, row-ascending -- guaranteed by CSC's
  # sorted row order within a column, so a subset sharing (row_b, col) is
  # still row-sorted). `unique_rc_inverse` maps each ORIGINAL nonzero to
  # its group's index in the sorted unique array, and -- because CSC order
  # is col-major with row ascending within col, and row_b is non-decreasing
  # in row for fixed col -- entries sharing a group are themselves
  # contiguous in the ORIGINAL array too, so the same run-start trick
  # applies a second time, on `unique_rc_inverse` directly.
  same_group_as_prev = np.empty(nnz, dtype=bool)
  same_group_as_prev[0] = False
  same_group_as_prev[1:] = unique_rc_inverse[1:] == unique_rc_inverse[:-1]
  group_run_start_idx = np.where(~same_group_as_prev)[0]
  group_run_len = np.diff(np.append(group_run_start_idx, nnz))
  pos_rel_rowidx = np.arange(nnz) - np.repeat(group_run_start_idx, group_run_len)

  pos_start_per_nz = A_colloc[(u_row_b, u_col_b, pos_in_block)][unique_rc_inverse]
  pos_rowidx_per_nz = pos_start_per_nz.astype(np.int64) + pos_rel_rowidx
  A_rows[(row_b_per_nz, col_b_per_nz, pos_rowidx_per_nz)] = row_l_per_nz.astype(np.uint16)

  matrix_info = {}
  matrix_info["nrows"] = nrows  # number of rows of the matrix
  matrix_info["ncols"] = ncols  # number of columns of the matrix
  matrix_info["nnz"] = nnz  # number of nonzeros of the matrix
  matrix_info["max_local_nnz"] = max_local_nnz
  matrix_info["max_local_nnz_cols"] = max_local_nnz_cols
  matrix_info["max_local_nnz_rows"] = max_local_nnz_rows
  matrix_info["mat_rows_buf"] = A_rows
  matrix_info["mat_col_loc_buf"] = A_colloc
  matrix_info["mat_col_len_buf"] = A_collen
  matrix_info["mat_col_idx_buf"] = A_colidx
  matrix_info["local_nnz"] = local_nnz
  matrix_info["local_nnz_cols"] = local_nzcols
  matrix_info["local_nnz_rows"] = local_nzrows

  return matrix_info
