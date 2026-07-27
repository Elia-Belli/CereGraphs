import numpy as np


# fp32 variant of bool_diag_spmv/preprocess_bool.py: same dense-direct-row-index
# hypersparse compressed-column format, plus a real-valued A_vals array
# (dropped in the boolean fork, restored here -- see sdk-hypersparse-spmv/preprocess.py's
# own A_vals handling, which this mirrors).
#
# name mapping between spmv kernel and this code
#   this code        spmv kernel
# ----------------------------------
#  local_nzcols     local_nnzcols
#  local_nnz        local_nnz
#  A_colloc         mat_col_loc_buf
#  A_collen         mat_col_len_buf
#  A_colidx         mat_col_idx_buf
#  A_rows           mat_rows_buf
#  A_vals           mat_vals_buf
#
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
    # (cscColPtr, cscRowInd, cscVal) is the CSC representation
    cscColPtr: np.ndarray,
    cscRowInd: np.ndarray,
    cscVal: np.ndarray,
):
  """
    Given a sparse real-valued matrix A of dimension nrows-by-ncols with nnz
    nonzeros and the dimension of core rectangle fabx-by-faby, partition the
    matrix A such that PE(px=j, py=i) contains the submatrix Aij with the
    following quantities:

    local_nzcols: number of nonzero columns
    local_nnz: number of nonzero elements
    A_colloc[local_nzcols]: prefix sum of A_collen, used to point to A_rows
    A_collen[local_nzcols]: A_collen[j] is number of nonzeros of j-th nonzero columns
    A_colidx[local_nzcols]: column index of nonzero columns
    A_rows[local_nnz]: dense block-local row index of nonzeros (row_l)
    A_vals[local_nnz]: value of nonzeros

    """
  assert csrRowPtr[0] == 0, "CSR must be base-0"
  assert cscColPtr[0] == 0, "CSC must be base-0"
  assert csrRowPtr[nrows] == nnz, "CSR has wrong nnz"
  assert cscColPtr[ncols] == nnz, "CSC has wrong nnz"

  bx = int((ncols + fabx - 1) / fabx)  # number of columns of a block
  by = int((nrows + faby - 1) / faby)  # number of rows of a block

  local_nzcols = np.zeros((faby, fabx, 1), dtype=np.int32)
  local_nnz = np.zeros((faby, fabx, 1), dtype=np.int32)

  max_grid_dim = max(faby, fabx)
  counted = np.zeros(max_grid_dim, dtype=np.int32)

  # step 1: compute local_nzcols and local_nnz
  counted[0:max_grid_dim] = -1  # invalid token
  for col in range(ncols):
    check_token = col
    # col = col_b * bx + col_l
    col_b = int(col / bx)
    start = cscColPtr[col]
    end = cscColPtr[col + 1]
    for colidx in range(start, end):
      row = cscRowInd[colidx]
      # row = row_b * by + row_l
      row_b = int(row / by)
      local_nnz[(row_b, col_b)] += 1
      if counted[row_b] != check_token:
        local_nzcols[(row_b, col_b)] += 1
        counted[row_b] = check_token

  # step 2: compute maximum dimension of Aij
  max_local_nnz = max(local_nnz.ravel())
  max_local_nnz_cols = max(local_nzcols.ravel())

  assert (max_local_nnz < np.iinfo(
      np.uint16).max), "LOCAL NUMBER OF NONZEROS WILL OVERFLOW, TRY USING A LARGER FABRIC"
  assert (max_local_nnz_cols < np.iinfo(
      np.uint16).max), "LOCAL NUMBER OF NZCOLS WILL OVERFLOW, TRY USING A LARGER FABRIC"
  # mat_rows_buf stores direct dense row-block indices (row_l), so the real
  # bound on its values is `by` (the per-PE dense row-block size, i.e. the
  # kernel's `blk`), not max_local_nnz.
  assert (by < np.iinfo(
      np.uint16).max), "PER-PE ROW BLOCK SIZE (by) WILL OVERFLOW, TRY USING A LARGER FABRIC"
  local_nnz = local_nnz.astype(np.uint16)
  local_nzcols = local_nzcols.astype(np.uint16)

  #     spmv kernel                      actual storage in preprocess
  # ------------------------------------------------------------------
  # mat_rows_buf[max_local_nnz]           A_rows[local_nnz]
  # mat_vals_buf[max_local_nnz]           A_vals[local_nnz]
  # mat_col_loc_buf[max_local_nnz_cols]   A_colloc[local_nzcols]
  # mat_col_len_buf[max_local_nnz_cols]   A_collen[local_nzcols]
  # mat_col_idx_buf[max_local_nnz_cols]   A_colidx[local_nzcols]
  A_rows = np.zeros((faby, fabx, max_local_nnz), dtype=np.uint16)
  A_vals = np.zeros((faby, fabx, max_local_nnz), dtype=np.float32)
  A_colloc = np.zeros((faby, fabx, max_local_nnz_cols), dtype=np.uint16)
  A_collen = np.zeros((faby, fabx, max_local_nnz_cols), dtype=np.uint16)
  A_colidx = np.zeros((faby, fabx, max_local_nnz_cols), dtype=np.uint16)

  # step 3: compute A_colloc, A_colidx, A_collen, A_rows and A_vals.
  local_pos = np.zeros((faby, fabx), dtype=np.int32)
  counted[0:max_grid_dim] = -1  # invalid token
  for col in range(ncols):
    check_token = col
    col_b = int(col / bx)
    col_l = col - col_b * bx
    start = cscColPtr[col]
    end = cscColPtr[col + 1]
    for colidx in range(start, end):
      row = cscRowInd[colidx]
      row_b = int(row / by)
      row_l = row - row_b * by
      if counted[row_b] != check_token:
        pos = local_pos[(row_b, col_b)]
        A_colidx[(row_b, col_b, pos)] = col_l
        if pos > 0:
          A_colloc[(row_b, col_b,
                    pos)] = A_colloc[(row_b, col_b, pos - 1)] + A_collen[(row_b, col_b, pos - 1)]
        local_pos[(row_b, col_b)] = pos + 1
        counted[row_b] = check_token
      pos_start = A_colloc[(row_b, col_b, pos)]
      pos_rel_rowidx = A_collen[(row_b, col_b, pos)]
      pos_rowidx = pos_rel_rowidx + pos_start
      A_rows[(row_b, col_b, pos_rowidx)] = row_l
      A_vals[(row_b, col_b, pos_rowidx)] = cscVal[colidx]
      A_collen[(row_b, col_b, pos)] = pos_rel_rowidx + 1

  matrix_info = {}
  matrix_info["nrows"] = nrows
  matrix_info["ncols"] = ncols
  matrix_info["nnz"] = nnz
  matrix_info["max_local_nnz"] = max_local_nnz
  matrix_info["max_local_nnz_cols"] = max_local_nnz_cols
  matrix_info["mat_rows_buf"] = A_rows
  matrix_info["mat_vals_buf"] = A_vals
  matrix_info["mat_col_loc_buf"] = A_colloc
  matrix_info["mat_col_len_buf"] = A_collen
  matrix_info["mat_col_idx_buf"] = A_colidx
  matrix_info["local_nnz"] = local_nnz
  matrix_info["local_nnz_cols"] = local_nzcols

  return matrix_info
