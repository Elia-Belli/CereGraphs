import numpy as np


# Boolean-semiring variant of preprocess.py: structural-only, no A_vals/mat_vals_buf.
#
# name mapping between spmv kernel and this C code
#   C code           spmv kernel
# ----------------------------------
#  local_nzcols     local_nnzcols
#  local_nnz        local_nnz
#  A_colloc         mat_col_loc_buf
#  A_collen         mat_col_len_buf
#  A_colidx         mat_col_idx_buf
#  A_rows           mat_rows_buf
#
# (local_nzrows/local_nnzrows used to be here too -- removed, see
# docs/ERRORS.md #26 follow-up: provably unread by every live caller.)
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
    # (cscColPtr, cscRowInd) is the CSC representation (structural, no values)
    # -- the ONLY representation this function needs (see docs/ERRORS.md
    # #26's follow-up entry): a CSR argument pair used to live here too,
    # but fed a full SECOND np.unique()-based sort whose only output
    # (local_nzrows/max_local_nnz_rows) turned out to be read by NEITHER
    # live caller (traced both -- see step 2's own comment below), so it
    # was deleted outright rather than kept in any form. Dropping the CSR
    # argument pair means callers no longer need to build (or even load) a
    # second scipy sparse representation of the matrix at all.
    cscColPtr: np.ndarray,
    cscRowInd: np.ndarray,
):
  """
    Given a sparse boolean matrix A of dimension nrows-by-ncols with nnz nonzeros
    and the dimension of core rectangle fabx-by-faby, parition the matrix
    A such that PE(px=j, py=i) contains the submatrix Aij with the
    following quantities:

    local_nzcols: number of nonzero columns
    local_nnz: number of nonzero elements
    A_colloc[local_nzcols]: prefix sum of A_collen, used to point to A_rows
    A_collen[local_nzcols]: A_collen[j] is number of nonzeros of j-th nonzero columns
    A_colidx[local_nzcols]: column index of nonzero columns
    A_rows[local_nnz]: dense block-local row index of nonzeros (row_l)

    """
  assert cscColPtr[0] == 0, "CSC must be base-0"
  assert cscColPtr[ncols] == nnz, "CSC has wrong nnz"

  bx = int((ncols + fabx - 1) / fabx)  # number of columns of a block
  by = int((nrows + faby - 1) / faby)  # number of rows of a block

  # Checked here (not just where `by` is used far below) because
  # row_l_per_nz's storage dtype right below depends on it -- see
  # docs/ERRORS.md #26. Same assert, just moved earlier so the invariant it
  # protects is verified before anything relies on it, not after.
  assert (by < np.iinfo(
      np.uint16).max), "PER-PE ROW BLOCK SIZE (by) WILL OVERFLOW, TRY USING A LARGER FABRIC"

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
  # col_l_per_nz used to be computed here too (col_per_nz - col_b_per_nz *
  # bx) but is never read anywhere below -- a whole nnz-sized int64 array
  # (8+ GB at RMAT-s25 scale) that just sat alive for the rest of the
  # function for no reason. Dropped entirely, see docs/ERRORS.md #26.

  # row_b_per_nz/col_b_per_nz are values in [0, faby)/[0, fabx) -- both stay
  # alive all the way to the final A_rows scatter near the end of this
  # function, so halving them from int64 to int32 matters for real (see
  # docs/ERRORS.md #26 for the RMAT-s25-scale OOM this whole pass of
  # changes is fixing). This was tried once already and REVERTED: wrapping
  # ncols in np.int64(...) at the rowb_col_key line below does NOT reliably
  # promote that expression to int64 just because the multiplier is
  # int64-typed -- numpy 1.25's actual behavior keeps an int32 array's own
  # dtype there (confirmed empirically, not just reasoned about), silently
  # overflowing once row_b_per_nz*ncols exceeds int32's range (real at
  # RMAT-s22 scale and up). Fixed correctly this time: `.astype(np.int64)`
  # explicitly on row_b_per_nz itself at that one call site (not just an
  # int64-typed multiplier) forces an actual conversion, independent of
  # numpy's scalar-promotion quirks -- a transient full-width copy exists
  # only for that one expression, not for row_b_per_nz's whole lifetime.
  row_b_per_nz = row_b_per_nz.astype(np.int32)
  col_b_per_nz = col_b_per_nz.astype(np.int32)
  # row_l_per_nz's only remaining use (the final A_rows assignment) already
  # downcasts to uint16 for storage there -- do it now instead, since the
  # `by < uint16 max` assert above already guarantees every real value
  # fits, and this array otherwise lives (at int64) all the way to the end
  # of the function too.
  row_l_per_nz = row_l_per_nz.astype(np.uint16)

  # row_per_nz's only other use was row_l_per_nz just above -- free it now
  # rather than let it sit alive (nnz-sized int64, one of several such
  # arrays that together drove a real host-side OOM at RMAT-s25 scale,
  # docs/ERRORS.md #26) for the rest of the function.
  del row_per_nz

  max_grid_dim = max(faby, fabx)
  del max_grid_dim  # unused now -- was the pure-Python loop's `counted[]` scratch size

  # step 1 (local_nnz): a nonzero's block is fully determined by (row_b,
  # col_b) -- a straight histogram over the flat block id, no per-nonzero
  # branching needed at all.
  block_id_per_nz = row_b_per_nz * fabx + col_b_per_nz
  local_nnz = np.bincount(block_id_per_nz, minlength=faby * fabx).reshape(faby, fabx, 1)
  del block_id_per_nz  # only other use was the bincount just above

  # step 1 (local_nzcols): count of DISTINCT (row_b, col) combinations per
  # block -- the original loop's `counted[row_b] != check_token(=col)` gate
  # counts each (row_b, col) pair at most once per col, i.e. exactly once
  # per distinct (row_b, col_b, col) triple (col_b follows deterministically
  # from col). `col < ncols` always, so `row_b * ncols + col` is a safe
  # unique key -- np.unique's own sort does in one vectorized pass what the
  # scalar `counted[]` scratch array did one element at a time.
  # .astype(np.int64) explicitly on row_b_per_nz -- see its own comment
  # above for why merely wrapping ncols in np.int64(...) is NOT sufficient
  # to force this expression to int64 when row_b_per_nz itself is int32.
  rowb_col_key = row_b_per_nz.astype(np.int64) * ncols + col_per_nz
  # col_per_nz's only other use was rowb_col_key just above; rowb_col_key
  # itself is only consumed by the np.unique() call right below (both
  # nnz-sized int64 -- freeing col_per_nz here also gives np.unique's own
  # internal sort/argsort scratch more headroom to work in, see #26).
  del col_per_nz
  # return_inverse=True was tried here and REVERTED for memory, not
  # correctness (see docs/ERRORS.md #26): numpy's own implementation builds
  # several MORE nnz-sized int64 temporaries internally to compute the
  # inverse mapping (an argsort permutation, a sorted copy, a boolean
  # "new value" mask, a cumulative-sum-based rank, and the inverse
  # permutation itself) -- invisible to any `del` on this side, since they
  # live and die entirely inside numpy's C implementation for the
  # DURATION of that one call. return_counts alone needs a strict subset
  # of that same internal work, so dropping return_inverse and getting the
  # same mapping back via a separate np.searchsorted() call (binary search
  # against the already-sorted, MUCH smaller `unique_rc_key`, not another
  # full-array sort) was the actual fix that got RMAT-s25 scale under the
  # 100GiB per-user cgroup ceiling.
  unique_rc_key, unique_rc_count = np.unique(rowb_col_key, return_counts=True)
  # Safe specifically because every element of rowb_col_key is GUARANTEED
  # to equal some element of unique_rc_key exactly (unique_rc_key is just
  # its own deduplicated, sorted value set) -- searchsorted therefore
  # always lands on an exact match, never a between-values insertion
  # point, making this numerically identical to return_inverse's own
  # mapping.
  unique_rc_inverse = np.searchsorted(unique_rc_key, rowb_col_key)
  del rowb_col_key
  u_row_b = unique_rc_key // ncols
  u_col = unique_rc_key % ncols
  u_col_b = u_col // bx
  u_col_l = u_col - u_col_b * bx
  u_block_id = u_row_b * fabx + u_col_b
  local_nzcols = np.bincount(u_block_id, minlength=faby * fabx).reshape(faby, fabx, 1)

  # step 2 (formerly local_nzrows: distinct local rows touched per block)
  # -- REMOVED entirely (docs/ERRORS.md #26 follow-up), not just cheapened.
  # This used to be computed via a SEPARATE full-nnz sort over a
  # CSR-ordered representation of the matrix, fed by a whole second scipy
  # sparse representation the caller had to build just for this. Tracing
  # every live caller (run_bfs.py, run_bfs.appliance.py -- the only two
  # actually exercised paths; run_graph500.py is a separate, already-
  # broken caller predating the #21 bottom-up swap, explicitly out of
  # scope per #21/#24's own precedent) shows NEITHER ever reads
  # `matrix_info["local_nnz_rows"]`/`["max_local_nnz_rows"]` -- both only
  # ever consume `["local_nnz_cols"]`/`["max_local_nnz_cols"]` (renamed
  # locally to "*_rows" post-#21's argument swap; see each caller's own
  # comment on this). The whole computation -- CSR-sort version or a
  # cheaper CSC-derived version alike -- was provably dead code for every
  # live path, so it's deleted outright rather than merely made cheaper:
  # zero cost beats any cost. If a future caller genuinely needs a
  # distinct-local-row count per block again, see this entry's own git
  # history for how to derive it cheaply from `block_id_per_nz`/
  # `row_l_per_nz` (one more np.unique() call, no second sort or second
  # scipy representation needed) -- don't resurrect the old CSR-ordered
  # version.

  # step 3: compute maximum dimension of Aij
  max_local_nnz = int(local_nnz.max())
  max_local_nnz_cols = int(local_nzcols.max())

  assert (max_local_nnz < np.iinfo(
      np.uint16).max), "LOCAL NUMBER OF NONZEROS WILL OVERFLOW, TRY USING A LARGER FABRIC"
  assert (max_local_nnz_cols < np.iinfo(
      np.uint16).max), "LOCAL NUMBER OF NZCOLS WILL OVERFLOW, TRY USING A LARGER FABRIC"
  # mat_rows_buf stores direct dense row-block indices (row_l, see step 4
  # below) instead of a compact position, so the real bound on its values
  # is `by` (the per-PE dense row-block size, i.e. the kernel's `blk`),
  # already asserted up front, next to where `by` is computed -- see
  # there for why.
  # no data overflows u16, we can convert the data to u16
  local_nnz = local_nnz.astype(np.uint16)
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
  del same_group_as_prev  # last use just above
  group_run_len = np.diff(np.append(group_run_start_idx, nnz))
  pos_rel_rowidx = np.arange(nnz) - np.repeat(group_run_start_idx, group_run_len)

  pos_start_per_nz = A_colloc[(u_row_b, u_col_b, pos_in_block)][unique_rc_inverse]
  del unique_rc_inverse  # last use just above
  pos_rowidx_per_nz = pos_start_per_nz.astype(np.int64) + pos_rel_rowidx
  # row_l_per_nz is already uint16 (downcast right after creation, above) --
  # no more .astype() needed here, unlike before.
  A_rows[(row_b_per_nz, col_b_per_nz, pos_rowidx_per_nz)] = row_l_per_nz

  matrix_info = {}
  matrix_info["nrows"] = nrows  # number of rows of the matrix
  matrix_info["ncols"] = ncols  # number of columns of the matrix
  matrix_info["nnz"] = nnz  # number of nonzeros of the matrix
  matrix_info["max_local_nnz"] = max_local_nnz
  matrix_info["max_local_nnz_cols"] = max_local_nnz_cols
  # No "max_local_nnz_rows"/"local_nnz_rows" keys any more -- see step 2's
  # own comment above (docs/ERRORS.md #26 follow-up): provably unread by
  # every live caller, removed entirely rather than kept as dead weight.
  matrix_info["mat_rows_buf"] = A_rows
  matrix_info["mat_col_loc_buf"] = A_colloc
  matrix_info["mat_col_len_buf"] = A_collen
  matrix_info["mat_col_idx_buf"] = A_colidx
  matrix_info["local_nnz"] = local_nnz
  matrix_info["local_nnz_cols"] = local_nzcols

  return matrix_info
