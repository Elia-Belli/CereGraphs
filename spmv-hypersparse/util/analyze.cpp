/*
 * Read a sparse matrix A from a mtx file and a 2D grid of dimension
 *fabx-by-faby,
 * find a pair of permutation vectors P, Q such that
 *    A(P,Q) has a load balance distribution in the 2D grid
 *
 * The block size of each PE is bx-by-by
 * where
 *    bx = ceil(nrows/fabx)
 *    by = ceil(ncols/faby)
 *
 * Aij is the submatix in PE(px=j, py=i)
 *
 * Load balance algorithm solves
 *    (P,Q) = argmin{ var(nnz(Bij)): B = A(P,Q)}
 *
 * How to compile
 *   g++ -Wall -g -I include -std=c++17 -O3 -g -c analyze.cpp
 *   gcc -c mmio.c
 *   g++ -Wall -g -I include -std=c++17 -O3 -g -o analyze analyze.o mmio.o
 *
 * How to run: suppose A is distributed into 5-by-4 grid
 *   ./analyze --matrix <path to input file>  --rand 0 --fabx 5 --faby 4
 *--omatrix <output filename>
 */

#include <argparse/argparse.hpp>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <unordered_set>
#include <vector>

#include "mmio.h"

using idx_t = int64_t;
/* first index is is col, second index is row */
using edge_t = std::tuple<idx_t, idx_t>;

/* Given a sparse matrix A with dimension nrows-by-ncols, nnz nonzeros,
 * and permutation vector P and Q, output A(P,Q) in MTX format
 *
 * The MTX is 1-based
 */
void write_matrix(
    const std::string &matname, /* mtx file */
    int nrows,                  /* number of rows of the matrix A */
    int ncols,                  /* number of columns of the matrix A */
    int nnz,                    /* number of nonzeros of the matrix A */
    const int *cooRowInd,       /* nonzero row indices of the matrix A */
    const int *cooColInd,       /* nonzero column indices of the matrix A */
    const double *cooVal,       /* nonzero values of the matrix A */
    const std::vector<idx_t> &iperm_r, /* inverse of the permutation vector P */
    const std::vector<idx_t> &iperm_c  /* inverse of the permutation vector Q */
) {
  std::ofstream outfile;
  outfile.open(matname, std::ios::out);

  outfile << "%%MatrixMarket matrix coordinate real general" << std::endl;
  outfile << nrows << " ";
  outfile << ncols << " ";
  outfile << nnz << std::endl;

  for (idx_t k = 0; k < nnz; k++) {
    /* (r, c) is the index of A */
    const int r = cooRowInd[k];
    const int c = cooColInd[k];
    const double Aij = cooVal[k];
    /* (row, col) is the index of B=A(P,Q) */
    const int row = iperm_r[r];
    const int col = iperm_c[c];
    /* 1-based */
    outfile << row + 1 << " " << col + 1 << " " << Aij << std::endl;
  }
}

/*
 * Given a sparse matrix A described by the graph "edges",
 * remove duplicated entries because SpMV requires unique entries
 */
void remove_duplicates(std::vector<edge_t> &edges) {
  // sort edge list by row index then by col index
  std::sort(edges.begin(), edges.end(), [](edge_t const &l, edge_t const &r) {
    if (std::get<1>(l) == std::get<1>(r))
      return std::get<0>(l) < std::get<0>(r);
    else
      return std::get<1>(l) < std::get<1>(r);
  });
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  // cleanup self edges too
  edges.erase(std::remove_if(edges.begin(), edges.end(),
                             [](edge_t const &l) {
                               return std::get<1>(l) == std::get<0>(l);
                             }),
              edges.end());
}

/* Given a sparse matrix A with dimension nrows-by-ncols,
 * count nonzeros for each row and nonzeros for each column
 *
 * nzrows[i] = number of nonzeros at i-th row
 * nzcols[j] = number of nonzeros at j-th column
 */
void count_nz(const std::vector<edge_t> &edges, idx_t nrows, idx_t ncols,
              std::vector<std::pair<idx_t, int>> &nzrows,
              std::vector<std::pair<idx_t, int>> &nzcols) {
  nzrows.assign(nrows, std::make_pair<idx_t, int>(0, 0));
  for (idx_t i = 0; i < nrows; i++) {
    nzrows[i].first = i;
  }
  nzcols.assign(ncols, std::make_pair<idx_t, int>(0, 0));
  for (idx_t i = 0; i < ncols; i++) {
    nzcols[i].first = i;
  }
  for (size_t i = 0; i < edges.size(); i++) {
    auto &e = edges[i];
    nzrows[std::get<1>(e)].second++;
    nzcols[std::get<0>(e)].second++;
  }

#if SANITY_CHECK
  size_t totcols = 0;
  for (idx_t i = 0; i < ncols; i++) {
    totcols += nzcols[i].second;
  }
  size_t totrows = 0;
  for (idx_t i = 0; i < nrows; i++) {
    totrows += nzrows[i].second;
  }
  assert(totcols = edges.size());
  assert(totrows = edges.size());
#endif
}

/*
 * Given a sparse matrix A described by the graph "edges",
 * construct the CSR representation
 */
void convert_to_csr(std::vector<edge_t> &edges, idx_t nrows,
                    std::vector<std::pair<idx_t, int>> &nzrows,
                    std::vector<idx_t> &rowptr, std::vector<idx_t> &colidx) {
  size_t nnz = edges.size();
  colidx.resize(nnz);
  rowptr.resize(nrows + 1);
  rowptr[0] = 0;
  // do a partial sum
  for (size_t i = 0; i < nzrows.size(); i++) {
    /* nzrows[i] is number of nonzeros in i-th row */
    rowptr[i + 1] = rowptr[i] + nzrows[i].second;
  }

  std::vector<idx_t> heads = rowptr;
  for (size_t i = 0; i < edges.size(); i++) {
    auto &e = edges[i];
    auto r = std::get<1>(e);
    auto c = std::get<0>(e);
    colidx[heads[r]++] = c;
  }
}

/*
 * Given a sparse matrix A described by the graph "edges",
 * construct the CSR representation
 */
void convert_to_csc(std::vector<edge_t> &edges, idx_t ncols,
                    std::vector<std::pair<idx_t, int>> &nzcols,
                    std::vector<idx_t> &colptr, std::vector<idx_t> &rowidx) {
  size_t nnz = edges.size();
  rowidx.resize(nnz);
  colptr.resize(ncols + 1);
  colptr[0] = 0;
  // do a partial sum
  for (size_t i = 0; i < nzcols.size(); i++) {
    /* nzcols[j] is number of nonzeros in j-th column */
    colptr[i + 1] = colptr[i] + nzcols[i].second;
  }

  std::vector<idx_t> heads = colptr;
  for (size_t i = 0; i < edges.size(); i++) {
    auto &e = edges[i];
    auto r = std::get<1>(e);
    auto c = std::get<0>(e);
    rowidx[heads[c]++] = r;
  }
}

void distribute(const std::vector<idx_t> &rowptr,
                const std::vector<idx_t> &colidx, idx_t bx, idx_t by, int fabx,
                int faby, std::vector<int> &buckets) {
  buckets.assign(fabx * faby, 0);
  for (idx_t r = 0; r < (idx_t)rowptr.size() - 1; r++) {
    idx_t row_b = r / by;
    for (auto colp = rowptr[r]; colp < rowptr[r + 1]; colp++) {
      auto c = colidx[colp];
      idx_t col_b = c / bx;
      int pidx = (col_b % fabx) * faby + (row_b % faby);
      buckets[pidx]++;
    }
  }
}

void distribute_permute(const std::vector<idx_t> &rowptr,
                        const std::vector<idx_t> &colidx, idx_t bx, idx_t by,
                        int fabx, int faby, const std::vector<idx_t> &iperm_r,
                        const std::vector<idx_t> &iperm_c,
                        std::vector<int> &buckets) {
  buckets.assign(fabx * faby, 0);
  for (idx_t r = 0; r < (idx_t)rowptr.size() - 1; r++) {
    idx_t row_b = iperm_r[r] / by;
    for (auto colp = rowptr[r]; colp < rowptr[r + 1]; colp++) {
      auto c = colidx[colp];
      idx_t col_b = iperm_c[c] / bx;
      int pidx = (col_b % fabx) * faby + (row_b % faby);
      buckets[pidx]++;
    }
  }
}

void iterative_load_balance(
    const std::vector<idx_t> &rowptr, const std::vector<idx_t> &colidx,
    const std::vector<idx_t> &colptr, const std::vector<idx_t> &rowidx,
    idx_t nrows, idx_t ncols, int fabx, int faby,
    std::vector<std::pair<idx_t, int>> &nzrows,
    std::vector<std::pair<idx_t, int>> &nzcols, std::vector<idx_t> &iperm_r,
    std::vector<idx_t> &iperm_c, std::vector<int> &buckets) {

  int bx = std::ceil((float)ncols / ((float)fabx));
  int by = std::ceil((float)nrows / ((float)faby));

  // sort nzrows and nzcols
  std::sort(nzrows.begin(), nzrows.end(),
            [](std::pair<idx_t, int> const &l, std::pair<idx_t, int> const &r) {
              return l.second > r.second;
            });
  std::sort(nzcols.begin(), nzcols.end(),
            [](std::pair<idx_t, int> const &l, std::pair<idx_t, int> const &r) {
              return l.second > r.second;
            });

  idx_t nnz_total = colidx.size();
  iperm_r.resize(nrows);
  std::iota(iperm_r.begin(), iperm_r.end(), 0);
  iperm_c.resize(ncols);
  std::iota(iperm_c.begin(), iperm_c.end(), 0);

  // initial distrib
  distribute_permute(rowptr, colidx, bx, by, fabx, faby, iperm_r, iperm_c,
                     buckets);

  std::vector<idx_t> newiperm_r(nrows);
  std::vector<idx_t> newiperm_c(ncols);

  idx_t nnz_max = (idx_t)*std::max_element(buckets.begin(), buckets.end());
  double npes = fabx * faby;
  double cur_lb = 0;
  double prev_lb = (double)nnz_total / (npes * (double)nnz_max);
  std::vector<std::tuple<int, int, bool>> rowload(faby);
  std::vector<int> rowcnt(faby, 0);

  do {
    {

#if SANITY_CHECK
      newiperm_r.assign(nrows, -1);
#endif
      rowcnt.assign(faby, 0);
      buckets.assign(fabx * faby, 0);
      rowload.assign(faby, std::tuple<int, int, bool>(0, 0, false));

      // choose new row mapping
      // find the processor row on which the heaviest loaded PE is loaded least
      int minpy = 0;

      // for each row in order of decreasing nnz in the row, choose a processor
      // row so that the heaviest loaded PE is loaded least, given the current
      // column distribution
      for (idx_t i = 0; i < nrows; i++) {
        auto row = nzrows[i].first;

// row should go on minpy;
#if SANITY_CHECK
        assert(rowcnt[minpy] < by);
#endif
        int newrowidx = minpy * by + rowcnt[minpy];
        newiperm_r[row] = newrowidx;
        rowcnt[minpy]++;
        if (minpy * by + rowcnt[minpy] >=
            std::min(idx_t((minpy + 1) * by), nrows))
          std::get<2>(rowload[minpy]) = true;

        // update the load matrix
        for (auto ptr = rowptr[row]; ptr < rowptr[row + 1]; ptr++) {
          auto permcol = iperm_c[colidx[ptr]];
          idx_t px = (permcol / bx) % fabx;
          buckets[px * faby + minpy]++;
        }

        // now check the load in the proc row
        auto maxpx = -1;
        int maxloadx = 0;
        for (int px = 0; px < fabx; px++) {
          int load = buckets[px * faby + minpy];
          if (load >= maxloadx) {
            maxloadx = load;
            maxpx = px;
          }
        }
        std::get<0>(rowload[minpy]) = maxpx;
        std::get<1>(rowload[minpy]) = maxloadx;

        auto it = std::min_element(
            rowload.begin(), rowload.end(),
            [](std::tuple<int, int, bool> &l, std::tuple<int, int, bool> &r) {
              if (std::get<2>(l)) {
                return false;
              } else if (std::get<2>(r)) {
                return true;
              } else {
                return std::get<1>(l) < std::get<1>(r);
              }
            });
        minpy = std::distance(rowload.begin(), it);
#if SANITY_CHECK
        if (i < nrows - 1)
          assert(rowcnt[minpy] < by);
#endif
      }
#if SANITY_CHECK
      for (idx_t i = 0; i < nrows; i++) {
        assert(newiperm_r[i] >= 0);
      }
      idx_t tot_buckets = 0;
      for (auto b : buckets) {
        tot_buckets += b;
      }
      assert(tot_buckets == nnz_total);
#endif
    }

    // A graph's adjacency matrix needs a single permutation shared by rows
    // and columns (row i and column i must stay the same vertex) so that a
    // source vertex survives balancing -- reuse the row assignment just
    // computed above instead of optimizing an independent column
    // permutation. (There used to be an independent-column-optimization
    // branch here for callers that didn't need vertex identity preserved;
    // it was removed since every caller now requires --symmetric or
    // --shared-perm, which both need this sharing.)
    newiperm_c = newiperm_r;

    distribute_permute(rowptr, colidx, bx, by, fabx, faby, newiperm_r,
                       newiperm_c, buckets);
    nnz_max = (idx_t)*std::max_element(buckets.begin(), buckets.end());
    prev_lb = cur_lb;
    cur_lb = (double)nnz_total / (npes * (double)nnz_max);
    std::cout << "new load: " << cur_lb << " vs " << prev_lb << std::endl;

    if (cur_lb > prev_lb) {
      iperm_r.swap(newiperm_r);
      iperm_c.swap(newiperm_c);
    }
  } while (cur_lb > prev_lb);
  // final distrib
  distribute_permute(rowptr, colidx, bx, by, fabx, faby, iperm_r, iperm_c,
                     buckets);
}

int main(int argc, char *argv[]) {

  argparse::ArgumentParser program("load balance analysis tool");

  program.add_argument("--matrix")
      .help("path to the input matrix (MTX format)")
      .required()
      .default_value(std::string(""));

  program.add_argument("--omatrix")
      .help("filename of output matrix A(P,Q)")
      .default_value(std::string(""));

  program.add_argument("--fabx")
      .help("size of the core rectangle in the x dimension")
      .scan<'i', int>()
      .default_value(int(0));

  program.add_argument("--faby")
      .help("size of the core rectangle in the y dimension")
      .scan<'i', int>()
      .default_value(int(0));

  program.add_argument("--rand")
      .help("random permutation")
      .scan<'i', int>()
      .default_value(int(0));

  program.add_argument("--remove_dup")
      .help("remove duplicate edges")
      .default_value(false)
      .implicit_value(true);

  program.add_argument("--symmetric")
      .help("treat the matrix as a symmetric (undirected graph) adjacency "
            "matrix and force a single shared row/column permutation, so "
            "row i and column i keep referring to the same vertex after "
            "balancing. REFUSES to run if the matrix isn't actually "
            "structurally symmetric (A == A^T) -- use --shared-perm instead "
            "for a genuinely directed graph that still needs vertex-identity- "
            "preserving balancing")
      .default_value(false)
      .implicit_value(true);

  program.add_argument("--shared-perm")
      .help("like --symmetric's row/column-permutation-sharing behavior "
            "(needed for --operm to have a single well-defined answer), but "
            "WITHOUT requiring the input matrix to be structurally "
            "symmetric -- for a genuinely directed graph (e.g. a web/"
            "hyperlink graph) that should be balanced as directed, not "
            "symmetrized, but still needs a coherent vertex identity across "
            "row/column roles for a canonical BFS source vertex to survive "
            "balancing. Mutually exclusive with --symmetric (which also "
            "implies this sharing, plus the structural-symmetry check).")
      .default_value(false)
      .implicit_value(true);

  program.add_argument("--operm")
      .help("write the balancing permutation to this file (one line per "
            "original vertex id i, containing its balanced-matrix vertex "
            "id, i.e. line i+1 holds iperm_r[i]) -- lets a caller translate "
            "a known original-graph vertex (e.g. a canonical BFS source) "
            "into its corresponding vertex id in the balanced matrix")
      .default_value(std::string(""));

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  auto matrix = program.get<std::string>("--matrix");
  auto omatrix = program.get<std::string>("--omatrix");

  auto fabx = program.get<int>("--fabx");
  auto faby = program.get<int>("--faby");
  auto randperm = program.get<int>("--rand");
  auto symmetric = program.get<bool>("--symmetric");
  auto shared_perm = program.get<bool>("--shared-perm");
  auto operm = program.get<std::string>("--operm");

  if (symmetric && shared_perm) {
    std::cerr << "--symmetric and --shared-perm are mutually exclusive "
                 "(--symmetric already implies --shared-perm's row/column "
                 "sharing, plus the structural-symmetry check --shared-perm "
                 "deliberately skips)" << std::endl;
    std::exit(1);
  }
  if (!symmetric && !shared_perm) {
    std::cerr << "one of --symmetric or --shared-perm is required: "
                 "balancing always preserves a single shared row/column "
                 "vertex identity now, so a caller must declare whether the "
                 "input is a verified structurally-symmetric graph "
                 "(--symmetric) or a directed graph that still needs a "
                 "coherent source vertex (--shared-perm)" << std::endl;
    std::exit(1);
  }
  // share_row_col_perm: true whenever row i and column i must keep
  // referring to the same vertex after balancing -- both --symmetric and
  // --shared-perm want this (now mandatory), they only differ on whether
  // the input matrix is required to already be structurally symmetric.
  bool share_row_col_perm = symmetric || shared_perm;

  /* size of the core rectangle must be positive */
  assert(0 < fabx);
  assert(0 < faby);

  std::cout << "randperm = " << randperm << std::endl;

  std::cout << "(1) Reading mtx file " << matrix << std::endl;
  int m, n, nnz;
  int *cooRowInd = NULL;
  int *cooColInd = NULL;
  double *cooVal = NULL;

  /* (cooRowInd, cooColInd, cooVal) is 0-based */
  int ret = mm_read_unsymmetric_sparse(matrix.c_str(), &m, &n, &nnz, &cooVal,
                                       &cooRowInd, &cooColInd);
  assert(0 == ret);

  idx_t nrows = m;
  idx_t ncols = n;
  std::vector<edge_t> edges(nnz);
  for (int i = 0; i < nnz; i++) {
    /* edge = <col, row> */
    edges[i] = std::make_tuple(cooColInd[i], cooRowInd[i]);
  }

  if (program["--remove_dup"] == true) {
    std::cout << "Removing duplicate edges from " << matrix << std::endl;
    remove_duplicates(edges);
  }

  if (share_row_col_perm) {
    if (m != n || fabx != faby) {
      std::cerr << "--symmetric/--shared-perm require a square matrix and a "
                   "square PE grid (a single shared row/column permutation "
                   "only makes sense when rows and columns share an index "
                   "space): got " << m << "x" << n << " matrix, " << fabx
                << "x" << faby << " grid" << std::endl;
      std::exit(1);
    }
  }
  if (symmetric) {
    // verify A == A^T structurally: edge (col,row) must have a matching
    // (row,col) counterpart, i.e. the edge set is closed under swap. Only
    // required for --symmetric -- --shared-perm deliberately allows a
    // genuinely directed (non-symmetric) matrix through, since it shares
    // the row/column permutation for vertex-identity reasons only, not to
    // claim the graph is undirected.
    std::vector<edge_t> fwd = edges;
    std::vector<edge_t> rev(edges.size());
    for (size_t i = 0; i < edges.size(); i++) {
      rev[i] = std::make_tuple(std::get<1>(edges[i]), std::get<0>(edges[i]));
    }
    std::sort(fwd.begin(), fwd.end());
    std::sort(rev.begin(), rev.end());
    if (fwd != rev) {
      std::cerr << "--symmetric was requested but " << matrix
                << " is not structurally symmetric (A != A^T) -- refusing "
                   "to balance it as an undirected graph. If it's genuinely "
                   "directed and you still want vertex-identity-preserving "
                   "balancing, use --shared-perm instead." << std::endl;
      std::exit(1);
    }
  }

  float num_block = 1;

  if (nrows < faby)
    faby = nrows;
  if (ncols < fabx)
    fabx = ncols;

  const idx_t bx = std::ceil((float)ncols / (num_block * (float)fabx));
  const idx_t by = std::ceil((float)nrows / (num_block * (float)faby));
  /* (nrows, ncols) expands the matrix */
  ncols = fabx * bx;
  nrows = faby * by;

#if SANITY_CHECK
  idx_t tot_nnz = edges.size();
#endif
  std::cout << "Nnz = " << edges.size() << std::endl;

  std::cout << "(number of rows of the matrix ) Nrows = " << nrows << std::endl;
  std::cout << "(number of columns of the matrix) Ncols = " << ncols
            << std::endl;

  std::cout << "(2) count nnz for each row and column" << std::endl;

  std::vector<std::pair<idx_t, int>> nzrows, nzcols;

  count_nz(edges, nrows, ncols, nzrows, nzcols);

  std::cout << "(3) convert COO to CSR" << std::endl;
  std::vector<idx_t> rowptr;
  std::vector<idx_t> colidx;
  convert_to_csr(edges, nrows, nzrows, rowptr, colidx);

  std::cout << "(4) convert COO to CSC" << std::endl;
  std::vector<idx_t> colptr;
  std::vector<idx_t> rowidx;
  convert_to_csc(edges, ncols, nzcols, colptr, rowidx);

  {
    std::vector<edge_t> tmp;
    // make sure edges is cleared
    edges.swap(tmp);
  }

  /*
   * The permuation vector P and Q are found by load balance algorithm such that
   *     B = A(P,Q) has small variation of nonzeros per PE
   * B(i,j) = A(P(i), Q(j))
   *
   * where P = perm_r and Q = perm_c
   */
  std::vector<idx_t> iperm_r(nrows);
  std::vector<idx_t> iperm_c(ncols);
  std::iota(iperm_r.begin(), iperm_r.end(), 0);
  std::iota(iperm_c.begin(), iperm_c.end(), 0);
  std::vector<idx_t> perm_r(nrows);
  std::iota(perm_r.begin(), perm_r.end(), 0);
  std::vector<idx_t> perm_c(ncols);
  std::iota(perm_c.begin(), perm_c.end(), 0);

  std::cout << "(5) distribute CSR to buckets" << std::endl;

  std::cout << "fabx = " << fabx << std::endl;
  std::cout << "faby = " << faby << std::endl;
  std::cout << "ncols = " << ncols << std::endl;
  std::cout << "nrows = " << nrows << std::endl;

  std::vector<int> buckets;
  distribute(rowptr, colidx, bx, by, fabx, faby, buckets);

  std::cout << "buckets.size = " << buckets.size() << std::endl;

  int max_nnz = *std::max_element(buckets.begin(), buckets.end());
  std::cout << "Block size " << bx << "-by-" << by << std::endl;
  std::cout << "Max loaded PE has " << max_nnz << " nz" << std::endl;
#if SANITY_CHECK
  idx_t tot_buckets = 0;
  for (auto b : buckets) {
    tot_buckets += b;
  }
  assert(tot_buckets == tot_nnz);
#endif
  std::cout << "Buckets: {" << std::endl << "px, py, nnz" << std::endl;
  for (size_t i = 0; i < buckets.size(); i++) {
    auto px = i / faby;
    auto py = i % faby;
    auto nnz = buckets[i];
    std::cout << px << ", " << py << ", " << nnz << std::endl;
  }
  std::cout << "}" << std::endl;

  if (randperm) {
    std::cout << "(6) random permutation" << std::endl;

    // first generate an identity perm vec
    std::random_shuffle(perm_r.begin(), perm_r.end());
    // now generate the inverse perm
    for (idx_t i = 0; i < (idx_t)perm_r.size(); i++) {
      iperm_r[perm_r[i]] = i;
    }

    // keep row and column permutations identical, so this diagnostic
    // random baseline stays a valid graph relabeling too (balancing always
    // shares the row/column permutation now).
    iperm_c = iperm_r;

    distribute_permute(rowptr, colidx, bx, by, fabx, faby, iperm_r, iperm_c,
                       buckets);

    int max_nnz = *std::max_element(buckets.begin(), buckets.end());
    std::cout << "Permuted Max loaded PE has " << max_nnz << " nz" << std::endl;
    std::cout << "Buckets: { ";
    for (auto b : buckets) {
      std::cout << b << " ";
    }
    std::cout << "}" << std::endl;
  }

  std::cout << "(7) iterate the load-balance partition" << std::endl;
  auto sorted_nzcols = nzcols;
  auto sorted_nzrows = nzrows;

  iterative_load_balance(rowptr, colidx, colptr, rowidx, nrows, ncols, fabx,
                         faby, sorted_nzrows, sorted_nzcols, iperm_r, iperm_c,
                         buckets);
  max_nnz = *std::max_element(buckets.begin(), buckets.end());
  std::cout << "Permuted Max loaded PE has " << max_nnz << " nz" << std::endl;
#if SANITY_CHECK
  idx_t tot_buckets = 0;
  for (auto b : buckets) {
    tot_buckets += b;
  }
  assert(tot_buckets == tot_nnz);
#endif
  std::cout << "Buckets: {" << std::endl << "px, py, nnz" << std::endl;
  for (size_t i = 0; i < buckets.size(); i++) {
    auto px = i / faby;
    auto py = i % faby;
    auto nnz = buckets[i];
    std::cout << px << ", " << py << ", " << nnz << std::endl;
  }
  std::cout << "}" << std::endl;

  /* construct the permutation vector */
  for (idx_t i = 0; i < (idx_t)iperm_r.size(); i++) {
    perm_r[iperm_r[i]] = i;
  }
  for (idx_t i = 0; i < (idx_t)iperm_c.size(); i++) {
    perm_c[iperm_c[i]] = i;
  }

  if (omatrix != "") {
    std::cout << "(8) output A(P,Q) to a mtx file " << omatrix << std::endl;
    write_matrix(omatrix, nrows, ncols, nnz, cooRowInd, cooColInd, cooVal,
                 iperm_r, iperm_c);
  }

  if (operm != "") {
    std::cout << "(9) output balancing permutation to " << operm << std::endl;
    std::ofstream permfile;
    permfile.open(operm, std::ios::out);
    // --symmetric or --shared-perm guaranteed iperm_c == iperm_r above, so
    // iperm_r alone fully describes the (single, shared) vertex mapping. One line per
    // ORIGINAL vertex id (0-based, matching the input SNAP/edge-list
    // numbering), each line holding that vertex's id in the balanced
    // matrix written above (also 0-based, unlike write_matrix's own
    // 1-based MTX output -- this file is meant for direct use as a
    // --source value, which is 0-based).
    for (idx_t i = 0; i < (idx_t)iperm_r.size(); i++) {
      permfile << iperm_r[i] << std::endl;
    }
    permfile.close();
  }

  if (NULL != cooRowInd) {
    free(cooRowInd);
  }
  if (NULL != cooColInd) {
    free(cooColInd);
  }
  if (NULL != cooVal) {
    free(cooVal);
  }

  return 0;
}
