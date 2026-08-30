"""Vectorized R-MAT generator (standard GRAPH500 parameters A=.57 B=.19 C=.19 D=.05),
producing a boolean square adjacency matrix in Matrix Market format.

Usage: cs_python gen_rmat.py <scale> <edgefactor> <seed> <outfile.mtx>
Example invocations, seed=0:
  gen_rmat.py 12 16 0 ../data/rmat_s12_e16.mtx   (n=4096,  avg degree ~24)
  gen_rmat.py 12 4  0 ../data/rmat_s12_e4.mtx    (n=4096,  avg degree ~7, sparser)
  gen_rmat.py 14 16 0 ../data/rmat_s14_e16.mtx   (n=16384, avg degree ~26, bigger)
"""
import sys
import numpy as np
from scipy import sparse
from scipy.io import mmwrite

def rmat(scale, edgefactor, seed, A=0.57, B=0.19, C=0.19, D=0.05):
  rng = np.random.default_rng(seed)
  n = 1 << scale
  m = n * edgefactor
  u = np.zeros(m, dtype=np.int64)
  v = np.zeros(m, dtype=np.int64)
  ab = A + B
  abc = A + B + C
  for level in range(scale):
    r = rng.random(m)
    bit = 1 << (scale - 1 - level)
    # quadrant (0,0): nothing to add
    q01 = (r >= A) & (r < ab)      # (0,1): v gets the bit
    q10 = (r >= ab) & (r < abc)    # (1,0): u gets the bit
    q11 = (r >= abc)               # (1,1): both get the bit
    v[q01] += bit
    u[q10] += bit
    u[q11] += bit
    v[q11] += bit
  # symmetrize (undirected graph, standard for BFS benchmarking) and dedupe/remove self-loops
  uu = np.concatenate([u, v])
  vv = np.concatenate([v, u])
  keep = uu != vv
  uu, vv = uu[keep], vv[keep]
  data = np.ones(len(uu), dtype=np.float64)
  A_mat = sparse.coo_matrix((data, (uu, vv)), shape=(n, n)).tocsr()
  A_mat.data[:] = 1.0
  A_mat.sum_duplicates()
  A_mat.data[:] = 1.0
  return A_mat

if __name__ == "__main__":
  scale = int(sys.argv[1])
  edgefactor = int(sys.argv[2])
  seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
  outfile = sys.argv[4]
  A = rmat(scale, edgefactor, seed)
  n = A.shape[0]
  print(f"n={n}, nnz={A.nnz}, avg_degree={A.nnz/n:.2f}, max_degree={A.getnnz(axis=1).max()}")
  mmwrite(outfile, A, field='real')
