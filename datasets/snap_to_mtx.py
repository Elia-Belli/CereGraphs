"""Converts a downloaded SNAP edge-list graph (data/snap/*.txt.gz, see
download_snap_graphs.sh) into Matrix Market format -- util/analyze (the
same balancing step gen_rmat.py's own outputs go through before running on
device) only reads .mtx, it has no edge-list support, so this is the SNAP
counterpart to gen_rmat.py's own mmwrite() call. Loading itself reuses
bfs/bool_diag_spmv/graph_loader.py's SNAP-edge-list auto-detection --
duplicated here as a sys.path import rather than converted to a package,
matching how run_bfs.appliance.py already borrows pieces of run_bfs.py by
copy instead of import (see that file's own module docstring).

Symmetrization is now OPT-IN (--symmetrize), not automatic: checked against
each dataset's own SNAP documentation, only com-orkut is actually
undirected ("com-orkut.ungraph"); berkstan/pokec/topcats/livejournal are
all genuinely DIRECTED (web-BerkStan: hyperlinks, 25% pattern symmetry;
Pokec: "friendships ... are oriented"; wiki-topcats; soc-LiveJournal1).
Blanket-symmetrizing all five (an earlier version of this script) silently
fabricated edges that don't exist in the real graph for the four directed
ones -- for berkstan specifically it roughly DOUBLED nnz (7.6M -> 13.3M),
which was enough on its own to push the matrix over the appliance's h2d
transfer-size ceiling. Use --symmetrize only for datasets that are
genuinely undirected; balance the rest with util/analyze's --shared-perm
(not --symmetric) to get vertex-identity-preserving balancing without
requiring (or fabricating) structural symmetry.

Usage: cs_python datasets/snap_to_mtx.py <infile.txt.gz> <outfile.mtx> [--symmetrize]
"""
import os
import sys

from scipy.io import mmwrite
from scipy.sparse import coo_matrix, eye

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bfs", "bool_diag_spmv"))
from graph_loader import load_graph  # pylint: disable=wrong-import-position


def symmetrize(matrix):
  """A | A^T (structural union), self-loops removed -- returns a boolean
  (0.0/1.0 float) coo_matrix, same convention load_graph's own edge-list
  path already uses (structural-only, no real edge weights). Only call this
  for datasets that are ACTUALLY undirected -- see module docstring."""
  sym = (matrix + matrix.transpose()).tocsr()
  sym = sym - sym.multiply(eye(sym.shape[0], format="csr"))  # drop self-loops
  sym.eliminate_zeros()
  sym.data[:] = 1.0
  return coo_matrix(sym)


def main():
  infile, outfile = sys.argv[1], sys.argv[2]
  do_symmetrize = "--symmetrize" in sys.argv[3:]
  matrix = load_graph(infile)
  print(f"loaded {infile}: {matrix.shape[0]}x{matrix.shape[1]}, nnz={matrix.nnz}")
  if do_symmetrize:
    matrix = symmetrize(matrix)
    print(f"symmetrized: {matrix.shape[0]}x{matrix.shape[1]}, nnz={matrix.nnz}")
  mmwrite(outfile, matrix)
  print(f"wrote {outfile}")


if __name__ == "__main__":
  main()
