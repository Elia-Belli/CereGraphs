"""Converts a downloaded SNAP edge-list graph (data/snap/*.txt.gz, see
download_snap_graphs.sh) into Matrix Market format -- util/analyze (the
same balancing step gen_rmat.py's own outputs go through before running on
device) only reads .mtx, it has no edge-list support, so this is the SNAP
counterpart to gen_rmat.py's own mmwrite() call. Loading itself reuses
bfs/bool_diag_spmv/graph_loader.py's SNAP-edge-list auto-detection --
duplicated here as a sys.path import rather than converted to a package,
matching how run_bfs.appliance.py already borrows pieces of run_bfs.py by
copy instead of import (see that file's own module docstring).

Usage: cs_python benchmarks/snap_to_mtx.py <infile.txt.gz> <outfile.mtx>
"""
import os
import sys

from scipy.io import mmwrite

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bfs", "bool_diag_spmv"))
from graph_loader import load_graph  # pylint: disable=wrong-import-position


def main():
  infile, outfile = sys.argv[1], sys.argv[2]
  matrix = load_graph(infile)
  print(f"loaded {infile}: {matrix.shape[0]}x{matrix.shape[1]}, nnz={matrix.nnz}")
  mmwrite(outfile, matrix)
  print(f"wrote {outfile}")


if __name__ == "__main__":
  main()
