""" Format-agnostic graph-matrix loading shared by bool_diag_spmv's scripts
  (run_bfs.py, run_single_spmv.py, run_host_driven_bfs.py, run_graph500.py) --
  each used to call scipy.io.mmread(infile_mtx) directly and assume Matrix
  Market. load_graph adds a second format: a SNAP-style edge list (plain
  whitespace/tab-separated "src dst" pairs, '#'-prefixed comment lines,
  optionally gzip-compressed), auto-detected from the file's own content so
  every caller keeps passing whatever --infile_mtx path it's given without
  a separate --format flag.
"""

import gzip
import re

import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix

MTX_MAGIC = "%%MatrixMarket"

# Every SNAP dataset file declares its true vertex count in a comment line
# like "# Nodes: 4847571 Edges: 68993773" -- the authoritative source, unlike
# max(id)+1, which silently drops the highest-numbered vertex if it happens
# to have zero edges (it then never appears in any edge line at all).
NODES_DECLARED_RE = re.compile(r"Nodes:\s*(\d+)")


def load_graph(path):
  """Loads a graph's adjacency matrix from either a Matrix Market file
  (detected by the '%%MatrixMarket' magic line; gzip/bz2 handled
  transparently by scipy.io.mmread itself) or a SNAP-style edge list
  (detected otherwise): plain text, one directed edge 'src dst' per line
  (whitespace-separated, extra trailing columns ignored), '#'-prefixed
  comment lines skipped except for a "Nodes: N" declaration (see
  NODES_DECLARED_RE) used as the vertex count when present, optionally
  gzip-compressed (.gz). Returns a scipy.sparse coo_matrix, structural
  (all-1.0 data -- callers here only ever consume .indptr/.indices via
  .tocsr(), never .data, so real edge weights are neither expected nor
  preserved)."""
  if _is_mtx(path):
    return _load_mtx(path)
  return _load_edgelist(path)


def _is_mtx(path):
  with _open(path, "rt") as f:
    first_line = f.readline()
  return first_line.startswith(MTX_MAGIC)


def _load_mtx(path):
  return mmread(path)


def _load_edgelist(path):
  src, dst = [], []
  declared_n = None
  with _open(path, "rt") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      if line.startswith("#"):
        match = NODES_DECLARED_RE.search(line)
        if match:
          declared_n = int(match.group(1))
        continue
      u, v = line.split()[:2]
      src.append(int(u))
      dst.append(int(v))
  # Prefer the file's own declared count (immune to trailing isolated
  # vertices); max() with the inferred count guards against a malformed/stale
  # declaration that undercounts the ids actually present.
  inferred_n = max(src + dst) + 1 if src else 0
  n = max(declared_n or 0, inferred_n)
  data = np.ones(len(src), dtype=np.float64)
  return coo_matrix((data, (src, dst)), shape=(n, n))


def _open(path, mode):
  return gzip.open(path, mode, encoding="utf-8") if path.endswith(".gz") else open(
      path, mode, encoding="utf-8")
