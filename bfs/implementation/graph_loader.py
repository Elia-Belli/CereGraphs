""" Format-agnostic graph-matrix loading shared by bool_diag_spmv's scripts
  (run_bfs.py, run_bfs.appliance.py, run_graph500.py) --
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
  A = _load_mtx(path) if _is_mtx(path) else _load_edgelist(path)
  # .data is genuinely never read anywhere downstream (see this docstring
  # above) -- mmread's own default float64 (or _load_edgelist's own
  # explicit float64) is 8 bytes/nonzero spent on a value nobody looks at,
  # and that same width then survives every later .tocsr()/.tocsc()/
  # .sorted_indices() copy the calling scripts make (3+ live copies of the
  # whole matrix at once, docs/ERRORS.md #26). uint8 is scipy's smallest
  # accepted numeric dtype and is more than sufficient for an all-1s
  # structural marker -- this was a real, measured contributor to a
  # host-side OOM at RMAT-s25 scale (~1B nonzeros, so ~7GB saved per live
  # copy just from this one change).
  A.data = np.ones(len(A.data), dtype=np.uint8)
  return A


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
  # (dst, src), NOT (src, dst): bool_pe.csl's compute_topdown() walks, per
  # local column c, the *row* list stored for c and marks those rows newly
  # visited when c is in the frontier -- i.e. it computes y = M @ x meaning
  # "row r becomes visited if r has an edge TO some frontier column c",
  # which is ancestor/in-edge reachability relative to M, not descendant/
  # out-edge reachability. Feeding M = A^T here (row=dst, col=src) makes the
  # kernel's native ancestor-of-M computation equal descendant-of-A -- i.e.
  # the standard "vertices reachable via out-edges from source" semantics a
  # directed BFS is expected to have. Verified against real SNAP ground
  # truth (berkstan vertex 546279: forward reaches 459,847, reverse reaches
  # 18; pre-fix, sourcing from 546279 gave 18 -- see docs/ERRORS.md). Irrelevant
  # for RMAT (self-symmetric, doesn't use this loader) and for orkut
  # (symmetrize() is its own transpose-symmetric fixed point, A+A^T ==
  # A^T+A, so this swap is a no-op there either way).
  return coo_matrix((data, (dst, src)), shape=(n, n))


def _open(path, mode):
  return gzip.open(path, mode, encoding="utf-8") if path.endswith(".gz") else open(
      path, mode, encoding="utf-8")
