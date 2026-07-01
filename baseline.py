"""
bfs_baseline.py
===============
Progressive Python baseline for BFS on the Cerebras WSE.

Stages
------
1. Naive SpMV          -- dense numpy matrix, A @ f (boolean semiring via clipping)
2. COO SpMV            -- sparse (col_idx, row_idx) pairs, inner loop mirrors WSE kernel
3. BFS via naive SpMV  -- level-synchronous loop using stage 1
4. BFS via COO SpMV    -- level-synchronous loop using stage 2, matches WSE design
5. Reference BFS       -- standard queue-based BFS for correctness comparison

All functions share the same small graph so results are directly comparable.
The adjacency convention throughout is A[dst][src] = 1 (column = source vertex).

Output convention (Graph500)
----------------------------
Graph500 validates the BFS tree via the *parent* array, not distances:

    parent[v] = u     u was the first active vertex that discovered v
    parent[source] = source
    parent[v] = -1    v is unreachable

Distances are *not* the benchmark output -- they are derived from parent on the
host in O(N) if needed.  The visited bitmask is an internal algorithm variable
and is not part of the output.

On the WSE the parent array is row-distributed (same slice as f') and written
back to the host after the BFS loop terminates.  A second COO scan per level
(coo_spmv_with_parent) captures the parent in one pass alongside the frontier
update, without extra communication.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Graph fixture
# ---------------------------------------------------------------------------

def make_example_graph() -> Tuple[int, List[Tuple[int, int]]]:
    """
    8-vertex directed graph used throughout our earlier diagrams.

    Edges (src -> dst):
      0->1, 0->4
      1->2, 1->5
      2->3, 2->6
      3->7
      4->5
      5->6
      6->7

    BFS levels from source=0:
      level 0: {0}
      level 1: {1, 4}
      level 2: {2, 5}
      level 3: {3, 6}
      level 4: {7}
    """
    N = 8
    edges = [
        (0, 1), (0, 4),
        (1, 2), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5),
        (5, 6),
        (6, 7),
    ]
    return N, edges


def build_dense_adjacency(N: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build dense adjacency matrix A[dst][src].
    A[i][j] = 1  <=>  edge j -> i exists.
    """
    A = np.zeros((N, N), dtype=np.uint32)
    for src, dst in edges:
        A[dst][src] = 1
    return A


@dataclass
class COOTile:
    """
    Sparse COO representation of the adjacency matrix.
    Sorted by col_idx (source vertex) to match the WSE column-broadcast phase.

    On the WSE each PE holds a COO tile for its local submatrix.
    Here the 'tile' is the full matrix (single-PE baseline).
    """
    col_idx: np.ndarray   # shape (nnz,), dtype uint16 -- source vertex
    row_idx: np.ndarray   # shape (nnz,), dtype uint16 -- destination vertex
    N: int                # number of vertices

    @classmethod
    def from_edges(cls, N: int, edges: List[Tuple[int, int]]) -> "COOTile":
        if not edges:
            return cls(
                col_idx=np.array([], dtype=np.uint16),
                row_idx=np.array([], dtype=np.uint16),
                N=N,
            )
        cols, rows = zip(*edges)            # src, dst
        order = np.argsort(cols)            # sort by source (column) index
        return cls(
            col_idx=np.array(cols, dtype=np.uint16)[order],
            row_idx=np.array(rows, dtype=np.uint16)[order],
            N=N,
        )

    def nnz(self) -> int:
        return len(self.col_idx)


# ---------------------------------------------------------------------------
# Stage 1 -- Naive SpMV (dense, boolean semiring)
# ---------------------------------------------------------------------------

def naive_spmv(A: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    One SpMV step over the boolean (OR, AND) semiring.

    Standard notation:  f' = A @ f
    Boolean semiring:   f'[i] = OR_j ( A[i][j] AND f[j] )

    Using numpy integer matmul then clipping to {0,1} gives the same result:
    any nonzero entry means at least one AND fired true.

    Parameters
    ----------
    A : (N, N) uint32  adjacency matrix, A[dst][src] = 1
    f : (N,)   uint32  frontier bitmask, f[v] = 1 if v is active

    Returns
    -------
    fp : (N,) uint32   raw SpMV output (before visited mask)
    """
    fp = A @ f          # integer matmul; fp[i] > 0 iff any active in-neighbor
    return np.clip(fp, 0, 1).astype(np.uint32)


# ---------------------------------------------------------------------------
# Stage 2 -- COO SpMV (sparse, boolean semiring)
# ---------------------------------------------------------------------------

def coo_spmv(coo: COOTile, f: np.ndarray) -> np.ndarray:
    """
    One SpMV step using COO tile -- mirrors the WSE inner kernel exactly.

    WSE inner kernel (CSL pseudocode):
        for each edge in coo_tile:          // linear scan, col-sorted
            if f_bitmask[edge.col]:         // is source in frontier?
                fp_bitmask[edge.row] = 1   // set destination in output

    On a real WSE PE, f is received via the column broadcast (North port),
    and fp is the partial result sent East for row OR-reduce.
    Here f is the full vector and fp is the full output (single-PE).

    Parameters
    ----------
    coo : COOTile   sparse adjacency, col-sorted
    f   : (N,) uint32  frontier vector

    Returns
    -------
    fp : (N,) uint32  raw SpMV output (before visited mask)
    """
    fp = np.zeros(coo.N, dtype=np.uint32)
    for col, row in zip(coo.col_idx, coo.row_idx):
        if f[col]:                  # AND: is source active?
            fp[row] = 1             # OR:  set destination
    return fp


def coo_spmv_vectorised(coo: COOTile, f: np.ndarray) -> np.ndarray:
    """
    Vectorised COO SpMV -- same result as coo_spmv but uses numpy indexing.
    Useful for larger graphs where the Python loop is too slow.
    Keeps the same logical structure as the scalar version.
    """
    fp = np.zeros(coo.N, dtype=np.uint32)
    active_mask = f[coo.col_idx].astype(bool)   # which edges have active source
    active_rows = coo.row_idx[active_mask]       # their destination vertices
    fp[active_rows] = 1
    return fp


def coo_spmv_with_parent(
    coo: COOTile,
    f: np.ndarray,
    visited: np.ndarray,
    parent: np.ndarray,
) -> np.ndarray:
    """
    COO SpMV that also records the parent vertex in one pass.

    Extends coo_spmv_vectorised: when an edge (col->row) fires and row has
    not yet been assigned a parent, record col as the parent of row.

    This is the kernel the WSE must run -- no extra communication round
    is needed because the parent information is local to each PE tile.
    The parent write is guarded by 'not yet visited AND not yet in fp'
    so only the first firing edge per destination wins (consistent with
    BFS tree semantics).

    On WSE:
      - parent[] lives in the same row-distributed SRAM slice as visited[].
      - The guard 'fp[row] == 0 AND NOT visited[row]' is a local bitmask
        check -- no fabric traffic.
      - parent values (u32 vertex IDs) are written back to the host after
        the BFS terminates via the standard memory copy path.

    Parameters
    ----------
    coo     : COOTile       col-sorted sparse adjacency
    f       : (N,) uint32   current frontier (column-distributed on WSE)
    visited : (N,) uint32   cumulative visited bitmask (row-distributed on WSE)
    parent  : (N,) int32    parent array, modified in place

    Returns
    -------
    fp : (N,) uint32  raw SpMV output (before visited mask)
    """
    fp = np.zeros(coo.N, dtype=np.uint32)
    for col, row in zip(coo.col_idx, coo.row_idx):
        if f[col]:                              # source is active
            if not fp[row] and not visited[row]:
                parent[row] = col               # first firing edge wins
            fp[row] = 1
    return fp


# ---------------------------------------------------------------------------
# Stage 3 -- BFS via naive SpMV
# ---------------------------------------------------------------------------

def bfs_naive_spmv(
    A: np.ndarray,
    source: int,
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Level-synchronous BFS using the dense naive SpMV.

    Loop:
        f'      = A @ f                 (SpMV, boolean semiring)
        f'      = f' AND NOT visited    (mask)
        parent  = record first active in-neighbour for each new vertex
        visited |= f'                   (accumulate)
        f       = f'                    (advance frontier)
        if f' == 0: done

    Returns
    -------
    parent : (N,) int32   BFS parent array (Graph500 output)
                          parent[source]=source, parent[unreachable]=-1
    dist   : (N,) int32   BFS distances, derived from parent for display
    levels : list of lists
    """
    N = A.shape[0]
    parent = np.full(N, -1, dtype=np.int32)
    parent[source] = source
    dist = np.full(N, -1, dtype=np.int32)
    dist[source] = 0

    visited = np.zeros(N, dtype=np.uint32)
    visited[source] = 1

    f = np.zeros(N, dtype=np.uint32)
    f[source] = 1

    levels = [[source]]
    level = 0

    while True:
        fp = naive_spmv(A, f)               # expand frontier
        fp = fp & ~visited                  # mask visited

        if fp.sum() == 0:
            break

        level += 1
        new_verts = np.where(fp)[0]
        for dst in new_verts:
            # find first active in-neighbour from dense row scan
            in_nbrs = np.where(A[dst] & f)[0]
            if len(in_nbrs):
                parent[dst] = int(in_nbrs[0])
            dist[dst] = level

        visited |= fp
        levels.append(new_verts.tolist())
        f = fp

    return parent, dist, levels


# ---------------------------------------------------------------------------
# Stage 4 -- BFS via COO SpMV (matches WSE design)
# ---------------------------------------------------------------------------

def bfs_coo_spmv(
    coo: COOTile,
    source: int,
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """
    Level-synchronous BFS using the COO SpMV kernel with parent tracking.

    This is the direct Python analogue of the WSE algorithm:

      Per level:
        Phase 1  -- broadcast f down columns          (trivial on single PE)
        Phase 2  -- local multiply + parent capture   <-- coo_spmv_with_parent()
        Phase 3  -- row OR-reduce                     (trivial on single PE)
        Phase 4  -- mask AND NOT visited
        Phase 5  -- termination check (popcount)
        Phase 6  -- redistribute f' -> f              (trivial on single PE)

    The parent array is the Graph500 output.  It lives in row-distributed
    SRAM on the WSE (same slice as visited[]) and is written back to the
    host after the BFS loop terminates.

    Parameters
    ----------
    coo    : COOTile  col-sorted sparse adjacency
    source : int      BFS source vertex

    Returns
    -------
    parent : (N,) int32   Graph500 BFS parent array
    dist   : (N,) int32   derived from parent, for display / verification
    levels : list of lists
    """
    N = coo.N
    parent = np.full(N, -1, dtype=np.int32)
    parent[source] = source
    dist = np.full(N, -1, dtype=np.int32)
    dist[source] = 0

    visited = np.zeros(N, dtype=np.uint32)
    visited[source] = 1

    f = np.zeros(N, dtype=np.uint32)
    f[source] = 1

    levels = [[source]]
    level = 0

    while True:
        # -- Phase 1: column broadcast (trivial on single PE)

        # -- Phase 2: local multiply + parent capture
        fp = coo_spmv_with_parent(coo, f, visited, parent)

        # -- Phase 3: row OR-reduce (trivial on single PE)

        # -- Phase 4: mask
        fp = fp & ~visited                  # AND NOT visited

        # -- Phase 5: termination check
        if fp.sum() == 0:
            break

        level += 1
        new_verts = np.where(fp)[0].tolist()
        dist[new_verts] = level
        visited |= fp
        levels.append(new_verts)

        # -- Phase 6: redistribute f' -> f (trivial on single PE)
        f = fp

    return parent, dist, levels


# ---------------------------------------------------------------------------
# Stage 5 -- Reference BFS (queue-based, for correctness comparison)
# ---------------------------------------------------------------------------

def bfs_reference(
    N: int,
    edges: List[Tuple[int, int]],
    source: int,
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """
    Textbook queue-based BFS.  Ground truth for parent and distance.
    Returns parent, dist, levels.
    """
    adj: List[List[int]] = [[] for _ in range(N)]
    for src, dst in edges:
        adj[src].append(dst)

    parent = np.full(N, -1, dtype=np.int32)
    parent[source] = source
    dist = np.full(N, -1, dtype=np.int32)
    dist[source] = 0

    levels: List[List[int]] = [[source]]
    queue = deque([source])

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
                if dist[v] == len(levels):
                    levels.append([])
                levels[dist[v]].append(v)

    return parent, dist, levels


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def print_levels(tag: str, levels: List[List[int]]) -> None:
    print(f"\n{tag}")
    for i, lvl in enumerate(levels):
        print(f"  level {i}: {sorted(lvl)}")


def print_parent(tag: str, parent: np.ndarray) -> None:
    print(f"\n{tag}")
    for v, p in enumerate(parent):
        desc = f"parent={p}" if p != -1 else "unreachable"
        print(f"  v{v}: {desc}")


def validate_parent(
    tag: str,
    parent: np.ndarray,
    source: int,
    N: int,
    edges: List[Tuple[int, int]],
) -> bool:
    """
    Graph500-style parent array validation.

    Rules:
      1. parent[source] == source
      2. For every reached vertex v != source:
           edge (parent[v], v) must exist in the graph
      3. Distances derived from parent must be consistent
         (parent is exactly one level above child)
      4. No unreached vertex appears as a parent
    """
    edge_set = set(edges)
    adj: List[List[int]] = [[] for _ in range(N)]
    for src, dst in edges:
        adj[src].append(dst)

    # derive dist from parent via BFS on the parent tree
    dist = np.full(N, -1, dtype=np.int32)
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in range(N):
            if parent[v] == u and v != source and dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)

    ok = True

    # rule 1
    if parent[source] != source:
        print(f"  [FAIL] {tag}: parent[source] != source")
        ok = False

    for v in range(N):
        if v == source:
            continue
        if parent[v] == -1:
            continue  # unreachable -- skip
        p = parent[v]
        # rule 2: edge must exist
        if (p, v) not in edge_set:
            print(f"  [FAIL] {tag}: edge ({p}->{v}) in parent tree not in graph")
            ok = False
        # rule 3: parent is exactly one level above
        if dist[p] == -1 or dist[v] != dist[p] + 1:
            print(f"  [FAIL] {tag}: level inconsistency at v{v} "
                  f"(dist[parent]={dist[p]}, dist[v]={dist[v]})")
            ok = False

    if ok:
        print(f"  [OK] {tag}: parent array is valid (Graph500 rules)")
    return ok


def assert_equal_parent(
    tag_a: str, parent_a: np.ndarray,
    tag_b: str, parent_b: np.ndarray,
    N: int,
    edges: List[Tuple[int, int]],
) -> None:
    """
    Parent arrays may differ (multiple valid BFS trees exist) so we compare
    the *distances* derived from each, not the arrays element-wise.
    """
    edge_set = set(edges)

    def parent_to_dist(parent, source):
        dist = np.full(N, -1, dtype=np.int32)
        source = int(np.where(parent == np.arange(N))[0][0])
        dist[source] = 0
        changed = True
        while changed:
            changed = False
            for v in range(N):
                p = parent[v]
                if p != -1 and v != source and dist[v] == -1 and dist[p] != -1:
                    dist[v] = dist[p] + 1
                    changed = True
        return dist

    # derive distances
    dist_a = np.full(N, -1, dtype=np.int32)
    dist_b = np.full(N, -1, dtype=np.int32)
    for v in range(N):
        if parent_a[v] == v:
            src = v
    dist_a[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in range(N):
            if parent_a[v] == u and v != src and dist_a[v] == -1:
                dist_a[v] = dist_a[u] + 1
                q.append(v)

    for v in range(N):
        if parent_b[v] == v:
            src = v
    dist_b[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in range(N):
            if parent_b[v] == u and v != src and dist_b[v] == -1:
                dist_b[v] = dist_b[u] + 1
                q.append(v)

    if np.array_equal(dist_a, dist_b):
        print(f"  [OK] dist({tag_a}) == dist({tag_b})")
    else:
        print(f"  [FAIL] dist({tag_a}) != dist({tag_b})")
        print(f"    {tag_a}: {dist_a}")
        print(f"    {tag_b}: {dist_b}")


# ---------------------------------------------------------------------------
# Main: run all stages and compare
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SOURCE = 0
    N, edges = make_example_graph()

    print("=" * 60)
    print(f"Graph: {N} vertices, {len(edges)} edges, source={SOURCE}")
    print("Output: parent array (Graph500 format)")
    print("=" * 60)

    # Build representations
    A   = build_dense_adjacency(N, edges)
    coo = COOTile.from_edges(N, edges)

    print(f"\nDense adjacency matrix A[dst][src]:")
    print(A)
    print(f"\nCOO tile ({coo.nnz()} edges, col-sorted):")
    for c, r in zip(coo.col_idx, coo.row_idx):
        print(f"  edge src={c} -> dst={r}")

    # ── Stage 1: one naive SpMV step ──────────────────────────────────────
    print("\n── Stage 1: naive SpMV (one step from source) ──")
    f0 = np.zeros(N, dtype=np.uint32); f0[SOURCE] = 1
    fp_naive = naive_spmv(A, f0)
    print(f"  f  = {f0}")
    print(f"  f' = {fp_naive}  (vertices discovered: {np.where(fp_naive)[0].tolist()})")

    # ── Stage 2: COO SpMV variants ────────────────────────────────────────
    print("\n── Stage 2: COO SpMV (one step from source) ──")
    fp_coo  = coo_spmv(coo, f0)
    fp_coov = coo_spmv_vectorised(coo, f0)
    print(f"  f' (scalar)     = {fp_coo}")
    print(f"  f' (vectorised) = {fp_coov}")
    assert np.array_equal(fp_naive, fp_coo),  "naive vs COO scalar mismatch"
    assert np.array_equal(fp_naive, fp_coov), "naive vs COO vectorised mismatch"
    print("  [OK] both COO variants match naive SpMV")

    # ── Stage 3: full BFS via naive SpMV ──────────────────────────────────
    print("\n── Stage 3: BFS via naive SpMV ──")
    parent_naive, dist_naive, levels_naive = bfs_naive_spmv(A, SOURCE)
    print_levels("  levels", levels_naive)
    print(f"  dist   = {dist_naive}")
    print(f"  parent = {parent_naive}")

    # ── Stage 4: full BFS via COO SpMV ────────────────────────────────────
    print("\n── Stage 4: BFS via COO SpMV (with parent tracking) ──")
    parent_coo, dist_coo, levels_coo = bfs_coo_spmv(coo, SOURCE)
    print_levels("  levels", levels_coo)
    print(f"  dist   = {dist_coo}")
    print(f"  parent = {parent_coo}")

    # ── Stage 5: reference queue BFS ──────────────────────────────────────
    print("\n── Stage 5: reference queue BFS ──")
    parent_ref, dist_ref, levels_ref = bfs_reference(N, edges, SOURCE)
    print_levels("  levels", levels_ref)
    print(f"  dist   = {dist_ref}")
    print(f"  parent = {parent_ref}")

    # ── Graph500 parent validation ─────────────────────────────────────────
    print("\n── Graph500 parent validation ──")
    validate_parent("naive SpMV BFS", parent_naive, SOURCE, N, edges)
    validate_parent("COO SpMV BFS",   parent_coo,   SOURCE, N, edges)
    validate_parent("reference BFS",  parent_ref,   SOURCE, N, edges)

    # ── Distance consistency ───────────────────────────────────────────────
    print("\n── Distance consistency ──")
    assert_equal_parent("naive SpMV", parent_naive, "reference", parent_ref, N, edges)
    assert_equal_parent("COO SpMV",   parent_coo,   "reference", parent_ref, N, edges)

    # ── Manual level trace showing parent capture ──────────────────────────
    print("\n── Manual level trace (parent captured in COO scan) ──")
    visited = np.zeros(N, dtype=np.uint32)
    visited[SOURCE] = 1
    parent_trace = np.full(N, -1, dtype=np.int32)
    parent_trace[SOURCE] = SOURCE
    f = np.zeros(N, dtype=np.uint32); f[SOURCE] = 1

    for step in range(1, 5):
        fp = coo_spmv_with_parent(coo, f, visited, parent_trace)
        fp = fp & ~visited
        if fp.sum() == 0:
            print(f"  step {step}: frontier empty -> done")
            break
        visited |= fp
        new_verts = np.where(fp)[0].tolist()
        print(f"  step {step}: new={new_verts}  "
              f"parents={[(v, int(parent_trace[v])) for v in new_verts]}")
        f = fp

    print(f"\n  Final parent array: {parent_trace}")
    print(f"  (parent[v]==-1 means unreachable; parent[source]==source)")