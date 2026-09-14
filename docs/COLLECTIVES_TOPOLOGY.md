# `collectives_2d`: communication topology (chain vs. tree)

**Verdict: chain-based, not tree-based.** Every collective primitive in
`collectives_2d` (broadcast, `reduce_fadds`, `reduce_or`) is implemented as
a **1-D bidirectional linear-chain relay** along a single row or column.
(Two other variants, `reduce_select_any` and `reduce_select_any_indexed`,
used to live here too and shared this same topology -- both were removed
entirely, see `ERRORS.md` #24; this doc keeps a brief note on them in the
Variant differences section below for historical context, but the
line-numbered source citations for their deleted code have been dropped
since they no longer exist.) There is no branching —
each PE has at most two communication partners (its `POS_DIR` and `NEG_DIR`
neighbor) — so there is no O(log n)-depth structure anywhere in this library.
Full 2-D collectives are built by composing two sequential 1-D chain phases
(row, then column), giving an O(√N) pattern for an N-PE grid, not a true
global tree.

This matters for any O-complexity claim in reports/papers about the BFS
kernel: **collective cost scales as O(P) hops per row/column (P = PEs along
that dimension)**, not O(log P).

## Source locations

- Canonical/vendor copy (unused elsewhere in this repo; only has
  `broadcast`/`reduce_fadds`, plus unused `scatter`/`gather`):
  [`collectives_2d/pe.csl`](../../../../collectives_2d/pe.csl),
  [`collectives_2d/params.csl`](../../../../collectives_2d/params.csl),
  [`collectives_2d/ctrl_wavelet.csl`](../../../../collectives_2d/ctrl_wavelet.csl)
- Live, extended fork actually used by the BFS kernel (adds `reduce_or`;
  `scatter`/`gather` deleted, see `ERRORS.md` #18; `reduce_select_any`/
  `reduce_select_any_indexed` also added then later deleted, see
  `ERRORS.md` #24 -- current file has only `broadcast`/`reduce_fadds`/
  `reduce_or`):
  [`src/collectives_2d/pe.csl`](../src/collectives_2d/pe.csl) (1044 lines),
  [`src/collectives_2d/params.csl`](../src/collectives_2d/params.csl),
  [`src/collectives_2d/ctrl_wavelet.csl`](../src/collectives_2d/ctrl_wavelet.csl)
- Caller / usage site: [`src/bool_pe.csl`](../src/bool_pe.csl)
- Design/error log: [`ERRORS.md`](ERRORS.md)

## How the chain topology works

Each `collectives_2d` module instance operates along **one dimension at a
time** — either a row (EAST/WEST, `"x"`) or a column (NORTH/SOUTH, `"y"`).
This is configured in `params.csl:47-93` (`DimParams.POS_DIR`/`NEG_DIR` set to
EAST/WEST or SOUTH/NORTH) and used in `pe.csl:38-39,68-69`. A PE's only two
possible communication partners for a given operation are its immediate
neighbor in `POS_DIR` and its immediate neighbor in `NEG_DIR` — strict
nearest-neighbor relay, never multi-child fan-out.

### Broadcast

`configure_broadcast_network`, `pe.csl:448-491` (diagram in the comment just
above it, `pe.csl:~430-447`): the root's data is routed out via
switch-configured routes toward one or both directions (EAST and/or WEST);
each intermediate PE either passes the wavelet straight through (mid-chain)
or terminates it locally (own compute core reads it). Fan-out from the root
is at most "2 directions," after which propagation is strictly linear in
each direction — not a branching tree.

### Reduce family

`reduce_fadds` and `reduce_or` share the same alternating-two-color
linear-chain pattern, diagrammed explicitly in source at `pe.csl:519-521`
(and again, for the teardown side, at `pe.csl:364-366`):

```
 C0  +----+  C1  +----+  C0  +----+  C1  +----+  C0  +----+  C1
---- | P0 | ---- | P1 | ---- | P2 | ---- | P3 | ---- | P4 | ----
     +----+      +----+      +----+      +----+      +----+
```

Data flows from both chain ends (`pe_id == 0` and `pe_id == NUM_PES-1`)
toward `root`. Every intermediate PE receives from one side, combines with
its own local value, and forwards one hop toward the root:

- `transfer_data_reduce()` (dense sum), `pe.csl:734-832` — receive one side
  + own value → `@fadds` → forward one hop. Each PE only ever talks to its
  immediate `POS_DIR`/`NEG_DIR` neighbor.
- `transfer_data_reduce_or()`, `pe.csl:833-921` — identical structure,
  `@or16` instead of `@fadds`.

(Two more variants used to live here, `transfer_data_reduce_select_any()`
and `transfer_data_reduce_select_any_indexed()`, sharing this exact
root/non-root/extreme-PE branching and chain structure — both were removed
entirely, see `ERRORS.md` #24 and the Variant differences section below.)

Teardown (`teardown_reduce_network`, `pe.csl:372-447`) is likewise described
in-source as the "extreme PEs" (index `0` and `w-1`) sending a teardown
wavelet that propagates hop-by-hop — again a linear, not branching,
structure.

## Explicit complexity statement in the codebase

`bool_pe.csl:284-297` states the complexity outright while explaining the
BFS termination-check design:

> Real termination check ... `<collectives_2d>` only gives us per-row
> (`mpi_x`) or per-column (`mpi_y`) primitives, never "the diagonal"
> directly ... which lands the aggregation point at `(MID, MID)` — the
> position that minimizes each phase's **worst-case chain latency
> (`max(root, P-1-root)` hops, per `<collectives_2d>`'s own linear-chain
> implementation)**.

This is first-party confirmation that per-1-D-collective-call cost is
**O(P) hops** (P = PEs along that row/column):

- Worst case `P-1` hops when root is at an extreme index.
- ~`P/2` hops when root is centered (`MID`).

This is also the documented reason the code deliberately roots most phases
at `MID` — see memory note `parent_resolve_root_repositioning_fix.md`: moving
root from an extreme (`P-1` hops) to the middle (`~P/2` hops) roughly halves
worst-case chain latency, consistent with the ~2x real-hardware speedup
already measured.

## 2-D composition: row-then-column, not a global tree

`bool_pe.csl` instantiates the module twice, `mpi_x` and `mpi_y`
(`bool_pe.csl:250,257`), and composes full operations as **sequential
row-chain and column-chain phases**, e.g.:

- `mpi_y.broadcast(root=pcol_id, x_bitmap, ...)` then
  `mpi_x.reduce_or(root=prow_id, y_bitmap, y_bitmap_reduced, ...)`
  (`bool_pe.csl:25-30`; calls at lines 566, 634).
- The 4-phase diagonal-to-diagonal relay (`bool_pe.csl:284-297`):
  column-broadcast → row-reduce → row-broadcast → column-broadcast (calls
  at lines 693, 719, 728, 735).

For a full P×P grid (N = P² PEs), a global collective therefore costs **two
sequential O(P) chain phases** (row then column) = **O(P) = O(√N)** hops
total — better than a single O(N) chain spanning all N PEs, but not the
O(log N) depth a genuine hierarchical tree over all N PEs would give. This is
a 2-D grid row/column relay, built from two orthogonal 1-D chains — not a
branching tree.

## Variant differences (topology is constant across all variants that existed)

Per `ERRORS.md` #16/#17, every reduce variant that ever existed in this
library — dense (`reduce_fadds`/`reduce_or`, still present today),
`reduce_select_any`, and `reduce_select_any_indexed` (both since deleted,
`ERRORS.md` #24) — used the **identical linear two-sided chain topology
and identical root/extreme-PE branching logic**. Differences were purely
in per-hop payload/protocol cost, not topology:

| Variant | Sub-transfers/hop | Merge cost/hop | vs. dense | Topology |
|---|---|---|---|---|
| `reduce_select_any` (dense bitmap) (**deleted**, ERRORS.md #24 — zero call sites since #23) | 1 | O(count) | baseline | chain |
| `reduce_select_any_sparse` (**deleted**, ERRORS.md #16) | 1 | manual bit-scan | 7.6-8.9x slower | chain (unchanged) |
| `reduce_select_any_indexed` (**deleted**, ERRORS.md #24 — its on-device relay caller removed, combine moved host-side) | 4 (bitmap/length/indices/values) | O(popcount) | 4.5-8.9x faster than dense, grows with `blk` | chain (unchanged) |
| `scatter`/`gather` (**deleted**, ERRORS.md #18) | — | — | zero call sites | chain-style switch config (unchanged) |

The library's live surface today (`broadcast`/`reduce_fadds`/`reduce_or`)
is unchanged in topology from what's described above; none of the deleted
variants ever changed the O(P)-hops chain structure — only the
constant-factor cost per hop and code/memory footprint (relevant to the
PE static-memory ceiling noted in ERRORS.md #8, and the eventual reason
#24 removed the indexed variant's on-device caller entirely).

## Summary for complexity claims

- **Topology**: linear chain / bidirectional nearest-neighbor relay,
  converging at a configurable root. Never a branching tree. No collective
  in this library achieves O(log n) depth.
- **Cost per 1-D collective call** (P PEs along a row or column): **O(P)**
  hops, specifically `max(root, P-1-root)`.
- **Cost for a full 2-D grid operation** (N = P² PEs): **O(√N)**, via two
  sequential 1-D chain phases (row then column) — a 2D-grid decomposition,
  not a tree, and not a single O(N) chain either.
- **Root placement matters directly**: centering root at `MID` roughly
  halves worst-case chain latency vs. rooting at an extreme index.
- **Variant choice** (historically dense/sparse/indexed, before #24 removed
  the sparse/indexed reduce-select variants and their on-device caller)
  affected constant-factor per-hop cost and code size only, never the
  underlying O(P) chain topology.
