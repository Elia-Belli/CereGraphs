# `collectives_2d`: communication topology (chain vs. tree)

**Verdict: chain-based, not tree-based.** Every collective primitive in
`collectives_2d` (broadcast, `reduce_fadds`, `reduce_or`, `reduce_select_any`,
`reduce_select_any_indexed`) is implemented as a **1-D bidirectional
linear-chain relay** along a single row or column. There is no branching —
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
- Live, extended fork actually used by the BFS kernel (adds `reduce_or`,
  `reduce_select_any`, `reduce_select_any_indexed`; `scatter`/`gather`
  deleted, see `ERRORS.md` #18):
  [`src/collectives_2d/pe.csl`](../src/collectives_2d/pe.csl) (1885 lines),
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

`configure_broadcast_network`, `pe.csl:764-806` (diagram at `pe.csl:749-763`):
the root's data is routed out via switch-configured routes toward one or both
directions (EAST and/or WEST); each intermediate PE either passes the wavelet
straight through (mid-chain) or terminates it locally (own compute core reads
it). Fan-out from the root is at most "2 directions," after which propagation
is strictly linear in each direction — not a branching tree.

### Reduce family

`reduce_fadds`, `reduce_or`, `reduce_select_any`, `reduce_select_any_indexed`
all share the same alternating-two-color linear-chain pattern, diagrammed
explicitly in source at `pe.csl:830-862`:

```
 C0  +----+  C1  +----+  C0  +----+  C1  +----+  C0  +----+  C1
---- | P0 | ---- | P1 | ---- | P2 | ---- | P3 | ---- | P4 | ----
     +----+      +----+      +----+      +----+      +----+
```

Data flows from both chain ends (`pe_id == 0` and `pe_id == NUM_PES-1`)
toward `root`. Every intermediate PE receives from one side, combines with
its own local value, and forwards one hop toward the root:

- `transfer_data_reduce()` (dense sum), `pe.csl:1050-1135` — non-root branch
  at `pe.csl:1119-1134`: receive one side + own value → `@fadds` → forward
  one hop. Each PE only ever talks to its immediate `POS_DIR`/`NEG_DIR`
  neighbor.
- `transfer_data_reduce_or()`, `pe.csl:1149-1234` — identical structure,
  `@or16` instead of `@fadds`.
- `transfer_data_reduce_select_any()`, `pe.csl:1256-1348` — identical
  structure; per-hop combine is `select_merge_u32()` (`pe.csl:360-365`)
  since there is no fused select ALU op.
- `transfer_data_reduce_select_any_indexed()`, `pe.csl:1381-1748` — same
  root/non-root/extreme-PE branching and same one-hop-at-a-time chain, just 4
  sub-transfers per hop (bitmap, length, indices, values) instead of 1.

Teardown (`teardown_reduce_network`, `pe.csl:688-747`) is likewise described
in-source as the "extreme PEs" (index `0` and `w-1`) sending a teardown
wavelet that propagates hop-by-hop — again a linear, not branching,
structure.

## Explicit complexity statement in the codebase

`bool_pe.csl:360-371` states the complexity outright while explaining the
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
(`bool_pe.csl:295,302`), and composes full operations as **sequential
row-chain and column-chain phases**, e.g.:

- `mpi_y.broadcast(root=pcol_id, x_bitmap, ...)` then
  `mpi_x.reduce_or(root=prow_id, y_bitmap, y_bitmap_reduced, ...)`
  (`bool_pe.csl:17-25`; calls at lines 744, 750, 976).
- The 4-phase diagonal-to-diagonal relay (`bool_pe.csl:360-375`):
  column-broadcast → row-reduce → row-broadcast → column-broadcast.

For a full P×P grid (N = P² PEs), a global collective therefore costs **two
sequential O(P) chain phases** (row then column) = **O(P) = O(√N)** hops
total — better than a single O(N) chain spanning all N PEs, but not the
O(log N) depth a genuine hierarchical tree over all N PEs would give. This is
a 2-D grid row/column relay, built from two orthogonal 1-D chains — not a
branching tree.

## Variant differences (topology is constant across all variants)

Per `ERRORS.md` #16/#17, all variants — dense (`reduce_fadds`/`reduce_or`),
`reduce_select_any`, and `reduce_select_any_indexed`
(`parent_resolve_variant=2`) — use the **identical linear two-sided chain
topology and identical root/extreme-PE branching logic**. Differences are
purely in per-hop payload/protocol cost, not topology:

| Variant | Sub-transfers/hop | Merge cost/hop | vs. dense | Topology |
|---|---|---|---|---|
| `reduce_select_any` (dense bitmap) | 1 | O(count) | baseline | chain |
| `reduce_select_any_sparse` (**deleted**, ERRORS.md #16) | 1 | manual bit-scan | 7.6-8.9x slower | chain (unchanged) |
| `reduce_select_any_indexed` (ERRORS.md #17) | 4 (bitmap/length/indices/values) | O(popcount) | 4.5-8.9x faster, grows with `blk` | chain (unchanged) |
| `scatter`/`gather` (**deleted**, ERRORS.md #18) | — | — | zero call sites | chain-style switch config (unchanged) |

None of these variants change the O(P)-hops chain structure — only the
constant-factor cost per hop and code/memory footprint (relevant to the
PE static-memory ceiling noted in ERRORS.md #8).

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
- **Variant choice** (dense/sparse/indexed) affects constant-factor per-hop
  cost and code size only, never the underlying O(P) chain topology.
