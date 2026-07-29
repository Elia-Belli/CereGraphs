#!/usr/bin/env python3
"""Poster figure: does util/analyze's identity-preserving balancing actually
spread load across the PE grid, or does forcing row i / column i to move
together just shuffle things along the diagonal?

Produces two side-by-side (before/after) figures from a real raw matrix and
its already-balanced counterpart (both already on disk from the normal
prep pipeline -- this script does not call util/analyze itself):

  1. sparsity_before_after.png -- scatter of every nonzero's (row, col) in
     the ORIGINAL vertex order vs. the BALANCED vertex order, with the PE
     grid lines overlaid on the "after" panel.
  2. nnz_per_pe_before_after.png -- nnz-per-PE-block heatmap computed two
     ways: "before" chops the ORIGINAL matrix into the same grid (the
     naive distribution util/analyze itself starts from, its own
     distribute() function), "after" is the real balanced matrix's actual
     per-block load. Same color scale on both panels so the improvement is
     visually honest, not an artifact of two different scales.

Reuses plot_bfs_scaling.py's exact SURFACE/TEXT_PRIMARY/GRIDLINE/BASELINE
palette and plot_grid_scale_heatmap.py's sequential single-hue (blue) ramp
convention for the heatmap panels.

Usage: cs_python plots/plot_balance_before_after.py
         --raw=../../data/rmat_s10_e16.mtx
         --balanced=../../data/rmat_s10_e16.balanced8x8.mtx
         --grid=8
         --outdir=plots/balance_before_after
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def read_mtx(path):
  with open(path) as f:
    line = f.readline()
    while line.startswith("%"):
      line = f.readline()
    nrows, ncols, nnz = (int(x) for x in line.split())
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)
    for i, line in enumerate(f):
      r, c = line.split()[:2]
      rows[i] = int(r) - 1
      cols[i] = int(c) - 1
  return nrows, ncols, rows, cols


def sequential_ramp(base_hex, n):
  base = np.array([int(base_hex[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
  white = np.array([1.0, 1.0, 1.0])
  fracs = np.linspace(0.92, 0.0, n)
  return matplotlib.colors.LinearSegmentedColormap.from_list(
      "blue_seq", [tuple(base * (1 - f) + white * f) for f in fracs])


def plot_sparsity(n_raw, rows_raw, cols_raw, n_bal, rows_bal, cols_bal, grid, out_path):
  fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
  panels = [
      (axes[0], "Before -- original RMAT vertex order", rows_raw, cols_raw, n_raw, False),
      (axes[1], "After -- balanced, identity-preserving permutation", rows_bal, cols_bal, n_bal, True),
  ]
  for ax, title, rows, cols, n, draw_grid in panels:
    ax.scatter(cols, rows, s=0.6, c=BLUE, marker="s", linewidths=0, rasterized=True)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
      spine.set_color(BASELINE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    if draw_grid:
      step = n / grid
      for g in range(1, grid):
        ax.axhline(g * step, color=ORANGE, linewidth=0.7, alpha=0.7)
        ax.axvline(g * step, color=ORANGE, linewidth=0.7, alpha=0.7)

  fig.suptitle(f"Same {len(rows_raw):,} nonzeros, before vs. after balancing "
               f"({grid}x{grid} PE grid)", color=TEXT_PRIMARY, fontsize=12)
  fig.patch.set_facecolor(SURFACE)
  # scatter layers are rasterized (set above) so the SVG stays small even at
  # 20k+ points; axes/text/gridlines remain real vector elements.
  fig.savefig(out_path, dpi=220, bbox_inches="tight")
  print(f"wrote {out_path}")


def block_counts(rows, cols, n, grid):
  bx = by = int(np.ceil(n / grid))
  blocks = np.zeros((grid, grid), dtype=np.int64)
  row_b = np.minimum(rows // by, grid - 1)
  col_b = np.minimum(cols // bx, grid - 1)
  np.add.at(blocks, (row_b, col_b), 1)
  return blocks


def plot_nnz_per_pe(blocks_before, blocks_after, out_path):
  vmax = max(blocks_before.max(), blocks_after.max())
  cmap = sequential_ramp(BLUE, 256)
  fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
  panels = [
      (axes[0], "Before (naive grid chop)", blocks_before),
      (axes[1], "After (balanced)", blocks_after),
  ]
  im = None
  for ax, title, blocks in panels:
    im = ax.imshow(blocks, cmap=cmap, vmin=0, vmax=vmax, origin="upper")
    grid = blocks.shape[0]
    for i in range(grid):
      for j in range(grid):
        v = blocks[i, j]
        color = "white" if v > vmax * 0.6 else TEXT_PRIMARY
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=8, color=color)
        if i == j:
          ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                      edgecolor=ORANGE, linewidth=1.6))
    ax.set_xticks(range(grid))
    ax.set_yticks(range(grid))
    ax.set_xticklabels(range(grid), fontsize=7, color=TEXT_MUTED)
    ax.set_yticklabels(range(grid), fontsize=7, color=TEXT_MUTED)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
      spine.set_visible(False)

  cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
  cbar.set_label("nnz per PE", color=TEXT_PRIMARY)
  cbar.ax.tick_params(colors=TEXT_MUTED)

  fig.suptitle("Nonzeros per PE block: naive chop vs. balanced "
               "(orange = block-diagonal, px == py)", color=TEXT_PRIMARY, fontsize=12)
  fig.patch.set_facecolor(SURFACE)
  fig.savefig(out_path, dpi=220, bbox_inches="tight")
  print(f"wrote {out_path}")
  print(f"before: min={blocks_before.min()} max={blocks_before.max()} "
        f"(ratio {blocks_before.max() / max(1, blocks_before.min()):.1f}x)")
  print(f"after:  min={blocks_after.min()} max={blocks_after.max()} "
        f"(ratio {blocks_after.max() / max(1, blocks_after.min()):.1f}x)")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--raw", required=True)
  p.add_argument("--balanced", required=True)
  p.add_argument("--grid", type=int, required=True)
  p.add_argument("--outdir", default=os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "balance_before_after"))
  args = p.parse_args()

  n_raw, _, rows_raw, cols_raw = read_mtx(args.raw)
  n_bal, _, rows_bal, cols_bal = read_mtx(args.balanced)

  os.makedirs(args.outdir, exist_ok=True)
  plot_sparsity(n_raw, rows_raw, cols_raw, n_bal, rows_bal, cols_bal, args.grid,
                os.path.join(args.outdir, "sparsity_before_after.svg"))

  blocks_before = block_counts(rows_raw, cols_raw, n_raw, args.grid)
  blocks_after = block_counts(rows_bal, cols_bal, n_bal, args.grid)
  plot_nnz_per_pe(blocks_before, blocks_after,
                  os.path.join(args.outdir, "nnz_per_pe_before_after.svg"))


if __name__ == "__main__":
  main()
