#!/usr/bin/env python3
"""Poster figure: does util/analyze's identity-preserving balancing actually
spread load across the PE grid, or does forcing row i / column i to move
together just shuffle things along the diagonal?

Produces two side-by-side (before/after) figures from a real raw matrix and
its already-balanced counterpart (both already on disk from the normal
prep pipeline -- this script does not call util/analyze itself):

  1. sparsity_before_after.svg -- scatter of every nonzero's (row, col) in
     the ORIGINAL vertex order vs. the BALANCED vertex order, with the PE
     grid lines overlaid on the "after" panel.
  2. nnz_per_pe_before_after.svg -- nnz-per-PE-block heatmap computed two
     ways: "before" chops the ORIGINAL matrix into the same grid (the
     naive distribution util/analyze itself starts from, its own
     distribute() function), "after" is the real balanced matrix's actual
     per-block load. Same color scale on both panels so the improvement is
     visually honest, not an artifact of two different scales.

SVG only, no PNG -- these are poster/report figures meant to be
embedded/rescaled as vector output, not viewed as standalone raster images
(same convention as plot_grid_scale_heatmap.py/plot_bfs_timing_poster.py).
Both figures share one FIGSIZE/RIGHT_MARGIN/TOP_MARGIN layout (see their own
comments below) so they come out the same pixel size despite only one of
them carrying a colorbar.

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

# Title/label font sizes and suptitle/title/plot spacing, shared verbatim
# (same names, same values) with plot_grid_scale_heatmap.py and
# plot_bfs_timing_poster.py -- the "hw/heatmap", "timing poster", and
# "balancing" figure families are meant to read as one visual system, not
# three scripts each with their own ad hoc sizing.
SUPTITLE_FONTSIZE = 14
PANEL_TITLE_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 8
SUPTITLE_Y = 0.98  # fraction of figure height; matplotlib's own suptitle default
TITLE_PAD = 10  # points between a panel's title and its own plot area
TOP_MARGIN = 0.88  # tight_layout rect top -- headroom reserved for the suptitle

# Shared by plot_sparsity/plot_nnz_per_pe so their two output figures are
# literally the same pixel size -- plot_nnz_per_pe's colorbar would
# otherwise push bbox_inches="tight"'s auto-cropped bbox wider than
# plot_sparsity's (which has no colorbar), even at an identical figsize.
# RIGHT_MARGIN reserves the same right-hand band in both (colorbar in one,
# blank in the other) and both now save at a fixed bbox instead of a
# "tight" one, so neither figure's final canvas depends on what it happens
# to draw near its own edges.
FIGSIZE = (11, 5.5)
RIGHT_MARGIN = 0.90


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


def plot_sparsity(n_raw, rows_raw, cols_raw, n_bal, rows_bal, cols_bal, out_path):
  fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"wspace": 0.06})
  panels = [
      (axes[0], "Original", rows_raw, cols_raw, n_raw, True),
      (axes[1], "Balanced", rows_bal, cols_bal, n_bal, False),
  ]
  for ax, title, rows, cols, n, show_ylabel in panels:
    ax.scatter(cols, rows, s=0.6, c=BLUE, marker="s", linewidths=0, rasterized=False)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=PANEL_TITLE_FONTSIZE, pad=TITLE_PAD)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
      spine.set_color(BASELINE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel("vertex index (column)", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
    if show_ylabel:
      ax.set_ylabel("vertex index (row)", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
    else:
      # Same 0..n range as the left panel (shared axis convention) -- the
      # tick numbers (and the ticks themselves) would just duplicate what's
      # already readable there.
      ax.tick_params(left=False, labelleft=False)

  fig.suptitle("Non-Zero Elements Layout across PE Grid", color=TEXT_PRIMARY,
               fontsize=SUPTITLE_FONTSIZE, y=SUPTITLE_Y)
  fig.patch.set_facecolor(SURFACE)
  # Reserve the same top/right margins plot_nnz_per_pe reserves for its own
  # suptitle/colorbar (see RIGHT_MARGIN/TOP_MARGIN comments above) -- this
  # panel has no colorbar, so its right band just stays blank, but the two
  # figures end up the same shape and (see the fixed-bbox save below) the
  # same pixel size.
  fig.tight_layout(rect=[0, 0, RIGHT_MARGIN, TOP_MARGIN])

  # SVG only, no PNG (poster/report figure, vector output meant to be
  # embedded/rescaled). No bbox_inches="tight" -- unlike this file's own
  # colorbar panel below, a "tight" bbox here would crop to exactly this
  # panel's own (colorbar-less) content and no longer match
  # plot_nnz_per_pe's saved size; a fixed FIGSIZE-at-dpi bbox for both
  # guarantees the two companion figures are pixel-identical.
  fig.savefig(out_path, dpi=220)
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
  fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"wspace": 0.06})
  panels = [
      (axes[0], "Original", blocks_before, True),
      (axes[1], "Balanced", blocks_after, False),
  ]
  im = None
  for ax, title, blocks, show_yticklabels in panels:
    grid = blocks.shape[0]
    # pcolormesh instead of imshow, rasterized=False -- imshow always embeds
    # a bitmap in SVG output with no vector option; pcolormesh draws each
    # cell as a real vector quad. Edges offset by -0.5 so cell i's center
    # lands on integer i, matching imshow's own pixel-center convention (and
    # this function's existing tick/text placement at integer coordinates).
    edges = np.arange(grid + 1) - 0.5
    im = ax.pcolormesh(edges, edges, blocks, cmap=cmap, vmin=0, vmax=vmax, rasterized=False)
    ax.set_xlim(-0.5, grid - 0.5)
    ax.set_ylim(grid - 0.5, -0.5)  # inverted to match imshow's origin="upper"
    ax.set_aspect("equal")
    for i in range(grid):
      for j in range(grid):
        v = blocks[i, j]
        color = "white" if v > vmax * 0.6 else TEXT_PRIMARY
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(grid))
    ax.set_yticks(range(grid))
    ax.set_xticklabels(range(grid), fontsize=TICK_LABEL_FONTSIZE, color=TEXT_MUTED)
    ax.set_xlabel("PE column", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
    if show_yticklabels:
      ax.set_yticklabels(range(grid), fontsize=TICK_LABEL_FONTSIZE, color=TEXT_MUTED)
      ax.set_ylabel("PE row", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
    else:
      # Same 0..grid-1 PE-row range as the left panel (shared axis
      # convention) -- the tick numbers (and the ticks themselves) would
      # just duplicate it.
      ax.tick_params(left=False, labelleft=False)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=PANEL_TITLE_FONTSIZE, pad=TITLE_PAD)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
      spine.set_visible(False)

  # Reserve the same top/right margins plot_sparsity reserves (see
  # RIGHT_MARGIN/TOP_MARGIN comments above) -- fig.colorbar(ax=axes) below
  # carves its space FROM these two (already tight_layout-positioned) axes
  # rather than growing the figure, so it lands inside the reserved
  # RIGHT_MARGIN band.
  fig.tight_layout(rect=[0, 0, RIGHT_MARGIN, TOP_MARGIN])

  cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
  cbar.set_label("Non-Zero Elements per PE", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
  cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=TICK_LABEL_FONTSIZE)
  # Colorbar.solids defaults to rasterized=True regardless of the mappable's
  # own type -- force it vector too, so the SVG has no embedded bitmaps left.
  cbar.solids.set_rasterized(False)

  fig.suptitle("Non-Zero Elements Distribution across PE grid",
               color=TEXT_PRIMARY, fontsize=SUPTITLE_FONTSIZE, y=SUPTITLE_Y)
  fig.patch.set_facecolor(SURFACE)
  # SVG only, no PNG; no bbox_inches="tight" -- see plot_sparsity's own
  # comment on why: a fixed FIGSIZE-at-dpi bbox on both is what makes these
  # two companion figures come out pixel-identical.
  fig.savefig(out_path, dpi=220)
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
  plot_sparsity(n_raw, rows_raw, cols_raw, n_bal, rows_bal, cols_bal,
                os.path.join(args.outdir, "sparsity_before_after.svg"))

  blocks_before = block_counts(rows_raw, cols_raw, n_raw, args.grid)
  blocks_after = block_counts(rows_bal, cols_bal, n_bal, args.grid)
  plot_nnz_per_pe(blocks_before, blocks_after,
                  os.path.join(args.outdir, "nnz_per_pe_before_after.svg"))


if __name__ == "__main__":
  main()
