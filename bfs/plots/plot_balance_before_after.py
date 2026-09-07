#!/usr/bin/env python3
"""Poster figure: does util/analyze's identity-preserving balancing actually
spread load across the PE grid, or does forcing row i / column i to move
together just shuffle things along the diagonal?

Produces two side-by-side (before/after) figures from a real raw matrix
and its already-balanced counterpart (both already on disk -- this
script does not call util/analyze itself):

  1. sparsity_*.svg -- every nonzero's (row, col) in the original vs.
     balanced vertex order. Below RASTERIZE_NNZ_THRESHOLD this is a plain
     vector scatter; past it (real SNAP graphs), an individual-point
     scatter can only show "empty" or "saturated", so it switches to a
     binned log-scale density heatmap instead. Panel titles print their
     own nnz count (did balancing add/drop edges?). A shorter row beneath
     each panel (see plot_load_distribution) plots that grid's per-PE
     nnz-load histogram, independent x-axis per panel (their absolute
     scales differ too much to share one usefully).
  2. nnz_per_pe_before_after.svg -- nnz-per-PE-block heatmap computed two
     ways: "before" chops the original matrix into the same grid (the
     naive distribution util/analyze itself starts from), "after" is the
     real balanced matrix's actual per-block load. Same color scale on
     both panels so the improvement is visually honest.

SVG only -- vector output meant to be embedded/rescaled, not viewed as
standalone raster images. Both figures share one FIGSIZE/RIGHT_MARGIN/
TOP_MARGIN layout so they come out the same pixel size despite only one
carrying a colorbar.

Uses the same SURFACE/TEXT_PRIMARY/GRIDLINE/BASELINE palette and
plot_grid_scale_heatmap.py's sequential single-hue (blue) ramp convention
for the heatmap panels.

Usage: cs_python plots/plot_balance_before_after.py
         --raw=../../data/rmat_s10_e16.mtx
         --balanced=../../data/rmat_s10_e16.balanced8x8.mtx
         --grid=8

Writes into results/balancing/<dataset>/ (one subfolder per --raw input,
named after its basename minus ".mtx") so multiple datasets/grid sizes
don't clobber each other's output. Override the parent with --outdir.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
import numpy as np  # noqa: E402

TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
MEAN_COLOR = "#c1440e"    # burnt orange -- plot_load_distribution's mean line
MEDIAN_COLOR = "#3f7d5c"  # forest green -- median line, distinct hue from MEAN_COLOR/BLUE

# Shared verbatim with plot_grid_scale_heatmap.py/plot_bfs_timing_poster.py
# so the three figure families read as one visual system.
SUPTITLE_FONTSIZE = 14
PANEL_TITLE_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 8
SUPTITLE_Y = 0.98  # fraction of figure height; matplotlib's own suptitle default
TITLE_PAD = 10  # points between a panel's title and its own plot area
TOP_MARGIN = 0.88  # tight_layout rect top -- headroom reserved for the suptitle

# LEFT_MARGIN/RIGHT_MARGIN/TOP_MARGIN are shared by plot_sparsity/
# plot_nnz_per_pe so their two output figures line up at the same pixel
# size when both exist -- plot_nnz_per_pe's colorbar would otherwise push
# bbox_inches="tight"'s auto-cropped bbox wider than plot_sparsity's, even
# at an identical figsize. Both now save at this fixed bbox instead of a
# "tight" one.
#
# subplots_adjust()/add_gridspec() margins, not tight_layout: tight_layout
# can't size aspect-locked (set_aspect("equal")) axes, so it silently
# no-ops on these square panels and the margins below wouldn't apply.
#
# LEFT_MARGIN needs more room than plot_nnz_per_pe's single-digit PE ticks
# strictly require -- plot_sparsity's y-axis carries 4-digit vertex
# indices plus its own rotated label.
FIGSIZE = (9.0, 5.5)
# plot_sparsity is taller than FIGSIZE -- it carries an extra row (the
# per-PE load histogram) that plot_nnz_per_pe doesn't have.
# save_matched_tight's union-bbox crop still ends up with matched canvases;
# the shorter nnz_*.svg just carries more blank padding.
SPARSITY_FIGSIZE = (9.0, 7.2)
LEFT_MARGIN = 0.095
RIGHT_MARGIN = 0.90

# rmat_s10 (21K nnz, grid=8) is small enough that "every mark a real vector
# element" is both true and cheap. Real SNAP graphs at grid=750 are not:
# tens of millions of scatter points, or (grid=750)^2 = 562K per-cell text
# labels, would be practically unrenderable and, for the text, illegible
# regardless. Past these thresholds the two plots fall back to a
# log-density heatmap / no per-cell text.
#
# Above RASTERIZE_NNZ_THRESHOLD, an individual-point scatter stops being
# informative -- every raster pixel it touches goes fully opaque on the
# first hit, so dense real graphs render as a flat "empty or saturated"
# wash with no way to tell "more spread out" from "more edges."
# plot_sparsity switches to a binned 2D density heatmap instead.
RASTERIZE_NNZ_THRESHOLD = 50_000  # nnz points, per panel
ANNOTATE_MAX_GRID = 40  # grid side length, per panel

# Bin count for plot_sparsity's density heatmap -- picked to roughly match
# FIGSIZE's own panel width at dpi=220, not tied to --grid's PE count:
# --grid is about *balancing*, this is purely *display* resolution.
DENSITY_BINS = 600


def dataset_tag(raw_path):
  """--raw's basename minus its .mtx extension, e.g. "snap_berkstan.mtx" ->
  "snap_berkstan" -- used as this run's own subfolder name under --outdir."""
  base = os.path.basename(raw_path)
  return base[:-len(".mtx")] if base.endswith(".mtx") else base


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


def plot_load_distribution(ax, blocks, show_ylabel):
  """The per-PE load histogram strip beneath each sparsity panel -- answers
  "how much does each PE deviate from the mean load" directly, rather than
  making a reader infer spread from peak_mean_ratio's single number or the
  sparsity panel's own density gradient.

  No shared x-axis between the "Original"/"Balanced" callers -- their
  absolute per-PE loads differ by 1-2 orders of magnitude, so one shared
  range would flatten whichever panel has the smaller scale. The y-axis IS
  the same quantity (% of nonzero PEs) in both, just not the same range,
  so only the left panel repeats the y-label, and its ticks move to the
  right panel's own right edge to avoid crowding the narrow gap between
  panels.

  y-axis is a plain percentage (weights=..., not density=True): each bar's
  height already is "% of nonzero PEs in that bin." density=True would be
  wrong here since these bins are unequal width (log-spaced) -- two bars
  of the same height would not mean the same percentage of PEs.

  x-axis: log, and only the nonzero per-PE loads are ever binned -- real
  graphs are sparse enough that most of a naive grid chop is exactly
  empty (berkstan's "before" is 96.9% at 750x750), and that empty share is
  stated as plain text instead of a histogram bar: an exact zero can't
  share a log-spaced bin with anything positive, so an in-histogram zero
  bin would dwarf every other bar regardless of scale.

  Bin edges are log-spaced and integer-snapped (np.unique(np.round(...))):
  plain logspace over integer count data puts several of its lowest edges
  less than 1 apart, which would inflate the density of the bin covering
  the single most common nonzero load (1) into a spike of its own.
  Rounding to the nearest integer puts a floor of 1 under every bin width.

  Two vertical reference lines over the nonzero loads specifically (not
  blocks.mean()/median() including empty PEs -- a different, smaller
  number):
    - MEAN_COLOR, dashed = mean. For a heavy-tailed real-graph load this
      sits well right of the bulk (e.g. berkstan before: most PEs at
      loads of 1-10, mean ~436) -- itself a skew signal, not a bug.
    - MEDIAN_COLOR, dotted = median. Robust to those outliers, so it lands
      inside the populated bins as intuition expects.
  Distinct hues, not just linestyles, so the two stay unambiguous even
  skimmed quickly.

  The "N% of PEs empty" figure lives as this legend's own title (grouped
  with the mean/median lines it explains) rather than a separate
  annotation. "max N nnz/PE" rides in that same title block, always
  present -- the histogram's x-axis shows the busiest bar's rough
  location, but not its exact count.
  """
  data = blocks.flatten().astype(np.float64)
  frac_zero = (data == 0).mean()
  positive = data[data > 0]
  if positive.size:
    edges = np.unique(np.round(np.logspace(np.log10(positive.min()), np.log10(positive.max()), 61)))
    weights = np.full(len(positive), 100.0 / len(positive))
    ax.hist(positive, bins=edges, weights=weights, color=BLUE, edgecolor="none")
    max_load = int(positive.max())
    title_parts = [f"{frac_zero:.0%} of PEs empty"] if frac_zero > 0 else []
    title_parts.append(f"max {max_load:,} nnz/PE")
    # "\n", not " · " -- two lines reads cleanly as a legend title (which
    # matplotlib already centers/left-aligns per line on its own) and keeps
    # either single fact skimmable on its own, rather than one run-on line.
    legend_title = "\n".join(title_parts)
    if positive.std() > 0:
      ax.axvline(positive.mean(), color=MEAN_COLOR, linestyle="--", linewidth=1.3, label="mean")
      ax.axvline(np.median(positive), color=MEDIAN_COLOR, linestyle=":", linewidth=1.6,
                 label="median")
      # loc="best", not a fixed corner: matplotlib scores every corner
      # against the plotted bars' own bounding boxes and picks whichever
      # overlaps least -- a tight post-balancing distribution can peak
      # right under any fixed corner. frameon=True + a SURFACE-tinted face
      # regardless, since even the least-bad corner can brush a bar's edge.
      legend = ax.legend(title=legend_title, loc="best", frameon=True,
                          facecolor=SURFACE, edgecolor="none", framealpha=0.85,
                          fontsize=TICK_LABEL_FONTSIZE, labelcolor=TEXT_MUTED,
                          handlelength=1.4, borderaxespad=0.2)
      legend.get_title().set_color(TEXT_MUTED)
      legend.get_title().set_fontsize(TICK_LABEL_FONTSIZE)
    else:
      # Degenerate case: every nonzero PE has the identical load, so
      # there's no mean/median line to attach a legend to -- fall back to
      # a standalone annotation.
      ax.annotate(legend_title, xy=(0.12, 0.85), xycoords="axes fraction",
                  ha="left", va="top", color=TEXT_MUTED, fontsize=TICK_LABEL_FONTSIZE)
    ax.set_xscale("log")
  ax.set_facecolor(SURFACE)
  for spine in ax.spines.values():
    spine.set_color(BASELINE)
  # which="both": the log-scaled x-axis's minor-tick locator kicks in with
  # numeric labels whenever the plotted range is narrow (true for most
  # "Balanced" panels) -- without "both" those labels would keep
  # matplotlib's own default style instead of this plot's convention.
  ax.tick_params(which="both", colors=TEXT_MUTED, labelsize=TICK_LABEL_FONTSIZE)
  ax.set_xlabel("nnz per PE (excl. empty)", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
  if show_ylabel:
    ax.set_ylabel("% of nonzero PEs", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
  else:
    ax.yaxis.tick_right()


def plot_sparsity(n_raw, rows_raw, cols_raw, n_bal, rows_bal, cols_bal,
                   nnz_raw, nnz_bal, blocks_before, blocks_after,
                   ratio_before=None, ratio_after=None):
  """nnz_raw/nnz_bal are printed on both panel titles -- the direct answer
  to "did balancing add/drop edges," visible on the figure itself.

  blocks_before/blocks_after (per-PE nnz counts, see block_counts) feed
  the per-PE load histogram row beneath the sparsity panels -- see
  plot_load_distribution.

  ratio_before/ratio_after (peak-PE-load / mean-PE-load, see
  peak_mean_ratio) are optional -- only passed by callers skipping
  plot_nnz_per_pe entirely (see ANNOTATE_MAX_GRID), so this becomes the
  only place that quantifies balance quality."""
  fig = plt.figure(figsize=SPARSITY_FIGSIZE)
  # 2 rows: top is the scatter/density panels (aspect-locked squares),
  # bottom is the shorter per-PE load histogram strip (see
  # plot_load_distribution). Margins passed directly to add_gridspec so
  # they apply whether or not a colorbar carves into the top row.
  gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], hspace=0.22, wspace=0.06,
                         left=LEFT_MARGIN, right=RIGHT_MARGIN, top=TOP_MARGIN, bottom=0.08)
  axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
  dist_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
  panels = [
      (axes[0], "Original", rows_raw, cols_raw, n_raw, nnz_raw, True, ratio_before),
      (axes[1], "Balanced", rows_bal, cols_bal, n_bal, nnz_bal, False, ratio_after),
  ]

  for ax, blocks, show_ylabel in zip(dist_axes, (blocks_before, blocks_after), (True, False)):
    plot_load_distribution(ax, blocks, show_ylabel)

  # See RASTERIZE_NNZ_THRESHOLD above. Decided once for the whole figure,
  # not per panel, so "Original" and "Balanced" are never in different
  # modes.
  use_density = max(len(rows_raw), len(rows_bal)) > RASTERIZE_NNZ_THRESHOLD
  im = None
  cmap = norm = None
  histograms = None
  if use_density:
    cmap = sequential_ramp(BLUE, 256)
    cmap.set_bad(SURFACE)  # zero-count bins -> plot background, not "bottom of the log scale"
    histograms = []
    for _, _, rows, cols, n, _, _, _ in panels:
      # histogram2d(x=cols, y=rows) returns H indexed [col_bin, row_bin];
      # transpose so axis 0 is the one imshow's origin="lower" maps to
      # the y (row) axis below.
      h, _, _ = np.histogram2d(cols, rows, bins=DENSITY_BINS, range=[[0, n], [0, n]])
      histograms.append(np.ma.masked_equal(h.T, 0))
    # Shared color scale across both panels -- vmin=1 since LogNorm can't
    # represent 0 anyway (those bins are masked out above).
    vmax = max(h.max() for h in histograms)
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=vmax)

  for idx, (ax, title, rows, cols, n, nnz, show_ylabel, ratio) in enumerate(panels):
    if use_density:
      # imshow, not pcolormesh: this is DENSITY_BINS^2 cells (360K at the
      # default) and a raster bitmap either way, so pcolormesh's per-cell
      # vector quads would only cost render time/file size for no benefit.
      im = ax.imshow(histograms[idx], extent=(0, n, 0, n), origin="lower",
                      cmap=cmap, norm=norm, interpolation="nearest", rasterized=True)
    else:
      # Below RASTERIZE_NNZ_THRESHOLD (e.g. rmat_s10) -- plain vector scatter.
      ax.scatter(cols, rows, s=0.6, c=BLUE, marker="s", linewidths=0, rasterized=False)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    title_line2 = f"nnz {nnz:,}"
    if ratio is not None:
      title_line2 += f" · peak/mean {ratio:.1f}x"
    # Shorter metric line at a smaller size than the title above it:
    # set_title only takes one fontsize for its whole string, so the
    # metric line is a separate Text placed via annotate instead --
    # folding it into the title overflows into the neighboring panel at
    # PANEL_TITLE_FONTSIZE given wspace's narrow gap.
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=PANEL_TITLE_FONTSIZE,
                 pad=TITLE_PAD + 14)
    ax.annotate(title_line2, xy=(0.5, 1.0), xytext=(0, TITLE_PAD),
                xycoords="axes fraction", textcoords="offset points",
                ha="center", va="bottom", color=TEXT_MUTED, fontsize=TICK_LABEL_FONTSIZE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
      spine.set_color(BASELINE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel("vertex index (column)", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
    if show_ylabel:
      ax.set_ylabel("vertex index (row)", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
    else:
      # Same 0..n range as the left panel -- ticks would just duplicate it.
      ax.tick_params(left=False, labelleft=False)

  fig.suptitle("Non-Zero Elements Layout across PE Grid", color=TEXT_PRIMARY,
               fontsize=SUPTITLE_FONTSIZE, y=SUPTITLE_Y)
  fig.patch.set_facecolor(SURFACE)

  if use_density:
    # A legend earns its place now that this figure carries real
    # quantitative color data. RIGHT_MARGIN already reserves this band
    # (added so plot_sparsity/plot_nnz_per_pe's outputs line up), reused
    # here even for the large-grid caller running this figure standalone.
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("Non-Zero Elements per bin (log scale)", color=TEXT_PRIMARY,
                   fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=TICK_LABEL_FONTSIZE)

  # axes (top row) are set_aspect("equal") -- since their GridSpec cell
  # isn't itself square, matplotlib centers a narrower square plot inside
  # it at draw time. dist_axes (bottom row) fill their full cell width
  # regardless, so without this the histogram strip overhangs the
  # (narrower) square panel above it. Force a draw so aspect-locked
  # positions are final, then re-home each dist_axes at its own top axes'
  # real left edge/width.
  fig.canvas.draw()
  for ax_top, ax_bottom in zip(axes, dist_axes):
    top_pos = ax_top.get_position()
    bot_pos = ax_bottom.get_position()
    ax_bottom.set_position([top_pos.x0, bot_pos.y0, top_pos.width, bot_pos.height])
  return fig


def block_counts(rows, cols, n, grid):
  bx = by = int(np.ceil(n / grid))
  blocks = np.zeros((grid, grid), dtype=np.int64)
  row_b = np.minimum(rows // by, grid - 1)
  col_b = np.minimum(cols // bx, grid - 1)
  np.add.at(blocks, (row_b, col_b), 1)
  return blocks


def peak_mean_ratio(blocks):
  """How many times heavier the single busiest PE's load is than the
  grid's own mean load -- readable off plot_nnz_per_pe's heatmap only up
  to ANNOTATE_MAX_GRID. Unlike max/min, stays meaningful even when the
  least-loaded PE is 0 (real at real-graph scale, e.g. a 750x750 grid
  over a ~1M-vertex graph), which would make a max/min ratio divide-by-
  (nearly)-zero noise."""
  return blocks.max() / blocks.mean()


def plot_nnz_per_pe(blocks_before, blocks_after):
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
    # pcolormesh, not imshow: imshow always embeds a bitmap in SVG output,
    # pcolormesh draws each cell as a real vector quad. Edges offset by
    # -0.5 so cell i's center lands on integer i.
    edges = np.arange(grid + 1) - 0.5
    # grid^2 vector quads is fine at grid=8 (64) but not at grid=750
    # (562K) -- rasterize past ANNOTATE_MAX_GRID for file-size/render-time.
    im = ax.pcolormesh(edges, edges, blocks, cmap=cmap, vmin=0, vmax=vmax,
                        rasterized=grid > ANNOTATE_MAX_GRID)
    ax.set_xlim(-0.5, grid - 0.5)
    ax.set_ylim(grid - 0.5, -0.5)  # inverted to match imshow's origin="upper"
    ax.set_aspect("equal")
    # At grid=750 this loop would be 562K Text objects per panel --
    # unreadable even if it rendered in a sane amount of time.
    if grid <= ANNOTATE_MAX_GRID:
      for i in range(grid):
        for j in range(grid):
          v = blocks[i, j]
          color = "white" if v > vmax * 0.6 else TEXT_PRIMARY
          ax.text(j, i, f"{v}", ha="center", va="center", fontsize=8, color=color)
    # Below the threshold, one tick per PE. Past it, ticks at 0/grid-1 and
    # every 100th PE -- 750 individual labels would be as unreadable as
    # the text annotations above.
    if grid <= ANNOTATE_MAX_GRID:
      ticks = list(range(grid))
    else:
      ticks = sorted(set([0] + list(range(100, grid, 100)) + [grid - 1]))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=TICK_LABEL_FONTSIZE, color=TEXT_MUTED)
    ax.set_xlabel("PE column", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
    if show_yticklabels:
      ax.set_yticklabels(ticks, fontsize=TICK_LABEL_FONTSIZE, color=TEXT_MUTED)
      ax.set_ylabel("PE row", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
    else:
      # Same 0..grid-1 PE-row range as the left panel -- ticks would
      # just duplicate it.
      ax.tick_params(left=False, labelleft=False)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=PANEL_TITLE_FONTSIZE, pad=TITLE_PAD)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
      spine.set_visible(False)

  # Reserve the same top/right margins plot_sparsity reserves -- fig.colorbar
  # below carves its space FROM these (already subplots_adjust-positioned)
  # axes rather than growing the figure. subplots_adjust, not tight_layout
  # (see FIGSIZE comment above for why that doesn't apply here).
  fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN, top=TOP_MARGIN, bottom=0.11)

  cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
  cbar.set_label("Non-Zero Elements per PE", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
  cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=TICK_LABEL_FONTSIZE)
  # Colorbar.solids defaults to rasterized=True -- force vector too, so
  # the SVG has no embedded bitmaps left.
  cbar.solids.set_rasterized(False)

  fig.suptitle("Non-Zero Elements Distribution across PE grid",
               color=TEXT_PRIMARY, fontsize=SUPTITLE_FONTSIZE, y=SUPTITLE_Y)
  fig.patch.set_facecolor(SURFACE)
  print(f"before: min={blocks_before.min()} max={blocks_before.max()} "
        f"(ratio {blocks_before.max() / max(1, blocks_before.min()):.1f}x)")
  print(f"after:  min={blocks_after.min()} max={blocks_after.max()} "
        f"(ratio {blocks_after.max() / max(1, blocks_after.min()):.1f}x)")
  return fig


# Padding (in inches) around the union of both figures' tight bounding
# boxes, so antialiasing/hinting at the final save dpi can't shave a text
# descender or marker edge off at the crop line.
TIGHT_CROP_PAD_INCHES = 0.06


def save_matched_tight(fig_a, path_a, fig_b, path_b, dpi=220):
  """Crop both companion figures to the union of their own tight bounding
  boxes instead of the fixed FIGSIZE canvas -- strips the leftover margin
  on every side without clipping either figure's real content, and keeps
  the two outputs pixel-identical since both save against the same bbox."""
  bbox_a = fig_a.get_tightbbox(fig_a.canvas.get_renderer())
  bbox_b = fig_b.get_tightbbox(fig_b.canvas.get_renderer())
  union = Bbox.union([bbox_a, bbox_b]).padded(TIGHT_CROP_PAD_INCHES)
  fig_a.savefig(path_a, dpi=dpi, bbox_inches=union)
  print(f"wrote {path_a}")
  fig_b.savefig(path_b, dpi=dpi, bbox_inches=union)
  print(f"wrote {path_b}")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--raw", required=True)
  p.add_argument("--balanced", required=True)
  p.add_argument("--grid", type=int, required=True)
  p.add_argument("--outdir", default=os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "balancing"))
  args = p.parse_args()

  n_raw, _, rows_raw, cols_raw = read_mtx(args.raw)
  n_bal, _, rows_bal, cols_bal = read_mtx(args.balanced)
  nnz_raw, nnz_bal = len(rows_raw), len(rows_bal)
  print(f"nnz -- raw: {nnz_raw:,}, balanced: {nnz_bal:,} "
        f"({'match' if nnz_raw == nnz_bal else 'MISMATCH -- balancing changed the edge count!'})")

  outdir = os.path.join(args.outdir, dataset_tag(args.raw))
  os.makedirs(outdir, exist_ok=True)

  blocks_before = block_counts(rows_raw, cols_raw, n_raw, args.grid)
  blocks_after = block_counts(rows_bal, cols_bal, n_bal, args.grid)
  grid_tag = f"{args.grid}x{args.grid}"

  if args.grid <= ANNOTATE_MAX_GRID:
    fig_sparsity = plot_sparsity(n_raw, rows_raw, cols_raw, n_bal, rows_bal, cols_bal,
                                  nnz_raw, nnz_bal, blocks_before, blocks_after)
    fig_nnz = plot_nnz_per_pe(blocks_before, blocks_after)
    save_matched_tight(fig_sparsity, os.path.join(outdir, f"sparsity_{grid_tag}.svg"),
                        fig_nnz, os.path.join(outdir, f"nnz_{grid_tag}.svg"))
  else:
    # Past ANNOTATE_MAX_GRID the nnz-per-PE heatmap itself is illegible
    # regardless of rasterization -- drop it and fold the one number it
    # existed to convey (peak/mean PE load) into the sparsity plot's own
    # panel titles instead.
    ratio_before = peak_mean_ratio(blocks_before)
    ratio_after = peak_mean_ratio(blocks_after)
    print(f"peak/mean PE load -- before: {ratio_before:.1f}x, after: {ratio_after:.1f}x")
    fig_sparsity = plot_sparsity(n_raw, rows_raw, cols_raw, n_bal, rows_bal, cols_bal,
                                  nnz_raw, nnz_bal, blocks_before, blocks_after,
                                  ratio_before=ratio_before, ratio_after=ratio_after)
    fig_sparsity.patch.set_facecolor(SURFACE)
    sparsity_path = os.path.join(outdir, f"sparsity_{grid_tag}.svg")
    fig_sparsity.savefig(sparsity_path, dpi=220, bbox_inches="tight",
                          pad_inches=TIGHT_CROP_PAD_INCHES)
    print(f"wrote {sparsity_path}")


if __name__ == "__main__":
  main()
