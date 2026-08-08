#!/usr/bin/env python3
"""Poster figure: does util/analyze's identity-preserving balancing actually
spread load across the PE grid, or does forcing row i / column i to move
together just shuffle things along the diagonal?

Produces two side-by-side (before/after) figures from a real raw matrix and
its already-balanced counterpart (both already on disk from the normal
prep pipeline -- this script does not call util/analyze itself):

  1. sparsity_*.svg -- every nonzero's (row, col) in the ORIGINAL vertex
     order vs. the BALANCED vertex order. Below RASTERIZE_NNZ_THRESHOLD this
     is a plain vector scatter (one mark per nonzero); past it (real SNAP
     graphs), an individual-point scatter can only show "empty" or
     "saturated" -- every raster pixel it touches goes fully opaque on the
     first hit -- so it switches to a binned log-scale density heatmap
     instead, which actually distinguishes "more spread out" from "more
     edges." Both panel titles also print their own nnz count, the direct
     answer to "did balancing add/drop edges." A shorter row beneath each
     panel (see plot_load_distribution) plots that grid's per-PE nnz-load
     histogram with a Gaussian curve fit overlaid -- the direct "how much
     does each PE deviate from the mean load" view, independent x-axis per
     panel (their absolute scales differ too much to share one usefully).
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

Writes into plots/balancing/<dataset>/ (one subfolder per --raw input, named
after its basename minus ".mtx" -- e.g. "snap_berkstan", "rmat_s10_e16") so
multiple datasets/grid sizes don't clobber each other's output. Filenames
inside that folder carry the grid size instead of the dataset name (the
folder already disambiguates that): sparsity_<grid>x<grid>.svg,
nnz_<grid>x<grid>.svg. Override the parent with --outdir if plots/balancing
itself isn't where a given run should land.
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
MEDIAN_COLOR = "#3f7d5c"  # forest green -- plot_load_distribution's median line, kept
                          # a distinct hue (not just a lighter/darker gray) from MEAN_COLOR
                          # and from BLUE's histogram bars

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

# LEFT_MARGIN/RIGHT_MARGIN/TOP_MARGIN are shared by plot_sparsity/
# plot_nnz_per_pe so their two output figures line up at the same pixel
# size when both exist (small-grid path) -- plot_nnz_per_pe's colorbar would
# otherwise push bbox_inches="tight"'s auto-cropped bbox wider than
# plot_sparsity's (which only gets a colorbar past RASTERIZE_NNZ_THRESHOLD),
# even at an identical figsize. RIGHT_MARGIN reserves the same right-hand
# band in both (colorbar in one, blank in the other) and both now save at a
# fixed bbox instead of a "tight" one, so neither figure's final canvas
# depends on what it happens to draw near its own edges.
#
# FIGSIZE's width used to be 11in against an 8-PE-wide grid's own
# aspect-locked (set_aspect("equal")) square panels -- tight_layout can't
# size aspect-locked axes (it warns "not compatible with tight_layout" and
# silently no-ops), so the panels fell back to matplotlib's stock subplot
# margins (left=0.125/right=0.9) regardless of what rect=[...] asked for,
# leaving a wide unused band at the figure's right edge on top of the
# colorbar's own carve-out. Narrower FIGSIZE plus explicit
# subplots_adjust()/add_gridspec() margins below (which DO apply to
# aspect-locked axes, unlike tight_layout) actually honor RIGHT_MARGIN/
# TOP_MARGIN instead of just coincidentally matching matplotlib's defaults.
#
# LEFT_MARGIN needs more room than plot_nnz_per_pe's own single-digit PE
# ticks strictly require -- plot_sparsity's y-axis carries 4-digit vertex
# indices (e.g. "4000") plus its own rotated "vertex index (row)" label,
# which clipped off the left edge of the fixed canvas at a tighter margin.
# Shared across both functions anyway (doesn't need to match for the
# same-pixel-size goal, LEFT_MARGIN isn't part of that -- but one value is
# simpler than tracking why they'd differ).
FIGSIZE = (9.0, 5.5)
# plot_sparsity is taller than FIGSIZE -- it carries an extra row (the
# per-PE load histogram, see SPARSITY_FIGSIZE usage below) that
# plot_nnz_per_pe doesn't have. save_matched_tight's union-bbox crop works
# off each figure's own rendered tight bbox regardless of its starting
# figsize, so the two companion outputs (small-grid path) still end up with
# matched canvases -- the shorter nnz_*.svg just carries more blank padding.
SPARSITY_FIGSIZE = (9.0, 7.2)
LEFT_MARGIN = 0.095
RIGHT_MARGIN = 0.90

# rmat_s10 (21K nnz, grid=8) is small enough that "every mark a real vector
# element" (see module docstring) is both true and cheap. Real SNAP graphs at
# grid=750 are not: tens of millions of scatter points, or (grid=750)^2 = 562K
# per-cell text labels, would each turn into that many individual SVG
# objects -- practically unrenderable (multi-minute/GB-scale output) and, for
# the per-cell numbers, unreadable at that density regardless. Past these
# thresholds the two plots fall back to a log-density heatmap / no per-cell
# text so real-scale graphs still produce a plot at all; small synthetic
# matrices are unaffected (both thresholds sit well above rmat_s10's 21K/8x8).
#
# RASTERIZE_NNZ_THRESHOLD's name predates the density-heatmap switch (it used
# to just toggle scatter rasterization) but the meaning is the same: above
# this many points, an individual-point scatter stops being informative --
# every raster pixel it touches goes fully opaque on the first hit, so dense
# real graphs render as a flat "empty or saturated" wash with no way to tell
# "more spread out" from "more edges." plot_sparsity switches to a binned 2D
# density heatmap instead once either panel crosses this line.
RASTERIZE_NNZ_THRESHOLD = 50_000  # nnz points, per panel
ANNOTATE_MAX_GRID = 40  # grid side length, per panel

# Bin count for plot_sparsity's density heatmap (see RASTERIZE_NNZ_THRESHOLD
# above) -- picked to roughly match FIGSIZE's own panel width at dpi=220
# (~4in * 220 ~= 900px), not tied to --grid's PE count: --grid is about
# *balancing* (how nnz is distributed across PEs), this is purely about
# *display* resolution for a completely different plot.
DENSITY_BINS = 600


def dataset_tag(raw_path):
  """--raw's basename minus its .mtx extension -- e.g. "snap_berkstan.mtx" ->
  "snap_berkstan" -- used as this run's own subfolder name under --outdir so
  different datasets (and reruns at a different --grid) land side by side
  instead of overwriting each other's sparsity_*/nnz_* files."""
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
  "how much does each PE deviate from the mean load" directly (the grid's
  own per-PE nnz counts), rather than making a reader infer spread from
  peak_mean_ratio's single number or the sparsity panel's own density
  gradient. (An earlier version of this also overlaid a Gaussian curve
  fit to the nonzero loads' own mean/std -- dropped: for a heavy-tailed
  real-graph load distribution the fit is often such a poor match in shape
  that it added visual noise without a corresponding insight, and per-panel
  it could come out anywhere from a reasonable match to entirely flat
  depending on how outlier-heavy that panel's own nonzero tail was.)

  No shared x-axis between the "Original"/"Balanced" callers -- their
  absolute per-PE loads differ by 1-2 orders of magnitude (e.g. berkstan:
  peak ~50K before vs. ~3.5K after), so one shared range would flatten
  whichever panel has the smaller scale. Each call gets its own, independent
  auto-scaled x-axis instead. y-axis IS the same quantity (% of nonzero
  PEs) in both panels, just not the same range (their empty-PE fractions
  differ too), so only the left panel repeats the y-label -- same
  convention the panels above already use for "vertex index (row)" -- and
  its ticks move to the RIGHT panel's own right edge (ax.yaxis.tick_right())
  rather than sitting at its left edge, which is where "Original"'s own
  right edge already is (wspace between the two is narrow, so two sets of
  tick labels both crowding that one gap collided).

  y-axis is a plain percentage (weights=..., NOT density=True) -- each
  bar's own height already IS "% of nonzero PEs in that bin," readable
  without any extra math. A first version used density=True (a probability
  density: height * bin width = fraction, area under all bars = 1), the
  standard convention for overlaying a continuous PDF -- but with the
  Gaussian curve gone (see above) nothing here still needs that property,
  and it actively got in the way of a quick read: these bins are unequal
  width (log-spaced), so two bars of the same height did NOT mean the same
  percentage of PEs under density=True, only under the plain-percentage
  version this switched to.

  x-axis: log, and only the NONZERO per-PE loads are ever binned/plotted --
  0 isn't a threshold, it's an exact equality check (data == 0); real graphs
  are sparse enough that most of a naive grid chop is exactly, not
  approximately, empty (berkstan's own "before" is 96.9% at 750x750). The
  empty share is stated as plain text (below), not as a histogram bar --
  there's no bin for it. An in-histogram zero bin was tried and dropped:
  whatever fraction is exactly empty ends up in one bin of its own (0 can't
  share a log-spaced bin with anything positive), and since that fraction
  is typically the large majority, that one bin dwarfs every other bar
  regardless of how the rest of the axis is scaled -- the empty count is
  better read as an exact percentage than eyeballed off a bar height anyway.

  Bin edges are log-spaced AND integer-snapped (np.unique(np.round(...))):
  plain logspace over integer count data puts several of its lowest edges
  less than 1 apart (e.g. two edges at 1.0 and 1.15), so the very first bin
  -- covering only the exact value 1, real graphs' single most common
  nonzero load -- ends up far narrower than every other bin, which inflates
  its density (count / width) into a dominant spike of its own, just at 1
  instead of 0. Rounding bin edges to the nearest integer (and dropping the
  resulting duplicates) puts a floor of 1 under every bin width.

  Two vertical reference lines, both over the nonzero loads specifically
  (not blocks.mean()/median() over every PE, empty ones included -- that's
  a different, smaller number, the one peak_mean_ratio's title-line metric
  actually divides by):
    - MEAN_COLOR, dashed = mean. For a heavy-tailed real-graph load
      distribution this is NOT where most of the bars are -- a handful of
      extreme-outlier PEs pull it well to the right of the bulk (e.g.
      berkstan before: most PEs sit at loads of 1-10, but the mean is
      ~436). That gap is itself a skew signal, not a bug: the further the
      mean sits from the populated bins, the more a few outlier PEs
      dominate the average.
    - MEDIAN_COLOR, dotted = median. Robust to those same outliers, so it
      lands inside the populated bins as intuition expects -- the two
      lines together make the skew's *size* legible at a glance instead of
      requiring the reader to already know mean != mode for skewed data.
  Distinct hues (not just distinct linestyles, or a light/dark gray pair)
  so the two are unambiguous even skimmed quickly or half-remembered.

  The "N% of PEs empty" figure lives as this legend's own title rather than
  a separate annotation elsewhere in the axes -- both are the same kind of
  fact (a number about this panel's own distribution the reader needs
  alongside the mean/median lines to interpret them), so grouping them into
  one small block beats scattering call-outs around the panel."""
  data = blocks.flatten().astype(np.float64)
  frac_zero = (data == 0).mean()
  positive = data[data > 0]
  if positive.size:
    edges = np.unique(np.round(np.logspace(np.log10(positive.min()), np.log10(positive.max()), 61)))
    weights = np.full(len(positive), 100.0 / len(positive))
    ax.hist(positive, bins=edges, weights=weights, color=BLUE, edgecolor="none")
    if positive.std() > 0:
      ax.axvline(positive.mean(), color=MEAN_COLOR, linestyle="--", linewidth=1.3, label="mean")
      ax.axvline(np.median(positive), color=MEDIAN_COLOR, linestyle=":", linewidth=1.6,
                 label="median")
      legend_title = f"{frac_zero:.0%} of PEs empty" if frac_zero > 0 else None
      # frameon=True + a SURFACE-tinted face, not the frameless look used
      # elsewhere in this file -- a tight/narrow distribution (e.g. after
      # balancing, most PEs land in a tight band) can peak right where a
      # fixed "upper right" legend sits, and frameless text directly over a
      # tall bar was unreadable. Same bbox styling as the standalone
      # empty-PE annotation's own fallback path below, so both stay legible
      # regardless of what's plotted underneath them.
      legend = ax.legend(title=legend_title, loc="upper right", frameon=True,
                          facecolor=SURFACE, edgecolor="none", framealpha=0.85,
                          fontsize=TICK_LABEL_FONTSIZE, labelcolor=TEXT_MUTED,
                          handlelength=1.4, borderaxespad=0.2)
      legend.get_title().set_color(TEXT_MUTED)
      legend.get_title().set_fontsize(TICK_LABEL_FONTSIZE)
    elif frac_zero > 0:
      # Degenerate case (every nonzero PE has the identical load, so there's
      # no mean/median line and thus no legend to attach this to) -- falls
      # back to a standalone annotation instead of a legend title.
      ax.annotate(f"{frac_zero:.0%} of PEs empty", xy=(0.12, 0.85), xycoords="axes fraction",
                  ha="left", va="top", color=TEXT_MUTED, fontsize=TICK_LABEL_FONTSIZE)
    ax.set_xscale("log")
  ax.set_facecolor(SURFACE)
  for spine in ax.spines.values():
    spine.set_color(BASELINE)
  ax.tick_params(colors=TEXT_MUTED, labelsize=TICK_LABEL_FONTSIZE)
  ax.set_xlabel("nnz per PE (excl. empty)", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
  if show_ylabel:
    ax.set_ylabel("% of nonzero PEs", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
  else:
    ax.yaxis.tick_right()


def plot_sparsity(n_raw, rows_raw, cols_raw, n_bal, rows_bal, cols_bal,
                   nnz_raw, nnz_bal, blocks_before, blocks_after,
                   ratio_before=None, ratio_after=None):
  """nnz_raw/nnz_bal are the direct answer to "did balancing add/drop
  edges" -- printed on both panel titles so that check is visible on the
  figure itself, not just a console line a caller has to remember to read.

  blocks_before/blocks_after (per-PE nnz counts, see block_counts) feed the
  per-PE load histogram row beneath the sparsity panels themselves -- see
  plot_load_distribution.

  ratio_before/ratio_after (peak-PE-load / mean-PE-load, see
  peak_mean_ratio) are optional -- only passed by callers that are skipping
  plot_nnz_per_pe entirely (see ANNOTATE_MAX_GRID), so this becomes the only
  place that quantifies the balance quality. None (the default) leaves that
  part of the title out, matching every caller that still gets a companion
  nnz-per-PE heatmap."""
  fig = plt.figure(figsize=SPARSITY_FIGSIZE)
  # 2 rows: top is the existing scatter/density panels (aspect-locked
  # squares, unchanged), bottom is the new shorter "rectangle" per-PE load
  # histogram strip (see plot_load_distribution). Margins passed directly
  # to add_gridspec (rather than a later fig.subplots_adjust() call) so they
  # apply whether or not a colorbar ends up carving into the top row only.
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

  # See RASTERIZE_NNZ_THRESHOLD comment above -- past it, an individual-point
  # scatter can only ever show "empty" or "saturated" (every raster pixel it
  # touches goes fully opaque on the first hit), which is exactly the "looks
  # like balancing added edges" artifact a real density gradient avoids.
  # Decided once for the whole figure (not per panel) so "Original" and
  # "Balanced" are never in different modes -- they're always the same
  # dataset's nnz, so whichever crosses the line, both should.
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
      # transpose to [row_bin, col_bin] so axis 0 is the one imshow's
      # origin="lower" maps to the y (row) axis below.
      h, _, _ = np.histogram2d(cols, rows, bins=DENSITY_BINS, range=[[0, n], [0, n]])
      histograms.append(np.ma.masked_equal(h.T, 0))
    # Shared color scale across both panels (same "honest, not independently
    # rescaled" principle plot_nnz_per_pe already follows) -- vmin=1 since
    # LogNorm can't represent 0 anyway (those bins are masked out above).
    vmax = max(h.max() for h in histograms)
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=vmax)

  for idx, (ax, title, rows, cols, n, nnz, show_ylabel, ratio) in enumerate(panels):
    if use_density:
      # imshow, not pcolormesh -- this is DENSITY_BINS^2 cells (360K at the
      # default), plus it's a raster bitmap either way (imshow has no vector
      # option, see plot_nnz_per_pe's own comment on this), so pcolormesh's
      # per-cell vector quads would only cost render time/file size for zero
      # benefit. origin="lower" + non-flipped extent, then the explicit
      # set_ylim(n, 0) below does the same top-to-bottom flip plot_nnz_per_pe
      # gets from pcolormesh's own coordinate convention.
      im = ax.imshow(histograms[idx], extent=(0, n, 0, n), origin="lower",
                      cmap=cmap, norm=norm, interpolation="nearest", rasterized=True)
    else:
      # Below RASTERIZE_NNZ_THRESHOLD (e.g. rmat_s10) -- plain vector scatter,
      # unchanged from before the density-heatmap switch.
      ax.scatter(cols, rows, s=0.6, c=BLUE, marker="s", linewidths=0, rasterized=False)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    title_line2 = f"nnz {nnz:,}"
    if ratio is not None:
      title_line2 += f" · peak/mean {ratio:.1f}x"
    # Shorter metric line at a smaller size than the "Original"/"Balanced"
    # line above it -- set_title only takes one fontsize for its whole (\n-
    # joined) string, so the metric line is a second, separately-sized Text
    # placed just below via annotate rather than folded into the title
    # itself. Matters because the metric line's text (nnz count Â· ratio) is
    # wide enough at PANEL_TITLE_FONTSIZE to overflow into the neighboring
    # panel given wspace's narrow gap -- the OG combined-string version did
    # exactly that, visibly colliding between panels.
    # Title sits further out (bigger pad, leaving room for the metric line's
    # own height beneath it); metric line sits close to the axes' top edge,
    # so top-to-bottom reads "Original" (big) then "nnz .../peak-mean ..."
    # (small) then the plot -- not the reverse.
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
      # Same 0..n range as the left panel (shared axis convention) -- the
      # tick numbers (and the ticks themselves) would just duplicate what's
      # already readable there.
      ax.tick_params(left=False, labelleft=False)

  fig.suptitle("Non-Zero Elements Layout across PE Grid", color=TEXT_PRIMARY,
               fontsize=SUPTITLE_FONTSIZE, y=SUPTITLE_Y)
  fig.patch.set_facecolor(SURFACE)
  # LEFT_MARGIN/RIGHT_MARGIN/TOP_MARGIN already passed directly to
  # add_gridspec() above -- when use_density is False this panel has no
  # colorbar and its right band just stays blank pre-crop, but the two
  # figures end up the same shape going into save_matched_tight's own
  # union-bbox crop below (see SPARSITY_FIGSIZE comment on why plot_sparsity
  # and plot_nnz_per_pe no longer share one FIGSIZE, just those margins).

  if use_density:
    # This figure now carries real quantitative color data (it didn't when
    # every point was the same opaque BLUE), so a legend earns its place.
    # RIGHT_MARGIN already reserves this exact band (added so plot_sparsity/
    # plot_nnz_per_pe's outputs line up when both are generated) -- reused
    # here even though the large-grid caller runs this figure standalone
    # (no companion nnz figure to match), since the space is already there.
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("Non-Zero Elements per bin (log scale)", color=TEXT_PRIMARY,
                   fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=TICK_LABEL_FONTSIZE)

  # axes (top row) are set_aspect("equal") -- since their GridSpec cell isn't
  # itself square, matplotlib centers a smaller square plot inside that cell
  # at draw time, narrower than the cell's own width. dist_axes (bottom row,
  # not aspect-locked) fill their full cell width regardless, so without
  # this the histogram strip visibly overhangs past both edges of the
  # (narrower) square panel above it -- not just imprecise, actually
  # misaligned. Positions before a draw() call aren't final for aspect-
  # locked axes, so force one, then re-home each dist_axes at its own top
  # axes' real (post-aspect, post-colorbar) left edge/width, keeping
  # dist_axes' own existing y0/height untouched.
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
  """How many times heavier the single busiest PE's load is than the grid's
  own mean load -- the number plot_nnz_per_pe's heatmap is otherwise the only
  way to read off, which stops being readable past ANNOTATE_MAX_GRID (a
  750x750 heatmap's per-cell detail is illegible regardless of vmax/vmin
  color scale). Unlike max/min (what plot_nnz_per_pe already prints below),
  this stays meaningful even when the single least-loaded PE is a 0 -- a real
  possibility at real-graph scale (a 750x750 grid over a ~1M-vertex graph
  averages under 2 nnz/row per block) -- which would make a max/min ratio
  divide-by-(nearly)-zero noise instead of a stable quality signal."""
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
    # pcolormesh instead of imshow, rasterized=False -- imshow always embeds
    # a bitmap in SVG output with no vector option; pcolormesh draws each
    # cell as a real vector quad. Edges offset by -0.5 so cell i's center
    # lands on integer i, matching imshow's own pixel-center convention (and
    # this function's existing tick/text placement at integer coordinates).
    edges = np.arange(grid + 1) - 0.5
    # grid^2 vector quads is fine at grid=8 (64) but not at grid=750 (562K) --
    # same ANNOTATE_MAX_GRID threshold as the per-cell text above, rasterize
    # past it for the same file-size/render-time reason.
    im = ax.pcolormesh(edges, edges, blocks, cmap=cmap, vmin=0, vmax=vmax,
                        rasterized=grid > ANNOTATE_MAX_GRID)
    ax.set_xlim(-0.5, grid - 0.5)
    ax.set_ylim(grid - 0.5, -0.5)  # inverted to match imshow's origin="upper"
    ax.set_aspect("equal")
    # See ANNOTATE_MAX_GRID comment above -- at grid=750 this loop is 562K
    # Text objects per panel, unreadable even if it did render in a sane
    # amount of time. Below the threshold (e.g. the 8x8 rmat case), every
    # cell still gets its real nnz count printed on top of the color.
    if grid <= ANNOTATE_MAX_GRID:
      for i in range(grid):
        for j in range(grid):
          v = blocks[i, j]
          color = "white" if v > vmax * 0.6 else TEXT_PRIMARY
          ax.text(j, i, f"{v}", ha="center", va="center", fontsize=8, color=color)
    # Tick locations/labels: below the threshold, one tick per PE (matches
    # the per-cell text above). Past it, ticks at 0/grid-1 and every 100th
    # PE -- 750 individual tick labels would be as unreadable as the text
    # annotations, and MaxNLocator's usual "nice round number" step doesn't
    # know grid-1 (a PE row/column count, not a round number) is worth a
    # tick of its own.
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
  # carves its space FROM these two (already subplots_adjust-positioned)
  # axes rather than growing the figure, so it lands inside the reserved
  # RIGHT_MARGIN band. subplots_adjust, not tight_layout -- see FIGSIZE
  # comment above for why tight_layout doesn't actually apply here.
  fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN, top=TOP_MARGIN, bottom=0.11)

  cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
  cbar.set_label("Non-Zero Elements per PE", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
  cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=TICK_LABEL_FONTSIZE)
  # Colorbar.solids defaults to rasterized=True regardless of the mappable's
  # own type -- force it vector too, so the SVG has no embedded bitmaps left.
  cbar.solids.set_rasterized(False)

  fig.suptitle("Non-Zero Elements Distribution across PE grid",
               color=TEXT_PRIMARY, fontsize=SUPTITLE_FONTSIZE, y=SUPTITLE_Y)
  fig.patch.set_facecolor(SURFACE)
  print(f"before: min={blocks_before.min()} max={blocks_before.max()} "
        f"(ratio {blocks_before.max() / max(1, blocks_before.min()):.1f}x)")
  print(f"after:  min={blocks_after.min()} max={blocks_after.max()} "
        f"(ratio {blocks_after.max() / max(1, blocks_after.min()):.1f}x)")
  return fig


# Padding (in inches) added around the union of both figures' own tight
# bounding boxes -- just enough that antialiasing/hinting at the final save
# dpi (220, vs. the renderer's own dpi used to measure the tightbbox) can't
# shave a text descender or marker edge off at the crop line.
TIGHT_CROP_PAD_INCHES = 0.06


def save_matched_tight(fig_a, path_a, fig_b, path_b, dpi=220):
  """Crop both companion figures to the union of their own tight bounding
  boxes (each in figure inches, via get_tightbbox) instead of the fixed
  FIGSIZE canvas -- strips the leftover margin on every side (whichever of
  the two figures needs less room on a given side still only gets cropped
  down to the UNION, i.e. down to whichever figure needs the most room
  there) without clipping either figure's real content, and keeps the two
  outputs pixel-identical since they both get saved against the exact same
  bbox."""
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
      os.path.dirname(os.path.abspath(__file__)), "balancing"))
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
    # Past ANNOTATE_MAX_GRID the nnz-per-PE heatmap itself is the useless
    # artifact (grid^2 cells, illegible regardless of rasterization) -- drop
    # it and fold the one number it existed to convey (peak/mean PE load, see
    # peak_mean_ratio) into the sparsity plot's own panel titles instead, so
    # a single figure still says whether balancing helped.
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
