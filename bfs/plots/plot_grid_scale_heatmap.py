#!/usr/bin/env python3
"""RMAT scale x PE-grid-size sweep, as two heatmaps side by side: GTEPS
(excl. host transfer and parent_resolve, see gteps_excl_parent_resolve)
and the % of on-device time spent in communication vs. compute. Plots one
axis of the 2-D sweep run by rmat_grid_sweep.sh -- same input graph across
every reachable PE grid size, for every RMAT scale.

Shares its styling constants and figure-sizing conventions with
plot_bfs_timing_poster.py and plot_balance_before_after.py so the three
figure families read as one visual system.

Usage: cs_python plots/plot_grid_scale_heatmap.py
         [--csv=results/hw/timings_heatmap.csv] [--out=results/hw/heatmap/rmat_grid_scale.svg]

SVG only -- a poster/report figure meant to be embedded and rescaled as
vector output, not viewed as a standalone raster image.

A missing (scale, grid) cell -- not yet run, or run and never landed a CSV
row (compile/link failure) -- is drawn hatched, not colored zero; GTEPS=0
and "never run" are different facts and must not look the same.

results/hw/timings_heatmap.csv is a separate, hand-trimmed file from the
ongoing results/hw/bfs_timing.csv log -- restored from an older
real-hardware sweep and stripped to just the columns this script reads
(these never depended on the skew-adjustment machinery a later refactor
removed, so the numbers are still correct under the current methodology).
Host-transfer columns (h2d/d2h and the GTEPS built from them) are dropped
entirely: they used a per-PE-max method later found to understate the
true cross-PE span (see docs/GRAPH500_BENCHMARK.md section 15), and the
raw data needed to recompute it no longer exists.
"""

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"  # same accent used for "RMAT" throughout this repo's plots
ORANGE = "#eb6834"  # communication-bound pole of the comm/compute diverging panel
AQUA = "#1baf7a"  # compute-bound pole of the comm/compute diverging panel
NEUTRAL_MID = "#f0efec"  # this repo's documented diverging-pair midpoint
# Best-per-row callout color. Carried only by the underline below a cell's
# number, never by the number's own ink -- against mid-to-dark BLUE/ORANGE
# fills this violet's contrast is too low for a glyph, so text stays in the
# same adaptive TEXT_PRIMARY/white every other cell uses; a thin underline
# reads fine at that lower contrast.
BEST_HIGHLIGHT = "#5b2a86"

# Shared verbatim with plot_bfs_timing_poster.py/plot_balance_before_after.py
# so all three figure families read as one visual system.
SUPTITLE_FONTSIZE = 14
PANEL_TITLE_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 8
SUPTITLE_Y = 0.98  # fraction of figure height; matplotlib's own suptitle default
TITLE_PAD = 10  # points between a panel's title and its own plot area
# tight_layout rect top: NOT the siblings' TOP_MARGIN=0.88 -- these panels
# aren't aspect-locked, so tight_layout actually resizes them here (unlike
# the aspect-locked siblings) and 0.88 left a large dead band above the
# panel titles.
TOP_MARGIN = 0.965

RMAT_RE = re.compile(r"^rmat_s(\d+)_e16\.balanced(\d+)x(\d+)\.mtx$")

GRID_LADDER = [4, 8, 16, 32, 64, 128, 256, 512, 750]

# (scale, grid) keys to skip because their CSV row is stale (a leftover row
# from an earlier session that an idempotency check failed to re-run). Add
# an entry here if that happens again; remove it once the cell has a fresh
# row. See git history for past incidents.
STALE_KEYS = set()

# "Compute" is the two real local-work phases (bfs_timing.py's PHASES,
# each bracketed by a PE's own entry/exit timestamps); "communication" is
# the rest of search_time_cycles_no_transfer (minus parent_resolve, see
# pct_communication) -- bcast/reduce/relay, computed as a remainder rather
# than measured directly (a per-phase breakdown was tried and abandoned as
# unreliable, see docs/GRAPH500_BENCHMARK.md).
COMPUTE_COLS = ["local_compute_max_cycles", "local_term_cond_max_cycles"]

# parent_resolve_max_cycles (the end-of-run parent-array resolve/readback)
# is treated as host-transfer overhead here, not on-device communication --
# excluded from both panels' totals, same as h2d/d2h. It can otherwise
# dominate the total at large scale/grid (~99% of it for s20/750x750),
# swamping the bcast/reduce/relay signal this panel is meant to show.
PARENT_RESOLVE_COL = "parent_resolve_max_cycles"


def _hex_to_rgb(h):
  h = h.lstrip("#")
  return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def sequential_ramp(base_hex, n):
  """One hue, light -> dark, lightening toward white -- convention for a
  magnitude fill."""
  base = np.array(_hex_to_rgb(base_hex))
  white = np.array([1.0, 1.0, 1.0])
  fracs = np.linspace(0.92, 0.0, n)
  return [tuple(base * (1 - f) + white * f) for f in fracs]


def diverging_ramp(low_hex, high_hex, n):
  """Two hues + a neutral gray midpoint (convention for polarity):
  low_hex at 0, NEUTRAL_MID at the center, high_hex at 1."""
  low = np.array(_hex_to_rgb(low_hex))
  mid = np.array(_hex_to_rgb(NEUTRAL_MID))
  high = np.array(_hex_to_rgb(high_hex))
  half = n // 2
  lower_arm = [tuple(low * (1 - f) + mid * f) for f in np.linspace(0.0, 1.0, half)]
  upper_arm = [tuple(mid * (1 - f) + high * f) for f in np.linspace(0.0, 1.0, n - half)]
  return lower_arm + upper_arm


def sum_semicolon_cycles(s):
  """Sum a semicolon-joined per-round cycle-count string, e.g.
  "27683;26588;26481". Raises ValueError on a blank/malformed field."""
  return sum(int(v) for v in s.split(";") if v)


def total_on_device_cycles(row):
  """search_time_cycles_no_transfer minus parent_resolve_max_cycles -- the
  shared 'total' denominator for both panels (see PARENT_RESOLVE_COL).
  Returns None if unparseable or the result is <= 0."""
  try:
    total = float(row["search_time_cycles_no_transfer"]) - float(row[PARENT_RESOLVE_COL])
  except (KeyError, ValueError):
    return None
  return total if total > 0 else None


def gteps_excl_parent_resolve(row):
  """m_edges_traversed / (total_on_device_cycles / clock_freq_hz) / 1e9.
  Recomputed rather than read from the CSV's own gteps_no_transfer column,
  since that column's denominator still includes parent_resolve_max_cycles
  -- this keeps the panel consistent with pct_communication's excluded
  total. Returns None if any required field is missing, unparseable, or
  the total is <= 0."""
  total_cycles = total_on_device_cycles(row)
  if total_cycles is None:
    return None
  try:
    m = float(row["m_edges_traversed"])
    clock_freq_hz = float(row["clock_freq_hz"])
  except (KeyError, ValueError):
    return None
  search_time_seconds = total_cycles / clock_freq_hz
  return m / search_time_seconds / 1e9 if search_time_seconds > 0 else None


def pct_communication(row):
  """(total - compute) / total * 100, where total is
  total_on_device_cycles -- 0% fully compute-bound, 100% fully
  communication-bound. Returns None if any required field is missing,
  unparseable, or the row's total is <= 0."""
  total_cycles = total_on_device_cycles(row)
  if total_cycles is None:
    return None
  try:
    compute_cycles = sum(sum_semicolon_cycles(row[c]) for c in COMPUTE_COLS)
  except (KeyError, ValueError):
    return None
  return (total_cycles - compute_cycles) / total_cycles * 100


def load_rows(csv_path):
  with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
  by_key = {}
  for row in rows:
    m = RMAT_RE.match(row["infile_mtx"])
    if not m:
      continue
    scale = int(m.group(1))
    grid = int(m.group(2))
    if (scale, grid) in STALE_KEYS:
      continue
    by_key[(scale, grid)] = row
  return by_key


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                "results", "hw", "timings_heatmap.csv"))
  p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                "results", "hw", "heatmap", "rmat_grid_scale.svg"))
  args = p.parse_args()

  by_key = load_rows(args.csv)
  scales = sorted({s for s, _g in by_key})
  grids = [g for g in GRID_LADDER if any((s, g) in by_key for s in scales)]
  if not grids:
    grids = GRID_LADDER

  fig, axes = plt.subplots(1, 2, figsize=(1.15 * len(grids) + 3, 0.55 * len(scales) + 3))

  gteps_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
      "blue_seq", sequential_ramp(BLUE, 256))
  comm_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
      "aqua_orange_div", diverging_ramp(AQUA, ORANGE, 256))

  def build_grid(value_fn):
    grid_mat = np.full((len(scales), len(grids)), np.nan)
    for i, s in enumerate(scales):
      for j, g in enumerate(grids):
        row = by_key.get((s, g))
        if row is None:
          continue
        try:
          val = value_fn(row)
        except (KeyError, ValueError):
          continue
        if val is not None:
          grid_mat[i, j] = val
    return grid_mat

  # Each panel is its own unit/scale, so each gets its own vmin/vmax and
  # colorbar. GTEPS is log-scaled: it spans ~0.04 to ~100+ across the
  # sweep, a range a linear color scale would crush toward white at the
  # small end.
  panels = [
      # (matrix-fill function, label, cmap, vmin, vmax, cell text formatter,
      #  colorbar label, log-scale color+values)
      (gteps_excl_parent_resolve,
       "GTEPS (excl. host transfer + parent resolve)", gteps_cmap, None, None,
       lambda v, _vmax: f"{v:.2g}", "GTEPS (log)", True),
      (pct_communication,
       "% time in communication (vs. compute)", comm_cmap, 0, 100,
       lambda v, _vmax: f"{v:.0f}%", "% communication", False),
  ]

  # Best-GTEPS PE grid per RMAT scale, computed once from the GTEPS panel
  # and marked on both panels, so the % communication panel shows what
  # that same choice costs in communication share.
  gteps_grid = build_grid(panels[0][0])
  best_j_per_row = []
  for i in range(len(scales)):
    row_vals = gteps_grid[i, :]
    best_j_per_row.append(None if np.all(np.isnan(row_vals)) else int(np.nanargmax(row_vals)))

  for ax, (value_fn, label, cmap, vmin, vmax, fmt, cbar_label, log_scale) in zip(axes, panels):
    grid_mat = build_grid(value_fn)
    finite = grid_mat[np.isfinite(grid_mat)]

    if log_scale:
      panel_vmin = finite[finite > 0].min() if np.any(finite > 0) else 1e-3
      panel_vmax = finite.max() if finite.size else 1.0
      norm = matplotlib.colors.LogNorm(vmin=panel_vmin, vmax=panel_vmax)
      contrast_fn = lambda v: norm(v)  # noqa: E731
    else:
      panel_vmax = vmax
      if panel_vmax is None:
        panel_vmax = finite.max() * 1.02 if finite.size else 1.0
      norm = None
      contrast_fn = lambda v: v / panel_vmax  # noqa: E731

    # pcolormesh, not imshow: imshow always embeds a bitmap in SVG output,
    # pcolormesh draws each cell as a real vector quad. Edges offset by
    # -0.5 so cell (i, j)'s center lands on integer (j, i), matching this
    # function's tick/text placement.
    masked = np.ma.masked_invalid(grid_mat)
    x_edges = np.arange(len(grids) + 1) - 0.5
    y_edges = np.arange(len(scales) + 1) - 0.5
    if norm is not None:
      im = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap, norm=norm, rasterized=False)
    else:
      im = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap, vmin=vmin, vmax=panel_vmax,
                          rasterized=False)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    # Hatch every missing cell so "not run / failed" is never confused with
    # a real, low value. The best-GTEPS cell (best_j_per_row) is called out
    # as bold text with a BEST_HIGHLIGHT underline (see comment above).
    for i in range(len(scales)):
      for j in range(len(grids)):
        if np.isnan(grid_mat[i, j]):
          ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                      hatch="////", edgecolor=BASELINE, linewidth=0))
          continue
        is_best = best_j_per_row[i] == j
        cell_label = fmt(grid_mat[i, j], panel_vmax)
        color = "white" if contrast_fn(grid_mat[i, j]) >= 0.6 else TEXT_PRIMARY
        ax.text(j, i, cell_label, ha="center", va="center", fontsize=7,
                fontweight="bold" if is_best else "normal", color=color, zorder=7)
        if is_best:
          half_width = min(0.42, 0.09 + 0.09 * len(cell_label))
          ax.plot([j - half_width, j + half_width], [i - 0.19, i - 0.19],
                  color=BEST_HIGHLIGHT, linewidth=1.4, solid_capstyle="butt", zorder=7)

    # Dark staircase border between the run region and the never-run (OOM)
    # region -- every missing cell here is a small-grid/large-scale
    # combination that ran out of memory, never a scattered compile/link
    # failure, so the region reads as one boundary rather than per-cell
    # hatching alone.
    boundary_xs, boundary_ys = [], []
    for i in range(len(scales)):
      finite_js = np.where(~np.isnan(grid_mat[i, :]))[0]
      b = int(finite_js[0]) if finite_js.size else len(grids)
      x = b - 0.5
      if boundary_xs:
        boundary_xs.append(boundary_xs[-1])
        boundary_ys.append(i - 0.5)
        boundary_xs.append(x)
        boundary_ys.append(i - 0.5)
      else:
        boundary_xs.append(x)
        boundary_ys.append(i - 0.5)
      boundary_xs.append(x)
      boundary_ys.append(i + 0.5)
    ax.plot(boundary_xs, boundary_ys, color=TEXT_PRIMARY, linewidth=1.5, zorder=6)

    nan_i, nan_j = np.where(np.isnan(grid_mat))
    if nan_i.size:
      ax.text(nan_j.mean(), nan_i.mean(), "OOM", ha="center", va="center",
              fontsize=15, fontweight="bold", color=TEXT_MUTED, zorder=4)

    ax.set_xticks(range(len(grids)))
    ax.set_xticklabels([f"{g}x{g}" for g in grids], rotation=45, ha="right",
                        fontsize=TICK_LABEL_FONTSIZE, color=TEXT_MUTED)
    ax.set_yticks(range(len(scales)))
    if ax is axes[0]:
      ax.set_yticklabels([f"s{s}" for s in scales], fontsize=TICK_LABEL_FONTSIZE,
                          color=TEXT_MUTED)
    else:
      # Same RMAT-scale rows as the left panel -- ticks would just duplicate it.
      ax.tick_params(left=False, labelleft=False)
    ax.set_xlabel("PE grid", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(label, color=TEXT_PRIMARY, fontsize=PANEL_TITLE_FONTSIZE, pad=TITLE_PAD)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
      spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=TICK_LABEL_FONTSIZE)
    # Colorbar.solids defaults to rasterized=True -- force vector too, so
    # the SVG has no embedded bitmaps left.
    cbar.solids.set_rasterized(False)

  axes[0].set_ylabel("RMAT scale", color=TEXT_PRIMARY, fontsize=AXIS_LABEL_FONTSIZE)

  fig.suptitle("Performance and Communication Share across Scales and PE Grids",
               color=TEXT_PRIMARY, fontsize=SUPTITLE_FONTSIZE, y=SUPTITLE_Y)
  fig.patch.set_facecolor(SURFACE)
  # Reserve headroom for the suptitle above both panel titles.
  fig.tight_layout(rect=[0, 0, 1, TOP_MARGIN])

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  fig.savefig(args.out, dpi=200, bbox_inches="tight")
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
