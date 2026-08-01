#!/usr/bin/env python3
"""RMAT scale x PE-grid-size sweep, as two heatmaps side by side: GTEPS
(excl. host transfer AND parent_resolve, see gteps_excl_parent_resolve) and
the % of on-device time spent in communication vs. compute. Companion to
plot_bfs_scaling.py, which plots GTEPS vs. n at a single fixed grid; this
script plots the other axis of the 2-D sweep
(bfs/bool_diag_spmv/rmat_grid_sweep.sh) -- same input graph across every
reachable PE grid size, for every RMAT scale.

Reuses plot_bfs_scaling.py's exact SURFACE/TEXT_PRIMARY/GRIDLINE/BASELINE
styling and its "last CSV occurrence per key wins" dedup convention (the CSV
is append-only; a matrix file can be rebalanced and rerun under the same
(infile_mtx, pe_grid) key).

Usage: cs_python plots/plot_grid_scale_heatmap.py
         [--csv=results/hw/bfs_timing.csv] [--out=plots/heatmap/rmat_grid_scale.png]

A missing (scale, grid) cell -- not yet run, or run and never landed a CSV
row (compile/link failure) -- is drawn hatched, not colored zero; GTEPS=0
and "never run" are different facts and must not look the same.
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
# best-per-row cell callout -- a single dark violet, distinct from every hue
# used in either panel's own colormap (blue, orange, aqua, gray). Dark enough
# to stay legible against both the darkest cells (base BLUE/ORANGE) and the
# lightest ones, so it doesn't need to switch shade per cell like the
# ordinary (non-highlighted) cell text below does.
BEST_HIGHLIGHT = "#5b2a86"

RMAT_RE = re.compile(r"^rmat_s(\d+)_e16\.balanced(\d+)x(\d+)\.mtx$")

GRID_LADDER = [4, 8, 16, 32, 64, 128, 256, 512, 750]

# (scale, grid) keys with a stale CSV row from an earlier, separate session --
# see git history for the (18, 750)/(20, 750) incident this set was
# originally added for (2026-07-28 leftover rows the 2026-07-29 sweep's
# idempotency check skipped re-running). Both have since been re-run with
# fresh, trend-consistent rows (2026-07-29T13:03/13:23) and removed from
# this set. Add an entry here (and note why) if a similar stale-row
# situation shows up again; remove it once that cell has a fresh row.
STALE_KEYS = set()

# the two REAL local-work phases (bfs_timing.py's PHASES -- each bracketed
# by a PE's own entry/exit timestamps, nothing to adjust) make up "compute";
# the rest of search_time_cycles_no_transfer (minus parent_resolve_max_cycles,
# see pct_communication) is "communication" (bcast/reduce/relay, computed as
# a remainder, not individually measured -- see docs/GRAPH500_BENCHMARK.md for why
# a per-phase skew-adjusted breakdown was tried and abandoned as unreliable).
COMPUTE_COLS = ["local_compute_max_cycles", "local_term_cond_max_cycles"]

# parent_resolve_max_cycles (the end-of-run parent-array resolve/readback,
# see run_bfs.py's comment on parent_resolve_cycles) is treated as host
# transfer overhead here, not on-device communication -- excluded from both
# the numerator and denominator below, same as h2d_seed/d2h are already
# excluded by using search_time_cycles_no_transfer as the starting total.
# It can otherwise dominate search_time_cycles_no_transfer at large
# scale/grid (e.g. ~99% of it for s20/750x750), swamping the per-round
# bcast/reduce/relay signal this panel is meant to show.
PARENT_RESOLVE_COL = "parent_resolve_max_cycles"


def _hex_to_rgb(h):
  h = h.lstrip("#")
  return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def sequential_ramp(base_hex, n):
  """One hue, light -> dark (dataviz convention for a magnitude fill) --
  lighten toward white, never toward a second hue."""
  base = np.array(_hex_to_rgb(base_hex))
  white = np.array([1.0, 1.0, 1.0])
  fracs = np.linspace(0.92, 0.0, n)
  return [tuple(base * (1 - f) + white * f) for f in fracs]


def diverging_ramp(low_hex, high_hex, n):
  """Two hues + a neutral gray midpoint (dataviz convention for polarity) --
  low_hex at 0, NEUTRAL_MID at the center, high_hex at 1, equal step count
  per arm."""
  low = np.array(_hex_to_rgb(low_hex))
  mid = np.array(_hex_to_rgb(NEUTRAL_MID))
  high = np.array(_hex_to_rgb(high_hex))
  half = n // 2
  lower_arm = [tuple(low * (1 - f) + mid * f) for f in np.linspace(0.0, 1.0, half)]
  upper_arm = [tuple(mid * (1 - f) + high * f) for f in np.linspace(0.0, 1.0, n - half)]
  return lower_arm + upper_arm


def sum_semicolon_cycles(s):
  """Sum a semicolon-joined per-round cycle-count string (decode_phase_row's
  CSV convention, e.g. "27683;26588;26481") into a single total. Raises
  ValueError on a blank/malformed field -- caller decides how to treat a
  missing cell."""
  return sum(int(v) for v in s.split(";") if v)


def total_on_device_cycles(row):
  """search_time_cycles_no_transfer with parent_resolve_max_cycles subtracted
  out -- the shared 'total' denominator for both the GTEPS panel and the %
  communication panel, so both treat parent_resolve as host-transfer
  overhead consistently (see PARENT_RESOLVE_COL), not on-device work.
  Returns None if unparseable or the result is <= 0."""
  try:
    total = float(row["search_time_cycles_no_transfer"]) - float(row[PARENT_RESOLVE_COL])
  except (KeyError, ValueError):
    return None
  return total if total > 0 else None


def gteps_excl_parent_resolve(row):
  """m_edges_traversed / (total_on_device_cycles / clock_freq_hz) / 1e9 --
  same formula bfs_timing.py's compute_m_and_gteps uses for the CSV's own
  gteps_no_transfer column, except that column's own denominator
  (search_time_cycles_no_transfer) still includes parent_resolve_max_cycles;
  recomputed here so this panel is consistent with pct_communication's
  parent-resolve-excluded total. Returns None if any required field is
  missing, unparseable, or the total is <= 0."""
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
  total_on_device_cycles (search_time_cycles_no_transfer with
  parent_resolve_max_cycles subtracted out, treated as host transfer, not
  on-device communication -- see PARENT_RESOLVE_COL) -- 0% fully
  compute-bound, 100% fully communication-bound. Returns None if any
  required field is missing, unparseable, or the row's total is <= 0."""
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
                                                "results", "hw", "bfs_timing.csv"))
  p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "heatmap", "rmat_grid_scale.png"))
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

  # Each panel is its own unit/scale (GTEPS vs. % communication), so each
  # gets its own vmin/vmax and colorbar rather than one shared scale.
  # GTEPS is log-scaled: parent_resolve's share of on-device time grows from
  # ~8% at s10/4x4 to ~99% at s20/750x750, so excluding it (see
  # gteps_excl_parent_resolve) makes the remaining denominator shrink toward
  # zero at large scale/grid -- GTEPS spans ~0.04 to ~100+ across the sweep,
  # a range a linear color scale can't show without crushing the small end
  # to white.
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

  # The best-GTEPS PE grid for each RMAT scale, computed once from the GTEPS
  # panel's own values -- outlined on BOTH panels below so the % communication
  # panel shows what that same best-GTEPS choice costs in communication share.
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
      contrast_fn = lambda v: norm(v)  # noqa: E731 -- position in [0, 1] along the log color scale
    else:
      panel_vmax = vmax
      if panel_vmax is None:
        panel_vmax = finite.max() * 1.02 if finite.size else 1.0
      norm = None
      contrast_fn = lambda v: v / panel_vmax  # noqa: E731

    # pcolormesh instead of imshow, rasterized=False -- imshow always embeds
    # a bitmap in SVG output with no vector option; pcolormesh draws each
    # cell as a real vector quad. Edges offset by -0.5 so cell (i, j)'s
    # center lands on integer (j, i), matching imshow's own pixel-center
    # convention (and this function's existing tick/text placement at
    # integer coordinates).
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
    # a real, low value. The best-GTEPS PE grid for each RMAT scale (picked
    # once from the GTEPS panel, see best_j_per_row above) is called out on
    # BOTH panels as bold, underlined text in the violet BEST_HIGHLIGHT
    # shades, so it reads at a glance instead of requiring the reader to
    # spot a thin box outline.
    for i in range(len(scales)):
      for j in range(len(grids)):
        if np.isnan(grid_mat[i, j]):
          ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                      hatch="////", edgecolor=BASELINE, linewidth=0))
          continue
        is_best = best_j_per_row[i] == j
        cell_label = fmt(grid_mat[i, j], panel_vmax)
        if is_best:
          color = BEST_HIGHLIGHT
        else:
          color = "white" if contrast_fn(grid_mat[i, j]) >= 0.6 else TEXT_PRIMARY
        ax.text(j, i, cell_label, ha="center", va="center", fontsize=7,
                fontweight="bold" if is_best else "normal", color=color, zorder=7)
        if is_best:
          half_width = min(0.42, 0.09 + 0.09 * len(cell_label))
          ax.plot([j - half_width, j + half_width], [i - 0.19, i - 0.19],
                  color=color, linewidth=1.4, solid_capstyle="butt", zorder=7)

    # Dark staircase border between the run region and the never-run (OOM)
    # region -- every missing cell here is a small-grid/large-scale
    # combination that ran out of memory (each PE holds a bigger local
    # matrix chunk the fewer PEs the grid has), never a scattered
    # compile/link failure, so the whole region reads as one boundary rather
    # than per-cell hatching alone.
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
    ax.set_xticklabels([f"{g}x{g}" for g in grids], rotation=45, ha="right", fontsize=8,
                        color=TEXT_MUTED)
    ax.set_yticks(range(len(scales)))
    if ax is axes[0]:
      ax.set_yticklabels([f"s{s}" for s in scales], fontsize=8, color=TEXT_MUTED)
    else:
      # Same RMAT-scale rows as the left panel (shared y-axis convention) --
      # the tick labels (and the ticks themselves) would just duplicate it.
      ax.tick_params(left=False, labelleft=False)
    ax.set_xlabel("PE grid", color=TEXT_PRIMARY)
    ax.set_title(label, color=TEXT_PRIMARY, fontsize=13)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
      spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, color=TEXT_PRIMARY)
    cbar.ax.tick_params(colors=TEXT_MUTED)
    # Colorbar.solids defaults to rasterized=True regardless of the
    # mappable's own type -- force it vector too, so the SVG has no
    # embedded bitmaps left.
    cbar.solids.set_rasterized(False)

  axes[0].set_ylabel("RMAT scale", color=TEXT_PRIMARY)

  fig.suptitle("Performance and Communication share across Scales and PE Grids",
               color=TEXT_PRIMARY, fontsize=14)
  fig.patch.set_facecolor(SURFACE)

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  fig.savefig(args.out, dpi=200, bbox_inches="tight")
  print(f"wrote {args.out}")
  svg_path = os.path.splitext(args.out)[0] + ".svg"
  fig.savefig(svg_path, dpi=200, bbox_inches="tight")
  print(f"wrote {svg_path}")


if __name__ == "__main__":
  main()
