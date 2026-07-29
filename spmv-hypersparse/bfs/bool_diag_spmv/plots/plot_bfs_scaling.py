#!/usr/bin/env python3
"""Cross-input BFS timing chart (the poster's main benchmark, per the plan):
one bar per input graph (RMAT at various scales AND SNAP graphs -- RMAT's
diameter barely grows with scale, which is exactly why SNAP graphs are in
the mix; see this repo's own bfs_timing.csv), stacked/split by color into
h2d_matrix / h2d_seed / compute / resolve / d2h, y-axis in cycles. This is
the PRIMARY performance plot (per-input phase split); a summary table
condensing the same data is a planned follow-up, not built here.

"compute" is search_time_cycles_no_transfer with parent_resolve_max_cycles
(mpi_x.reduce_select_any()'s one-time end-of-run reduce, treated as
transfer-adjacent overhead, not on-device round work) split back out into
its own "resolve" segment -- see plot_bfs_timing.py's PARENT_RESOLVE_COLOR
for the same convention. Without this split, "compute" would silently
include a cost that grows from a small fraction to the large majority of
that column's own height as scale/grid grow.

Reuses bfs/bool_diag_spmv/plots/plot_bfs_timing.py's exact categorical
palette (H2D_BASE_HEX magenta shades for h2d_matrix/h2d_seed, local_compute
yellow, D2H_COLOR orange) and SURFACE/TEXT_PRIMARY/GRIDLINE/BASELINE
styling, plus plot_device_vs_host_timing.py's cycle-count + percentage
segment labels and two-pass label-fit check (draw all bars first, fix
ylim, THEN place/measure labels -- avoids the overlap bug an earlier,
single-pass version of that script had).

Usage: cs_python plots/plot_bfs_scaling.py
         [--csv=results/bfs_timing.csv] [--out=plots/scaling/bfs_scaling.png]

Reads bfs_timing.csv row-by-row; if the same (infile_mtx, pe_grid, source)
combination appears more than once (e.g. rerun during development), only
the LAST occurrence is plotted -- CSV rows are append-only, so "last" is
"most recent".
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Exact palette from plot_bfs_timing.py -- see that file's own H2D_BASE_HEX/
# D2H_COLOR/ROUND_SEGMENT_COLORS/PARENT_RESOLVE_COLOR/hue_shades for the
# full rationale.
H2D_BASE_HEX = "#e87ba4"  # magenta
COMPUTE_COLOR = "#eda100"  # yellow (local_compute)
D2H_COLOR = "#eb6834"  # orange
RESOLVE_COLOR = "#2a78d6"  # blue (mpi_x.reduce_select_any()'s one-time parent resolve)
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def _hex_to_rgb(h):
  h = h.lstrip("#")
  return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb):
  return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)


def hue_shades(base_hex, n):
  """Verbatim copy of plot_bfs_timing.py's own helper -- see there for the
  light->dark / chronological-order rationale."""
  base = np.array(_hex_to_rgb(base_hex))
  white = np.array([1.0, 1.0, 1.0])
  black = np.array([0.0, 0.0, 0.0])
  fracs = np.linspace(0.35, -0.25, n)
  out = []
  for f in fracs:
    if f >= 0:
      out.append(_rgb_to_hex(base * (1 - f) + white * f))
    else:
      out.append(_rgb_to_hex(base * (1 + f) + black * (-f)))
  return out


H2D_MATRIX_COLOR, H2D_SEED_COLOR = hue_shades(H2D_BASE_HEX, 2)

# search_time_cycles_no_transfer (rounds + the one-time transpose_structure()
# + the one-time parent_resolve reduce) still has parent_resolve folded in --
# split it back out into its own "resolve" segment (same convention
# plot_bfs_timing.py already uses) rather than mislabeling it as "compute";
# parent_resolve grows from a small fraction to the large majority of this
# column as scale/grid grow, so leaving it lumped into "compute" would
# increasingly mislabel most of that segment's own height.
SEGMENTS = [
    (lambda row: int(row["h2d_matrix_max_cycles"]), "h2d (matrix)", H2D_MATRIX_COLOR),
    (lambda row: int(row["h2d_seed_max_cycles"]), "h2d (seed)", H2D_SEED_COLOR),
    (lambda row: int(row["search_time_cycles_no_transfer"]) - int(row["parent_resolve_max_cycles"]),
     "compute", COMPUTE_COLOR),
    (lambda row: int(row["parent_resolve_max_cycles"]), "resolve", RESOLVE_COLOR),
    (lambda row: int(row["d2h_max_cycles"]), "d2h", D2H_COLOR),
]


def load_rows(csv_path):
  with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
  # dedupe by (infile_mtx, pe_grid, source), keeping the LAST occurrence
  # (CSV is append-only -> last = most recent rerun of the same input).
  by_key = {}
  for row in rows:
    key = (row["infile_mtx"], row["pe_grid"], row["source"])
    by_key[key] = row
  # preserve first-seen order (stable, doesn't reshuffle on every append)
  seen_order = []
  seen_keys = set()
  for row in rows:
    key = (row["infile_mtx"], row["pe_grid"], row["source"])
    if key not in seen_keys:
      seen_keys.add(key)
      seen_order.append(key)
  return [by_key[k] for k in seen_order]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                "results", "bfs_timing.csv"))
  p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "scaling", "bfs_scaling.png"))
  args = p.parse_args()

  rows = load_rows(args.csv)
  # Sort by vertex count ascending -- reads as a natural scale progression
  # across whatever mix of RMAT/SNAP inputs are in the CSV; the x-tick label
  # (matrix filename) makes which family each bar belongs to obvious.
  rows.sort(key=lambda r: int(r["n"]))

  n_bars = len(rows)
  bar_width = 0.6
  fig, ax = plt.subplots(figsize=(max(6, n_bars * 1.6), 6.5))

  totals = []
  for row in rows:
    total = sum(value_fn(row) for value_fn, _label, _color in SEGMENTS)
    totals.append(total)
  max_total = max(totals) if totals else 1
  ax.set_ylim(0, max_total * 1.08)

  bar_segments = []  # (xpos, bottom, height, label) for the label pass
  xticklabels = []
  for i, row in enumerate(rows):
    xpos = i
    total = totals[i]
    bottom = 0
    for value_fn, _label, color in SEGMENTS:
      height = value_fn(row)
      ax.bar([xpos], [height], width=bar_width, bottom=bottom, color=color,
             edgecolor=SURFACE, linewidth=0.5, zorder=3)
      if height > 0 and total > 0:
        pct = 100.0 * height / total
        bar_segments.append((xpos, bottom, height, f"{height:,}\n({pct:.1f}%)"))
      bottom += height
    matrix_stem = os.path.splitext(row["infile_mtx"])[0]
    xticklabels.append(f"{matrix_stem}\n{row['pe_grid']} n={row['n']}")

  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()
  for xpos, bottom, height, label in bar_segments:
    txt = ax.text(xpos, bottom + height / 2, label, ha="center", va="center",
                  color="white", fontsize=7.5, fontweight="bold", zorder=4)
    bbox = txt.get_window_extent(renderer=renderer)
    bar_top_px = ax.transData.transform((xpos, bottom + height))
    bar_bot_px = ax.transData.transform((xpos, bottom))
    bar_px_height = abs(bar_top_px[1] - bar_bot_px[1])
    if bbox.height > bar_px_height:
      txt.remove()

  from matplotlib.patches import Patch
  legend_handles = [Patch(facecolor=color, label=label) for _, label, color in SEGMENTS]
  ax.legend(handles=legend_handles, loc="upper left", frameon=False, labelcolor=TEXT_PRIMARY)

  ax.set_xticks(range(n_bars))
  ax.set_xticklabels(xticklabels, color=TEXT_PRIMARY, fontsize=8)
  ax.set_ylabel("cycles", color=TEXT_PRIMARY)
  ax.set_title("BFS timing by input graph: h2d (matrix/seed) / compute / resolve / d2h split",
                color=TEXT_PRIMARY)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_color(BASELINE)
  ax.spines["bottom"].set_color(BASELINE)
  ax.tick_params(colors=TEXT_MUTED)
  ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
  ax.set_facecolor(SURFACE)
  fig.patch.set_facecolor(SURFACE)

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  fig.savefig(args.out, dpi=200, bbox_inches="tight")
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
