#!/usr/bin/env python3
"""Grouped bar chart for the device-only-vs-SDK-host-driven timing
comparison (Workstream B): one group per (matrix, grid) test case, two bars
per group (sdk-hypersparse-spmv host-driven vs. fp32_diag_spmv device-only),
each bar stacked/split by color into h2d/compute/d2h segments, y-axis in
cycles. Reuses bool_diag_spmv/plots/plot_bfs_timing.py's categorical hex
palette and SURFACE/TEXT_PRIMARY/GRIDLINE/BASELINE styling for visual
consistency with the other poster panels, but labels each segment with both
its cycle count AND its percentage of that bar's total (plot_bfs_timing.py
only labels raw cycle counts) -- the one deliberate deviation from that
convention.

Usage: cs_python benchmarks/plot_device_vs_host_timing.py
         [--csv=benchmarks/results/device_vs_host_timing.csv]
         [--out=benchmarks/plots/device_vs_host_timing.png]
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Palette matches bool_diag_spmv/plots/plot_bfs_timing.py exactly.
H2D_COLOR = "#e87ba4"  # magenta
COMPUTE_COLOR = "#eda100"  # yellow
D2H_COLOR = "#eb6834"  # orange
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

SEGMENTS = [("h2d_cycles", "h2d", H2D_COLOR), ("compute_cycles", "compute", COMPUTE_COLOR),
            ("d2h_cycles", "d2h", D2H_COLOR)]


def load_rows(csv_path):
  with open(csv_path, encoding="utf-8") as f:
    return list(csv.DictReader(f))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "results", "device_vs_host_timing.csv"))
  p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "plots", "device_vs_host_timing.png"))
  args = p.parse_args()

  rows = load_rows(args.csv)
  cases = []
  for row in rows:
    key = (row["matrix"], row["pe_grid"])
    if key not in cases:
      cases.append(key)

  n_cases = len(cases)
  bar_width = 0.35
  group_gap = 1.0
  fig, ax = plt.subplots(figsize=(max(6, n_cases * 2.4), 6.5))

  # Pass 1: draw every bar/segment and fix the y-axis limit up front (from
  # the full dataset, not incrementally) -- doing this BEFORE any label is
  # placed matters: transData depends on ylim, and ylim only settles to its
  # final value once every bar is drawn, so measuring label fit against a
  # not-yet-final ylim (as an earlier version of this script did) silently
  # under-measures early bars and lets oversized labels through.
  max_total = max(int(row["total_cycles"]) for row in rows)
  ax.set_ylim(0, max_total * 1.08)

  xticks = []
  xticklabels = []
  bar_segments = []  # (xpos, bottom, height, color, label_text) for pass 2
  for i, (matrix, pe_grid) in enumerate(cases):
    group_x = i * group_gap
    case_rows = [r for r in rows if r["matrix"] == matrix and r["pe_grid"] == pe_grid]
    for j, row in enumerate(case_rows):
      xpos = group_x + (j - 0.5) * bar_width * 1.15
      total = int(row["total_cycles"])
      bottom = 0
      for col, _label, color in SEGMENTS:
        height = int(row[col])
        ax.bar([xpos], [height], width=bar_width, bottom=bottom, color=color,
               edgecolor=SURFACE, linewidth=0.5, zorder=3)
        if height > 0 and total > 0:
          pct = 100.0 * height / total
          bar_segments.append((xpos, bottom, height, f"{height:,}\n({pct:.1f}%)"))
        bottom += height
    xticks.append(group_x)
    xticklabels.append(f"{os.path.basename(matrix)}\n{pe_grid}")

  # Pass 2: now that ylim is final, place labels and drop any that don't fit
  # their own segment's rendered height (measure-first, matching
  # plot_bfs_timing.py's convention) -- avoids the overlapping-text failure
  # mode a single-pass, draw-as-you-go check produced.
  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()
  for xpos, bottom, height, label in bar_segments:
    txt = ax.text(xpos, bottom + height / 2, label, ha="center", va="center",
                  color="white", fontsize=8, fontweight="bold", zorder=4)
    bbox = txt.get_window_extent(renderer=renderer)
    bar_top_px = ax.transData.transform((xpos, bottom + height))
    bar_bot_px = ax.transData.transform((xpos, bottom))
    bar_px_height = abs(bar_top_px[1] - bar_bot_px[1])
    if bbox.height > bar_px_height:
      txt.remove()

  # Legend: one swatch per segment kind, plus a text note distinguishing the two bars per group.
  from matplotlib.patches import Patch
  legend_handles = [Patch(facecolor=color, label=label) for _, label, color in SEGMENTS]
  ax.legend(handles=legend_handles, loc="upper right", frameon=False, labelcolor=TEXT_PRIMARY)

  ax.set_xticks(xticks)
  ax.set_xticklabels(xticklabels, color=TEXT_PRIMARY)
  ax.set_ylabel("cycles", color=TEXT_PRIMARY)
  ax.set_title("Device-only vs. SDK host-driven: h2d/compute/d2h split (10 rounds)",
                color=TEXT_PRIMARY)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_color(BASELINE)
  ax.spines["bottom"].set_color(BASELINE)
  ax.tick_params(colors=TEXT_MUTED)
  ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
  ax.set_facecolor(SURFACE)
  fig.patch.set_facecolor(SURFACE)

  # Per-group sub-labels ("sdk" / "device") just under the x-axis.
  for i in range(n_cases):
    group_x = i * group_gap
    for j, tag in enumerate(["sdk", "device"]):
      xpos = group_x + (j - 0.5) * bar_width * 1.15
      ax.annotate(tag, (xpos, 0), xytext=(0, -28), textcoords="offset points",
                  ha="center", fontsize=7.5, color=TEXT_MUTED)

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  fig.savefig(args.out, dpi=200, bbox_inches="tight")
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
