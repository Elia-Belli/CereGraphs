#!/usr/bin/env python3
"""RMAT scale x PE-grid-size sweep, as a GTEPS heatmap (two panels: with and
without host transfer time). Companion to plot_bfs_scaling.py, which plots
GTEPS vs. n at a single fixed grid; this script plots the other axis of the
2-D sweep (bfs/bool_diag_spmv/rmat_grid_sweep.sh) -- same input graph across
every reachable PE grid size, for every RMAT scale.

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

RMAT_RE = re.compile(r"^rmat_s(\d+)_e16\.balanced(\d+)x(\d+)\.mtx$")

GRID_LADDER = [4, 8, 16, 32, 64, 128, 256, 512, 750]


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
  panels = [("gteps", "GTEPS (incl. host transfer)"),
            ("gteps_no_transfer", "GTEPS (compute only, excl. transfer)")]

  ramp_colors = sequential_ramp(BLUE, 256)
  cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue_seq", ramp_colors)

  # Global color scale shared by both panels so they're visually comparable.
  all_vals = []
  for s in scales:
    for g in grids:
      row = by_key.get((s, g))
      if row is None:
        continue
      for col, _label in panels:
        try:
          all_vals.append(float(row[col]))
        except (KeyError, ValueError):
          pass
  vmax = max(all_vals) * 1.02 if all_vals else 1.0

  for ax, (col, label) in zip(axes, panels):
    grid_mat = np.full((len(scales), len(grids)), np.nan)
    for i, s in enumerate(scales):
      for j, g in enumerate(grids):
        row = by_key.get((s, g))
        if row is None:
          continue
        try:
          grid_mat[i, j] = float(row[col])
        except (KeyError, ValueError):
          continue

    masked = np.ma.masked_invalid(grid_mat)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, aspect="auto", origin="lower")

    # Hatch every missing cell so "not run / failed" is never confused with
    # a real, low, GTEPS value.
    for i in range(len(scales)):
      for j in range(len(grids)):
        if np.isnan(grid_mat[i, j]):
          ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                      hatch="////", edgecolor=BASELINE, linewidth=0))
        else:
          ax.text(j, i, f"{grid_mat[i, j]:.2f}", ha="center", va="center",
                   fontsize=7, color=TEXT_PRIMARY if grid_mat[i, j] < vmax * 0.6 else "white")

    ax.set_xticks(range(len(grids)))
    ax.set_xticklabels([f"{g}x{g}" for g in grids], rotation=45, ha="right", fontsize=8,
                        color=TEXT_MUTED)
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels([f"s{s}" for s in scales], fontsize=8, color=TEXT_MUTED)
    ax.set_xlabel("PE grid", color=TEXT_PRIMARY)
    ax.set_title(label, color=TEXT_PRIMARY, fontsize=11)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
      spine.set_visible(False)

  axes[0].set_ylabel("RMAT scale", color=TEXT_PRIMARY)
  cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
  cbar.set_label("GTEPS", color=TEXT_PRIMARY)
  cbar.ax.tick_params(colors=TEXT_MUTED)

  fig.suptitle("RMAT: GTEPS across scale x PE-grid-size (hatched = not run / failed)",
               color=TEXT_PRIMARY, fontsize=12)
  fig.patch.set_facecolor(SURFACE)

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  fig.savefig(args.out, dpi=200, bbox_inches="tight")
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
