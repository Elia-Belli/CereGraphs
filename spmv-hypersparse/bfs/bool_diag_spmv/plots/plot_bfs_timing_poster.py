#!/usr/bin/env python3
""" Poster-simplified version of plot_bfs_timing.py's per-run stacked timing
  chart: TWO panels instead of four, each on its own ms scale -- one for the
  per-round device work, one for the one-shot h2d_seed/resolve/d2h bars.

  Dropped relative to plot_bfs_timing.py, for a less cluttered poster figure:
  - h2d_matrix (the one-time matrix-structure upload) -- only h2d_seed (the
    per-search source-seed upload) remains.
  - The whole local_compute_reset/compact/expand breakdown panel.
  - transpose_structure()'s stacked segment (zero on every round except a
    rare direction-switch round; not worth a poster segment).
  - Every min/avg stat tick -- bar height (max across PEs) is the only
    number shown now, for a cleaner poster figure.

  Round bars are now a 2-segment stack (not 3): local_compute and
  local_term_cond are fused into one "compute" segment (aqua, #1baf7a --
  matches plot_grid_scale_heatmap.py's compute-bound color), and
  "communication" (the bcast+reduce+relay remainder) is orange (#eb6834 --
  matches that same heatmap's communication-bound color), so this poster
  plot and the scale x grid heatmap read as one consistent visual language.

  Two panels, NOT one shared scale: h2d_seed/resolve/d2h are one-shot costs
  that can be 2-3 orders of magnitude bigger than a single round's device
  time (see plot_grid_scale_heatmap.py's own parent_resolve findings) --
  cramming both onto one y-axis crushes the round bars to invisible slivers.
  Each panel is auto-scaled to its own data instead; the scale difference is
  meant to read visually (two panels that just look completely different --
  full solid blocks vs. thin slivers), NOT via an explanatory sentence -- a
  poster audience won't stop to read prose.

  Reuses plot_bfs_timing.py's CLI row-selection (select_row) and h2d_seed/
  resolve base colors so this stays behaviorally consistent with the
  original -- only the layout and fused/recolored round segments are new.

  How to run (from bool_diag_spmv/):
     python3 plots/plot_bfs_timing_poster.py --csv results/hw/bfs_timing.csv --row -1
"""

import csv
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bfs_timing import CLOCK_FREQ_HZ  # pylint: disable=wrong-import-position
from plot_bfs_timing import (  # pylint: disable=wrong-import-position
    BASELINE, GRIDLINE, H2D_BASE_HEX, PARENT_RESOLVE_COLOR, SURFACE, TEXT_MUTED, TEXT_PRIMARY,
    parse_args, parse_cycle_list, select_row,
)

# Matches plot_grid_scale_heatmap.py's AQUA (compute-bound pole) and ORANGE
# (communication-bound pole) exactly, so this plot and that heatmap share
# one visual language for "compute" vs "communication".
COMPUTE_COLOR = "#1baf7a"  # aqua
COMMUNICATION_COLOR = "#eb6834"  # orange
# d2h no longer gets its own hue from plot_bfs_timing.py (that file's
# D2H_COLOR is the SAME "#eb6834" now claimed by "communication" above --
# reusing it here would make two different, differently-paneled quantities
# look identical) -- next unused step in this repo's validated categorical
# order (palette.md slot 4) instead.
D2H_COLOR = "#eda100"  # yellow

TITLE_FONTSIZE = 15
PANEL_TITLE_FONTSIZE = 13
LEGEND_FONTSIZE = 11


def cycles_to_ms(cycles, clock_freq_hz):
  return cycles / clock_freq_hz * 1000.0


RMAT_SCALE_RE = re.compile(r"^rmat_s(\d+)_e\d+(?:\.balanced\d+x\d+)?\.mtx$")


def poster_title_input(matrix):
  """'Scale <scale>' for an RMAT input (the poster's own input family); falls
  back to the bare matrix stem for anything else (e.g. a SNAP graph) so this
  script doesn't break on non-RMAT rows."""
  m = RMAT_SCALE_RE.match(matrix)
  return f"Scale {m.group(1)}" if m else os.path.splitext(matrix)[0]


def default_out_path(plots_dir, matrix, pe_grid, source, channels):
  matrix_stem = os.path.splitext(matrix)[0]
  timing_dir = os.path.join(plots_dir, "hw", "timings")
  return os.path.join(timing_dir, f"timing_poster_{matrix_stem}_{pe_grid}_src{source}_ch{channels}.png")


def plot_timing_row_poster(row, out_path):
  matrix = row["infile_mtx"]
  pe_grid = row["pe_grid"]
  clock_freq_hz = float(row.get("clock_freq_hz") or CLOCK_FREQ_HZ)

  round_duration = cycles_to_ms(parse_cycle_list(row["round_duration_cycles"]).astype(float),
                                 clock_freq_hz)
  # profiled_rounds (not rounds_completed) is authoritative -- see
  # plot_bfs_timing.py's own comment on this same truncation subtlety.
  profiled_rounds = len(round_duration)

  local_compute = cycles_to_ms(parse_cycle_list(row["local_compute_max_cycles"]).astype(float),
                                clock_freq_hz)
  local_term_cond = cycles_to_ms(parse_cycle_list(row["local_term_cond_max_cycles"]).astype(float),
                                  clock_freq_hz)
  # fused: local_compute + local_term_cond, one "compute" segment (see
  # module docstring).
  compute = local_compute + local_term_cond
  communication = np.clip(round_duration - local_compute - local_term_cond, 0.0, None)

  h2d_seed_max = cycles_to_ms(int(row["h2d_seed_max_cycles"]), clock_freq_hz)
  parent_resolve_max = cycles_to_ms(int(row["parent_resolve_max_cycles"]), clock_freq_hz)
  d2h_max = cycles_to_ms(int(row["d2h_max_cycles"]), clock_freq_hz)

  round_xs = np.arange(profiled_rounds)
  transfer_labels = ["h2d_seed", "resolve", "d2h"]
  transfer_xs = np.arange(len(transfer_labels))

  round_w_in = max(0.9 * profiled_rounds, 2.6)
  transfer_w_in = 1.3 * len(transfer_labels)
  fig, (ax_rounds, ax_transfer) = plt.subplots(
      1, 2, figsize=(round_w_in + transfer_w_in + 1.5, 6.5),
      gridspec_kw={"width_ratios": [round_w_in, transfer_w_in], "wspace": 0.25})

  bar_width = 0.62
  candidate_labels = []  # (ax, txt, xpos, seg_bottom, seg_top, w)

  def add_solo_bar(ax, xpos, height, color, label):
    ax.bar([xpos], [height], width=bar_width, color=color, edgecolor=SURFACE,
           linewidth=2, label=label, zorder=2)
    txt = ax.text(xpos, height / 2, f"{height:.3f}", ha="center", va="center",
                  fontsize=7, color="white", fontweight="bold", zorder=4)
    candidate_labels.append((ax, txt, xpos, 0.0, height, bar_width))

  bottom = np.zeros(profiled_rounds)
  segments = [("compute", compute, COMPUTE_COLOR), ("communication", communication,
                                                      COMMUNICATION_COLOR)]
  for name, heights, color in segments:
    ax_rounds.bar(round_xs, heights, width=bar_width, bottom=bottom, color=color,
                  edgecolor=color, linewidth=0, label=name, zorder=2)
    total_heights = bottom + heights
    for i, r in enumerate(round_xs):
      txt = ax_rounds.text(r, bottom[i] + heights[i] / 2, f"{heights[i]:.3f}",
                            ha="center", va="center", fontsize=7, color="white",
                            fontweight="bold", zorder=4)
      candidate_labels.append((ax_rounds, txt, r, bottom[i], total_heights[i], bar_width))
    bottom = total_heights

  add_solo_bar(ax_transfer, transfer_xs[0], h2d_seed_max, H2D_BASE_HEX, "h2d_seed")
  add_solo_bar(ax_transfer, transfer_xs[1], parent_resolve_max, PARENT_RESOLVE_COLOR, "resolve")
  add_solo_bar(ax_transfer, transfer_xs[2], d2h_max, D2H_COLOR, "d2h")

  round_max = max(round_duration.max() if profiled_rounds else 0.0, 1e-9)
  transfer_max = max(h2d_seed_max, parent_resolve_max, d2h_max, 1e-9)
  ax_rounds.set_ylim(0, round_max * 1.25)
  ax_transfer.set_ylim(0, transfer_max * 1.15)

  # No direction (TD/BU) suffix -- these poster plots only ever show
  # top-down rounds, so the label would be redundant on every round.
  round_labels = [f"round {r}" for r in range(profiled_rounds)]

  ax_rounds.set_xticks(round_xs)
  ax_rounds.set_xticklabels(round_labels)
  ax_rounds.set_xlim(-0.8, max(profiled_rounds - 0.2, 0.2))
  ax_transfer.set_xticks(transfer_xs)
  ax_transfer.set_xticklabels(transfer_labels)
  ax_transfer.set_xlim(-0.8, len(transfer_labels) - 0.2)

  # measure-first pass: a label only survives if its rendered bounding box
  # actually fits inside its own segment's rectangle -- same convention as
  # plot_bfs_timing.py's own fit check.
  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()
  pad_px = 2.0
  for ax, txt, xpos, seg_bottom, seg_top, w in candidate_labels:
    bbox = txt.get_window_extent(renderer=renderer)
    (x0_disp, y0_disp) = ax.transData.transform((xpos - w / 2, seg_bottom))
    (x1_disp, y1_disp) = ax.transData.transform((xpos + w / 2, seg_top))
    fits_w = (bbox.width + 2 * pad_px) <= (x1_disp - x0_disp)
    fits_h = (bbox.height + 2 * pad_px) <= (y1_disp - y0_disp)
    if not (fits_w and fits_h):
      txt.remove()

  ax_rounds.set_ylabel("ms", labelpad=8)
  ax_rounds.set_title("Per-Round Device Time", fontsize=PANEL_TITLE_FONTSIZE, color=TEXT_PRIMARY)
  ax_transfer.set_title("Host-Device + Parent Resolve",
                         fontsize=PANEL_TITLE_FONTSIZE, color=TEXT_PRIMARY)

  fig.suptitle(f"Timing Split on {poster_title_input(matrix)} and {pe_grid} PE Grid",
               fontsize=TITLE_FONTSIZE)
  for ax in (ax_rounds, ax_transfer):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=TEXT_MUTED)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
  fig.patch.set_facecolor(SURFACE)

  handles, labels = [], []
  for ax in (ax_rounds, ax_transfer):
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
  fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False,
             fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(0.5, 0.01), columnspacing=1.8,
             handletextpad=0.6, labelspacing=1.0)
  plt.tight_layout(rect=[0, 0.08, 1, 0.93])

  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  plt.savefig(out_path, dpi=200, bbox_inches="tight")
  print(f"saved poster timing plot to {out_path}")
  svg_path = os.path.splitext(out_path)[0] + ".svg"
  plt.savefig(svg_path, dpi=200, bbox_inches="tight")
  print(f"saved poster timing plot to {svg_path}")
  plt.close(fig)


def main():
  args = parse_args()

  csv_path = args.csv
  if csv_path is None:
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "results", "bfs_timing.csv")
  with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

  row = select_row(rows, args)
  plots_dir = os.path.dirname(os.path.abspath(__file__))
  out_path = args.out or default_out_path(
      plots_dir, row["infile_mtx"], row["pe_grid"], row["source"], row["channels"])
  plot_timing_row_poster(row, out_path)


if __name__ == "__main__":
  main()
