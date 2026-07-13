#!/usr/bin/env python3
""" plot a single bfs_timing.csv row (one run_bfs.py run) as a stacked
  bar chart: one bar per BFS round (left to right), bracketed by two
  standalone "h2d" bars (h2d_matrix, h2d_seed -- see bfs_timing.py's
  H2D_PARTS / GRAPH500_BENCHMARK.md for the Graph500-motivated split between
  the one-time matrix-structure upload and the per-search source-seed
  upload) before round 0, and a standalone "d2h" bar (visited_buf +
  parent_local_buf readback -- the actual BFS output, not run_bfs.py's own
  rounds_completed/ts_buf instrumentation reads -- after the last round) --
  the one-shot host<->device transfers that aren't part of any round.

  Round bar height = bfs_timing.compute_round_summary's round_duration_cycles
  (straggler-PE span from that round's TS_VBCAST_ISSUE to its
  TS_TERM_COL_BCAST_DONE) -- NOT a sum of individually skew-adjusted
  sub-phases. GRAPH500_BENCHMARK.md sections 10-13 found that decomposing
  a round's communication phases (broadcasts/reduces/the termination
  relay) into "real cost" vs "cross-PE wait" is a hard, still-unresolved
  problem on its own -- repeatedly produced results (impossible negative
  costs, then implausibly flat ones) that didn't survive scrutiny. Rather
  than keep guessing at that, this plot only decomposes what's actually
  reliably, locally measurable per PE with no cross-PE synchronization
  ambiguity at all: `local_compute` (the boolean SpMV multiply) and
  `local_term_cond` (the diagonal-only masking loop) -- both bracketed by
  a PE's own entry/exit timestamps, nothing to adjust. Everything else in
  the round (visited_bcast, vertical_bcast, the SpMV reduce, and the whole
  4-phase termination relay) is lumped into one "communication" segment,
  computed as the remainder: `round_duration - local_compute - local_term_cond`.
  This is deliberately NOT chronologically ordered within the bar (some of
  that communication time happens before local_compute, some after
  local_term_cond) -- the three segments are stacked in a fixed
  [local_compute, local_term_cond, communication] order purely for a
  stable, readable legend, not to claim a literal timeline.

  The suptitle also reports `total_runtime_cycles` (the straggler-PE span
  from round 0's very first TS_VBCAST_ISSUE to the last round's
  TS_TERM_COL_BCAST_DONE) next to the sum of the round bars, as a sanity
  check -- they should be close (the gap is just the first broadcast's own
  one-time fabric fill delay, not per-round recurring cost).

  This module is importable (plot_timing_row(row, out_path)) -- run_bfs.py
  calls it directly after appending a row, so one run_bfs.py invocation
  produces the plot without a separate manual step. It's also runnable
  standalone, to re-plot an existing CSV row without re-running the device:

  How to run (from bool_diag_spmv/, or use --csv's own default: ../results/bfs_timing.csv)
     python3 plots/plot_bfs_timing.py --csv results/bfs_timing.csv --row -1
     python3 plots/plot_bfs_timing.py --csv results/bfs_timing.csv --infile_mtx rand600.mtx --pe_grid 8x8
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# bfs_timing.py lives one directory up (bool_diag_spmv/, this script's own
# parent) -- add it to sys.path so this import works whether this script is
# run standalone or imported by run_bfs.py (which already adds plots/ to
# its own sys.path, see its own top-of-file comment).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bfs_timing import H2D_PARTS  # pylint: disable=wrong-import-position

# dataviz skill's validated categorical palette (references/palette.md) --
# light-mode hexes. Three round segments (local_compute, local_term_cond,
# communication) plus h2d/d2h's own two hues -- five slots total, all
# solid (no more per-relay-sub-phase shading now that the relay isn't
# broken out on its own).
ROUND_SEGMENT_COLORS = {
    "local_compute": "#eda100",     # yellow
    "local_term_cond": "#4a3aa7",   # violet
    "communication": "#e34948",     # red -- catch-all remainder, see module docstring
}
ROUND_SEGMENT_LABELS = {
    "local_compute": "local_compute",
    "local_term_cond": "local_term_cond",
    "communication": "communication (bcast+reduce+relay)",
}

# h2d/d2h aren't phases within a round -- standalone bars either side of the
# round bars, so they get their own two remaining unused categorical slots
# (magenta, orange), not a round-segment color. h2d_matrix/h2d_seed share
# the magenta hue (2 shades, "one hue family = related sub-parts") since
# they're both "h2d", just split per Graph500's construction-vs-per-search
# distinction (see GRAPH500_BENCHMARK.md).
H2D_BASE_HEX = "#e87ba4"  # magenta
D2H_COLOR = "#eb6834"  # orange


def _hex_to_rgb(h):
  h = h.lstrip("#")
  return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb):
  return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)


def hue_shades(base_hex, n):
  """n shades of base_hex, light->dark, for a group of related sub-parts
  (h2d's matrix/seed split) -- same hue family communicates 'these belong
  together', increasing darkness communicates chronological order within
  the group."""
  base = np.array(_hex_to_rgb(base_hex))
  white = np.array([1.0, 1.0, 1.0])
  black = np.array([0.0, 0.0, 0.0])
  # first shade tinted toward white (~35%), last shade shaded toward black
  # (~25%) -- keeps every step clearly non-white/non-black while spanning a
  # visible light->dark range.
  fracs = np.linspace(0.35, -0.25, n)
  out = []
  for f in fracs:
    if f >= 0:
      out.append(_rgb_to_hex(base * (1 - f) + white * f))
    else:
      out.append(_rgb_to_hex(base * (1 + f) + black * (-f)))
  return out


H2D_COLORS = dict(zip(H2D_PARTS, hue_shades(H2D_BASE_HEX, len(H2D_PARTS))))

TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def parse_cycle_list(s):
  return np.array([int(v) for v in s.split(";")], dtype=np.int64)


def default_out_path(plots_dir, matrix, pe_grid, source, channels):
  """plots_dir: the plots/ folder itself (this script's own directory when
  called from here; bool_diag_spmv/plots when called from run_bfs.py,
  which lives one directory up from here)."""
  matrix_stem = os.path.splitext(matrix)[0]
  timing_dir = os.path.join(plots_dir, "timing")
  return os.path.join(timing_dir, f"timing_{matrix_stem}_{pe_grid}_src{source}_ch{channels}.png")


def plot_timing_row(row, out_path):
  """row: a dict with the same keys bfs_timing.csv's header has (either
  read back via csv.DictReader, or the in-memory dict run_bfs.py just
  built before writing it) -- values may be str (from CSV) or native
  types (from run_bfs.py's own dict); everything is cast explicitly below
  so both sources work unmodified."""
  rounds_completed = int(row["rounds_completed"])
  matrix = row["infile_mtx"]
  pe_grid = row["pe_grid"]
  source = row["source"]
  channels = row["channels"]

  round_duration = parse_cycle_list(row["round_duration_cycles"]).astype(float)
  total_runtime_cycles = int(row["total_runtime_cycles"])
  assert len(round_duration) == rounds_completed, (
      f"round_duration_cycles has {len(round_duration)} entries, expected "
      f"rounds_completed={rounds_completed}")

  local_compute = parse_cycle_list(row["local_compute_max_cycles"]).astype(float)
  local_term_cond = parse_cycle_list(row["local_term_cond_max_cycles"]).astype(float)
  # remainder: everything else in the round (visited_bcast, vertical_bcast,
  # the SpMV reduce, the 4-phase termination relay) -- not chronologically
  # ordered within the bar, see module docstring.
  communication = np.clip(round_duration - local_compute - local_term_cond, 0.0, None)

  h2d_min = {p: int(row[f"{p}_min_cycles"]) for p in H2D_PARTS}
  h2d_max = {p: int(row[f"{p}_max_cycles"]) for p in H2D_PARTS}
  d2h_min, d2h_max = int(row["d2h_min_cycles"]), int(row["d2h_max_cycles"])

  rounds = np.arange(rounds_completed)

  # h2d/d2h (one-shot transfers, tens of thousands of cycles) and the
  # per-round breakdown (hundreds to a couple thousand cycles) are two
  # measures of different scale -- cramming both onto one linear axis
  # would crush the round bars to an unreadable sliver under d2h's height.
  # Small multiples instead: h2d/d2h share their own y-scale in narrow
  # flanking panels, the round bars keep their own in the wide middle
  # panel (see dataviz skill's anti-patterns: "two measures of different
  # scale -> two charts / small multiples", never a dual y-axis).
  #
  # Panel widths are set in fixed INCHES (not just a ratio) so the flanking
  # h2d/d2h panels stay a constant width regardless of round count --
  # otherwise a many-round run's wide middle panel would squeeze the side
  # panels down proportionally, and their inline labels would start failing
  # the measure-first fit check purely from round count, not their own
  # value's size.
  per_bar_w_in = 1.3
  h2d_w_in = per_bar_w_in * len(H2D_PARTS)
  d2h_w_in = per_bar_w_in * 1.3  # a little extra breathing room for one bar alone
  round_w_in = max(0.95 * rounds_completed, 3.0)
  fig, (ax_h2d, ax_rounds, ax_d2h) = plt.subplots(
      1, 3, figsize=(h2d_w_in + round_w_in + d2h_w_in, 6.5),
      gridspec_kw={"width_ratios": [h2d_w_in, round_w_in, d2h_w_in], "wspace": 0.08})

  bar_width = 0.62
  candidate_labels = []  # (ax, text_obj, xpos, segment_bottom, segment_top) -- fit-checked below

  def add_solo_bar(ax, xpos, height, tick_y, color, label):
    ax.bar([xpos], [height], width=bar_width, color=color, edgecolor=SURFACE,
           linewidth=2, label=label, zorder=2)
    ax.plot([xpos - bar_width / 2 * 0.7, xpos + bar_width / 2 * 0.7], [tick_y, tick_y],
            color=TEXT_PRIMARY, linewidth=1.4, solid_capstyle="butt", zorder=3)
    txt = ax.text(xpos, height / 2, f"{int(height)}", ha="center", va="center",
                  fontsize=7, color="white", fontweight="bold", zorder=4)
    candidate_labels.append((ax, txt, xpos, 0.0, height))

  for i, part in enumerate(H2D_PARTS):
    add_solo_bar(ax_h2d, i, h2d_max[part], h2d_min[part], H2D_COLORS[part], part)
  add_solo_bar(ax_d2h, 0, d2h_max, d2h_min, D2H_COLOR, "d2h")
  transfer_ylim = 1.15 * max(*h2d_max.values(), d2h_max)
  ax_h2d.set_ylim(0, transfer_ylim)
  ax_d2h.set_ylim(0, transfer_ylim)
  ax_h2d.set_xlim(-0.8, len(H2D_PARTS) - 0.2)
  ax_d2h.set_xlim(-0.9, 0.9)
  ax_h2d.set_xticks(range(len(H2D_PARTS)))
  ax_h2d.set_xticklabels(["matrix", "seed"])
  ax_d2h.set_xticks([0])
  ax_d2h.set_xticklabels(["d2h"])

  bottom = np.zeros(rounds_completed)
  segment_values = {
      "local_compute": local_compute,
      "local_term_cond": local_term_cond,
      "communication": communication,
  }
  for name, heights in segment_values.items():
    color = ROUND_SEGMENT_COLORS[name]
    # no edge/gap between stacked segments -- each one is literally
    # end-to-end time of a phase, contiguous with its neighbors, not a
    # separately-bounded block with dead time in between.
    ax_rounds.bar(rounds, heights, width=bar_width, bottom=bottom, color=color,
                  edgecolor=color, linewidth=0, label=ROUND_SEGMENT_LABELS[name], zorder=2)

    # direct label: just the cycle count (identity already comes from the
    # legend + the segment's own color) -- candidate only, fit-checked
    # below against the segment's actual rendered size once the figure is
    # laid out (see marks-and-anatomy.md: "measure first", never overflow).
    total_heights = bottom + heights
    for r in rounds:
      txt = ax_rounds.text(r, bottom[r] + heights[r] / 2, f"{int(heights[r])}",
                            ha="center", va="center", fontsize=7, color="white",
                            fontweight="bold", zorder=4)
      candidate_labels.append((ax_rounds, txt, r, bottom[r], total_heights[r]))
    bottom = total_heights

  ax_rounds.set_xticks(rounds)
  ax_rounds.set_xticklabels([f"round {r}" for r in rounds])

  # measure-first pass: a label only survives if its rendered bounding box
  # actually fits inside its own segment's rectangle (with a little
  # padding) -- otherwise remove it and let the legend + color carry
  # identity/magnitude for that segment instead of clipping text.
  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()
  pad_px = 2.0
  for ax, txt, xpos, seg_bottom, seg_top in candidate_labels:
    bbox = txt.get_window_extent(renderer=renderer)
    (x0_disp, y0_disp) = ax.transData.transform((xpos - bar_width / 2, seg_bottom))
    (x1_disp, y1_disp) = ax.transData.transform((xpos + bar_width / 2, seg_top))
    fits_w = (bbox.width + 2 * pad_px) <= (x1_disp - x0_disp)
    fits_h = (bbox.height + 2 * pad_px) <= (y1_disp - y0_disp)
    if not (fits_w and fits_h):
      txt.remove()

  ax_h2d.set_ylabel("cycles")
  sum_rounds = int(round_duration.sum())
  diff_pct = 100.0 * (total_runtime_cycles - sum_rounds) / total_runtime_cycles
  fig.suptitle(f"Per-round phase timing -- {matrix}, {pe_grid} grid, source={source}, "
               f"channels={channels}\n"
               f"n={row['n']}, nnz={row['nnz']}, {rounds_completed} rounds "
               "(bar height = round_duration_cycles, straggler PE span)\n"
               f"round bars sum to {sum_rounds} cycles vs total_runtime_cycles={total_runtime_cycles} "
               f"({diff_pct:+.1f}% -- gap is the first round's one-time fabric fill delay)",
               fontsize=10)
  for ax in (ax_h2d, ax_rounds, ax_d2h):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=TEXT_MUTED)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
  # h2d/d2h share transfer_ylim -- only the left panel needs the scale
  # printed; repeating it on the right would just be clutter (they're
  # visibly the same height range).
  ax_d2h.spines["left"].set_visible(False)
  ax_d2h.tick_params(left=False, labelleft=False)
  fig.patch.set_facecolor(SURFACE)

  handles, labels = [], []
  for ax in (ax_h2d, ax_rounds, ax_d2h):
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
  fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8,
             bbox_to_anchor=(0.5, -0.02))
  plt.tight_layout(rect=[0, 0.16, 1, 0.94])

  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  plt.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)
  print(f"saved timing plot to {out_path}")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--csv", default=None, help="bfs_timing.csv path (default: "
                                                    "../results/bfs_timing.csv, a sibling of "
                                                    "this script's own plots/ directory)")
  parser.add_argument("--row", type=int, default=-1,
                       help="which CSV row to plot (0-indexed, default: -1 = last/most recent). "
                            "Ignored if --infile_mtx/--pe_grid select exactly one row.")
  parser.add_argument("--infile_mtx", default=None,
                       help="filter to rows whose infile_mtx matches this basename")
  parser.add_argument("--pe_grid", default=None, help="filter to rows with this pe_grid, e.g. 8x8")
  parser.add_argument("--channels", type=int, default=None,
                       help="filter to rows with this --channels value (bench_timing.py's I/O "
                            "channel count)")
  parser.add_argument("--out", default=None, help="output PNG path (default: plots/timing_"
                                                    "<matrix>_<grid>_src<N>.png)")
  return parser.parse_args()


def select_row(rows, args):
  filtered = rows
  if args.infile_mtx is not None:
    filtered = [r for r in filtered if r["infile_mtx"] == args.infile_mtx]
  if args.pe_grid is not None:
    filtered = [r for r in filtered if r["pe_grid"] == args.pe_grid]
  if args.channels is not None:
    filtered = [r for r in filtered if int(r["channels"]) == args.channels]
  if args.infile_mtx is not None or args.pe_grid is not None or args.channels is not None:
    assert filtered, "no CSV rows match the given --infile_mtx/--pe_grid/--channels filters"
    if len(filtered) > 1:
      print(f"[[ {len(filtered)} rows match the filters -- using the most recent; "
            "narrow further or use --row if you meant a specific one ]]")
    return filtered[-1]
  assert rows, "CSV has no rows to plot"
  return rows[args.row]


def main():
  args = parse_args()

  csv_path = args.csv
  if csv_path is None:
    # results/ lives one directory up (bool_diag_spmv/results/), a sibling
    # of this script's own plots/ directory.
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "results", "bfs_timing.csv")
  with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

  row = select_row(rows, args)
  out_path = args.out or default_out_path(
      os.path.dirname(os.path.abspath(__file__)), row["infile_mtx"], row["pe_grid"],
      row["source"], row["channels"])
  plot_timing_row(row, out_path)


if __name__ == "__main__":
  main()