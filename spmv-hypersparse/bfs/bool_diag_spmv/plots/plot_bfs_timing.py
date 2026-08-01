#!/usr/bin/env python3
""" plot a single bfs_timing.csv row (one run_bfs.py run) as a stacked
  bar chart: one bar per BFS round (left to right), bracketed by two
  standalone "h2d" bars (h2d_matrix, h2d_seed -- see bfs_timing.py's
  H2D_PARTS / docs/GRAPH500_BENCHMARK.md for the Graph500-motivated split between
  the one-time matrix-structure upload and the per-search source-seed
  upload) before round 0, and two standalone bars after the last round:
  "resolve" (mpi_x.reduce_select_any()'s one-time on-device parent
  resolution, Phase B of the on-device parent resolution plan -- see
  bool_pe.csl's term_col_bcast_done()) then "d2h" (parent_local_buf
  readback -- the actual BFS output, not run_bfs.py's own
  rounds_completed/ts_buf instrumentation reads), in that chronological
  order -- the one-shot host<->device transfers (plus the one-shot
  on-device reduce immediately preceding them) that aren't part of any
  round.

  Round bar height = bfs_timing.compute_round_summary's round_duration_cycles
  (straggler-PE span from that round's TS_VBCAST_ISSUE to its
  TS_TERM_COL_BCAST_DONE) -- NOT a sum of individually skew-adjusted
  sub-phases. docs/GRAPH500_BENCHMARK.md sections 10-13 found that decomposing
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

  total_runtime_cycles (the straggler-PE span from round 0's very first
  TS_VBCAST_ISSUE to the last round's TS_TERM_COL_BCAST_DONE) is still in
  bfs_timing.csv, just not compared against the round bars' own sum in the
  suptitle any more -- that comparison is inherently noisy once a phase
  with wide per-PE variance (transpose_structure()) is involved, since each
  bar segment here is its own independent max-across-PEs and can come from
  a different PE than total_runtime_cycles' one coherent straggler (see
  run_gap_diagnostic.py for a worked confirmation this isn't double-
  counting, just that mismatch).

  local_compute's own segment additionally gets two tick-mark indicators
  (solid = min, dashed = avg, both across PEs that round) on top of its
  max-height segment -- the other two segments don't get this treatment
  (local_term_cond is a diagonal-only fixed cost with little PE variance to
  show; communication is a remainder, not a directly-measured quantity).

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
from matplotlib.lines import Line2D

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
# distinction (see docs/GRAPH500_BENCHMARK.md).
H2D_BASE_HEX = "#e87ba4"  # magenta
D2H_COLOR = "#eb6834"  # orange

# mpi_x.reduce_select_any()'s one-time end-of-run parent-resolution cost
# (Phase B of the on-device parent resolution plan) -- also a one-shot
# transfer-adjacent cost, not a per-round phase (it runs in
# term_col_bcast_done(), strictly after the last round and strictly before
# the host's own d2h read begins -- see run_bfs.py's own parent_resolve_cycles
# readback comment), so it's drawn in the same ax_d2h panel as d2h, sharing
# its y-scale, immediately to d2h's left in chronological order. Blue --
# palette.md's slot 1, the lowest-index unused slot in the validated
# 8-color categorical order (this file had already used slots 2/4/5/7/8;
# blue+orange (slots 1+2) is one of the palette's own documented passing
# ADJACENT pairs, the exact adjacency this panel needs since the two bars
# sit right next to each other).
PARENT_RESOLVE_COLOR = "#2a78d6"


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

# transpose_structure()'s one-time direction-optimizing-BFS cost (see the
# plan): NOT a sub-part of local_compute (it runs in term_col_bcast_done(),
# not compute()) -- stacked as its own segment on the round bars (see
# plot_timing_row), deliberately its own distinct hue, not a local_compute
# shade, so it doesn't read as "part of local_compute". transpose_*_cycles
# is zero in every round except whichever one the top-down -> bottom-up
# switch actually fires in (see run_bfs.py), so this segment is empty
# everywhere but that one round.
TRANSPOSE_COLOR = "#2a9d8f"  # teal -- distinct from every other hue family in use

# Also stacked directly onto the round-bars panel's own switch-round bar
# (see plot_timing_row) -- round_duration_cycles (what those bars are built
# from) brackets TS_VBCAST_ISSUE..TS_TERM_COL_BCAST_DONE, which does NOT
# span transpose_structure()'s own gap (see bool_pe.csl's
# term_col_bcast_done()), so without this the round bars would visibly sum
# to less than total_runtime_cycles -- an unaccounted-for gap the reader
# has no way to attribute from the bars alone, title text notwithstanding.
ROUND_SEGMENT_COLORS["transpose"] = TRANSPOSE_COLOR
ROUND_SEGMENT_LABELS["transpose"] = "transpose_structure() (one-time, switch round only)"

TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def parse_cycle_list(s):
  return np.array([int(v) for v in s.split(";")], dtype=np.int64)


def parse_cycle_list_float(s):
  """Like parse_cycle_list, for *_avg_cycles columns -- decode_phase_row
  formats those as f"{v:.1f}" (a real mean, not an integer cycle count)."""
  return np.array([float(v) for v in s.split(";")], dtype=np.float64)


def add_stat_ticks(ax, xpos, min_y, avg_y, w):
  """Solid tick at min_y, dashed tick at avg_y (both across PEs) -- laid on
  top of a bar/segment whose own height is already that phase's max across
  PEs, so min <= avg <= height always. Shared by plot_timing_row's
  local_compute segment and plot_compute_split_row's grouped bars."""
  half = w / 2 * 0.7
  ax.plot([xpos - half, xpos + half], [min_y, min_y],
          color=TEXT_PRIMARY, linewidth=1.4, solid_capstyle="butt", zorder=5)
  ax.plot([xpos - half, xpos + half], [avg_y, avg_y],
          color=TEXT_PRIMARY, linewidth=1.4, linestyle="--", dash_capstyle="butt", zorder=5)


STAT_TICK_HANDLES = [
    Line2D([0], [0], color=TEXT_PRIMARY, linewidth=1.4, label="min (across PEs)"),
    Line2D([0], [0], color=TEXT_PRIMARY, linewidth=1.4, linestyle="--", label="avg (across PEs)"),
]


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
  # profiled_rounds (NOT rounds_completed) is authoritative for every
  # per-round array's length below: bool_pe.csl's ts_buf (and thus every
  # per-round phase timestamp) only has slots for max_rounds rounds --
  # round_duration_cycles/local_compute_*_cycles/etc. are silently
  # TRUNCATED to profiled_rounds = min(rounds_completed, max_rounds) once a
  # BFS runs deeper than max_rounds (see decode_phase_row's own WARNING,
  # bfs_timing.py) -- asserting against rounds_completed here would reject
  # every such (deliberately truncated, still-correct-for-search_time_cycles)
  # row instead of just plotting the rounds actually profiled.
  profiled_rounds = len(round_duration)

  local_compute = parse_cycle_list(row["local_compute_max_cycles"]).astype(float)
  local_compute_min = parse_cycle_list(row["local_compute_min_cycles"]).astype(float)
  local_compute_avg = parse_cycle_list_float(row["local_compute_avg_cycles"])
  local_term_cond = parse_cycle_list(row["local_term_cond_max_cycles"]).astype(float)
  # remainder: everything else in the round (visited_bcast, vertical_bcast,
  # the SpMV reduce, the 4-phase termination relay) -- not chronologically
  # ordered within the bar, see module docstring.
  communication = np.clip(round_duration - local_compute - local_term_cond, 0.0, None)

  # transpose_structure()'s one-time cost (zero on every round except the
  # switch round, if any) -- stacked as its own segment on the round bars.
  transpose_heights = parse_cycle_list(row["transpose_max_cycles"]).astype(float)
  assert len(transpose_heights) == profiled_rounds, (
      f"transpose_max_cycles has {len(transpose_heights)} entries, expected "
      f"profiled_rounds={profiled_rounds}")

  h2d_min = {p: int(row[f"{p}_min_cycles"]) for p in H2D_PARTS}
  h2d_max = {p: int(row[f"{p}_max_cycles"]) for p in H2D_PARTS}
  d2h_min, d2h_max = int(row["d2h_min_cycles"]), int(row["d2h_max_cycles"])
  parent_resolve_min = int(row["parent_resolve_min_cycles"])
  parent_resolve_max = int(row["parent_resolve_max_cycles"])

  rounds = np.arange(profiled_rounds)

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
  # parent_resolve + d2h now share this panel (2 bars, not 1) -- same
  # per-bar width as h2d's panel, plus a little breathing room.
  d2h_w_in = per_bar_w_in * 2 + 0.3
  round_w_in = max(0.95 * profiled_rounds, 3.0)
  fig, (ax_h2d, ax_rounds, ax_d2h) = plt.subplots(
      1, 3, figsize=(h2d_w_in + round_w_in + d2h_w_in, 6.5),
      gridspec_kw={"width_ratios": [h2d_w_in, round_w_in, d2h_w_in],
                   "wspace": 0.1})

  bar_width = 0.62
  # (ax, text_obj, xpos, segment_bottom, segment_top, bar_w) -- fit-checked below.
  candidate_labels = []

  def add_solo_bar(ax, xpos, height, tick_y, color, label):
    ax.bar([xpos], [height], width=bar_width, color=color, edgecolor=SURFACE,
           linewidth=2, label=label, zorder=2)
    ax.plot([xpos - bar_width / 2 * 0.7, xpos + bar_width / 2 * 0.7], [tick_y, tick_y],
            color=TEXT_PRIMARY, linewidth=1.4, solid_capstyle="butt", zorder=3)
    txt = ax.text(xpos, height / 2, f"{int(height)}", ha="center", va="center",
                  fontsize=7, color="white", fontweight="bold", zorder=4)
    candidate_labels.append((ax, txt, xpos, 0.0, height, bar_width))

  for i, part in enumerate(H2D_PARTS):
    add_solo_bar(ax_h2d, i, h2d_max[part], h2d_min[part], H2D_COLORS[part], part)
  # parent_resolve drawn first (xpos=0), d2h second (xpos=1) -- matches
  # their actual chronological order (the on-device reduce finishes before
  # the host's own d2h read begins, see run_bfs.py's parent_resolve_cycles
  # readback comment), same left-to-right-is-chronological convention the
  # h2d panel's matrix->seed ordering already uses.
  add_solo_bar(ax_d2h, 0, parent_resolve_max, parent_resolve_min, PARENT_RESOLVE_COLOR,
               "parent_resolve")
  add_solo_bar(ax_d2h, 1, d2h_max, d2h_min, D2H_COLOR, "d2h")
  transfer_ylim = 1.15 * max(*h2d_max.values(), d2h_max, parent_resolve_max)
  ax_h2d.set_ylim(0, transfer_ylim)
  ax_d2h.set_ylim(0, transfer_ylim)
  ax_h2d.set_xlim(-0.8, len(H2D_PARTS) - 0.2)
  ax_d2h.set_xlim(-0.8, 1.8)
  ax_h2d.set_xticks(range(len(H2D_PARTS)))
  ax_h2d.set_xticklabels(["matrix", "seed"])
  ax_d2h.set_xticks([0, 1])
  # short tick labels (matches "matrix"/"seed"/"d2h"'s own single-word
  # convention -- "parent_resolve" would be wide enough to collide with
  # the "d2h" tick right next to it at this panel width); the legend still
  # carries the full "parent_resolve" name via add_solo_bar's own label=.
  ax_d2h.set_xticklabels(["resolve", "d2h"])

  bottom = np.zeros(profiled_rounds)
  segment_values = {
      "local_compute": local_compute,
      "local_term_cond": local_term_cond,
      "communication": communication,
      # stacked last (zero-height, hence invisible, on every round except
      # the switch round) -- see ROUND_SEGMENT_COLORS["transpose"]'s own
      # comment for why this needs to be here at all.
      "transpose": transpose_heights,
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
      candidate_labels.append((ax_rounds, txt, r, bottom[r], total_heights[r], bar_width))

    if name == "local_compute":
      # local_compute is always the first-stacked segment (bottom == 0
      # here), so its min/avg values are already absolute y-positions.
      for r in rounds:
        add_stat_ticks(ax_rounds, r, local_compute_min[r], local_compute_avg[r], bar_width)

    bottom = total_heights

  # direction-optimizing BFS Phase D (see the plan): label each round's own
  # xtick with which traversal strategy it actually used, when that column
  # is present (older CSV rows / runs predating this feature simply won't
  # have it -- .get() + truthiness check covers both a missing key and an
  # empty string the same way).
  direction_history_row = row.get("direction_history")
  if direction_history_row:
    directions = parse_cycle_list(direction_history_row)
    round_labels = [f"round {r}\n({'BU' if directions[r] else 'TD'})" for r in rounds]
  else:
    round_labels = [f"round {r}" for r in rounds]

  ax_rounds.set_xticks(rounds)
  ax_rounds.set_xticklabels(round_labels)

  # measure-first pass: a label only survives if its rendered bounding box
  # actually fits inside its own segment's rectangle (with a little
  # padding) -- otherwise remove it and let the legend + color carry
  # identity/magnitude for that segment instead of clipping text.
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

  ax_h2d.set_ylabel("cycles", labelpad=8)
  rounds_suffix = (f"{rounds_completed} rounds (only first {profiled_rounds} profiled -- "
                    f"bump max_rounds for full detail)" if profiled_rounds < rounds_completed
                    else f"{rounds_completed} rounds")
  fig.suptitle(f"Per-round phase timing -- {matrix}, {pe_grid} grid, source={source}, "
               f"channels={channels}\n"
               f"n={row['n']}, nnz={row['nnz']}, {rounds_suffix} "
               "(bar height = round_duration_cycles, + transpose_structure() on the switch round)",
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
  handles += STAT_TICK_HANDLES
  labels += [h.get_label() for h in STAT_TICK_HANDLES]
  fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8,
             bbox_to_anchor=(0.5, -0.16), columnspacing=1.8, handletextpad=0.6, labelspacing=1.0)
  plt.tight_layout(rect=[0, 0.26, 1, 0.93])

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
  plots_dir = os.path.dirname(os.path.abspath(__file__))
  out_path = args.out or default_out_path(
      plots_dir, row["infile_mtx"], row["pe_grid"], row["source"], row["channels"])
  plot_timing_row(row, out_path)


if __name__ == "__main__":
  main()