#!/usr/bin/env python3
"""Plot a single bfs_timing.csv row (one run_bfs.py run) as a stacked bar
chart: one bar per BFS round, bracketed by two standalone "h2d" bars
(h2d_matrix, h2d_seed -- see H2D_PARTS) before round 0, and "resolve"
(mpi_x.reduce_select_any()'s one-time end-of-run parent resolution) then
"d2h" (parent_local_buf readback) after the last round.

Round bar height = round_duration_cycles (that round's straggler-PE
span), decomposed only into what's reliably, locally measurable per PE
with no cross-PE synchronization ambiguity: local_compute (the boolean
SpMV multiply) and local_term_cond (diagonal-only masking), both
bracketed by a PE's own entry/exit timestamps. Everything else in the
round (broadcasts, the SpMV reduce, the termination relay) is lumped
into one "communication" segment, computed as a remainder -- decomposing
those phases into "real cost" vs. cross-PE wait was tried and abandoned
as unreliable (docs/GRAPH500_BENCHMARK.md sections 10-13). The segments
are stacked in a fixed order for a stable legend, not a literal timeline.

total_runtime_cycles is still in the CSV, just no longer compared
against the round bars' own sum -- that comparison gets noisy once
transpose_structure()'s high per-PE variance is involved (see
run_gap_diagnostic.py).

local_compute additionally gets min/avg tick marks across PEs on top of
its max-height segment.

Importable (plot_timing_row(row, out_path)) -- run_bfs.py calls it right
after appending a row. Also runnable standalone to re-plot an existing
CSV row:

   python3 plots/plot_bfs_timing.py --csv ../results/sim/bfs_timing.csv --row -1
   python3 plots/plot_bfs_timing.py --csv ../results/sim/bfs_timing.csv --infile_mtx rand600.mtx --pe_grid 8x8
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# bfs_timing.py lives in ../implementation/, a sibling of this script's
# plots/ directory -- add it to sys.path so this import works whether run
# standalone or imported by run_bfs.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "implementation"))
from bfs_timing import H2D_PARTS  # pylint: disable=wrong-import-position

# Categorical palette (light-mode hexes): three round segments plus h2d/d2h's
# own two hues.
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
# round bars, so they get their own two colors, not a round-segment color.
# h2d_matrix/h2d_seed share the magenta hue (one hue family = related
# sub-parts) since they're both "h2d", just split by Graph500's
# construction-vs-per-search distinction.
H2D_BASE_HEX = "#e87ba4"  # magenta
D2H_COLOR = "#eb6834"  # orange

# mpi_x.reduce_select_any()'s one-time end-of-run parent-resolution cost:
# also a one-shot transfer-adjacent cost, not a per-round phase, so it's
# drawn in the same ax_d2h panel as d2h, sharing its y-scale, immediately
# to d2h's left (matching their actual chronological order).
PARENT_RESOLVE_COLOR = "#2a78d6"  # blue


def _hex_to_rgb(h):
  h = h.lstrip("#")
  return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb):
  return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)


def hue_shades(base_hex, n):
  """n shades of base_hex, light->dark, for a group of related sub-parts
  (h2d's matrix/seed split): same hue family says "these belong together",
  increasing darkness says "chronological order within the group"."""
  base = np.array(_hex_to_rgb(base_hex))
  white = np.array([1.0, 1.0, 1.0])
  black = np.array([0.0, 0.0, 0.0])
  # First shade tinted toward white, last toward black -- stays clearly
  # non-white/non-black while spanning a visible light->dark range.
  fracs = np.linspace(0.35, -0.25, n)
  out = []
  for f in fracs:
    if f >= 0:
      out.append(_rgb_to_hex(base * (1 - f) + white * f))
    else:
      out.append(_rgb_to_hex(base * (1 + f) + black * (-f)))
  return out


H2D_COLORS = dict(zip(H2D_PARTS, hue_shades(H2D_BASE_HEX, len(H2D_PARTS))))

# transpose_structure()'s one-time direction-optimizing-BFS cost: not a
# sub-part of local_compute (it runs in term_col_bcast_done(), not
# compute()), so it gets its own distinct hue rather than a local_compute
# shade. transpose_*_cycles is zero in every round except whichever one
# the top-down -> bottom-up switch fires in, so this segment is empty
# everywhere but that one round.
TRANSPOSE_COLOR = "#2a9d8f"  # teal

# Stacked onto the round bars (see plot_timing_row) so the switch round's
# bar doesn't visibly sum to less than total_runtime_cycles -- without it,
# transpose_structure()'s gap would be an unaccounted-for span the reader
# has no way to attribute from the bars alone.
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
  """Like parse_cycle_list, for *_avg_cycles columns (a real mean, not an
  integer cycle count)."""
  return np.array([float(v) for v in s.split(";")], dtype=np.float64)


def add_stat_ticks(ax, xpos, min_y, avg_y, w):
  """Solid tick at min_y, dashed tick at avg_y (both across PEs), laid on
  top of a bar/segment whose own height is already that phase's max
  across PEs (so min <= avg <= height always)."""
  half = w / 2 * 0.7
  ax.plot([xpos - half, xpos + half], [min_y, min_y],
          color=TEXT_PRIMARY, linewidth=1.4, solid_capstyle="butt", zorder=5)
  ax.plot([xpos - half, xpos + half], [avg_y, avg_y],
          color=TEXT_PRIMARY, linewidth=1.4, linestyle="--", dash_capstyle="butt", zorder=5)


STAT_TICK_HANDLES = [
    Line2D([0], [0], color=TEXT_PRIMARY, linewidth=1.4, label="min (across PEs)"),
    Line2D([0], [0], color=TEXT_PRIMARY, linewidth=1.4, linestyle="--", label="avg (across PEs)"),
]


def results_variant(csv_path):
  """"hw" if csv_path resolves under a results/hw/ directory, else "sim" --
  lets the caller route output into the matching results/hw or
  results/sim subfolder."""
  return "hw" if "hw" in os.path.normpath(os.path.abspath(csv_path)).split(os.sep) else "sim"


def default_out_path(results_dir, matrix, pe_grid, source, channels):
  """results_dir: the results/hw or results/sim folder (see
  results_variant)."""
  matrix_stem = os.path.splitext(matrix)[0]
  timing_dir = os.path.join(results_dir, "timing")
  return os.path.join(timing_dir, f"timing_{matrix_stem}_{pe_grid}_src{source}_ch{channels}.png")


def plot_timing_row(row, out_path):
  """row: a dict with the same keys bfs_timing.csv's header has, either
  read back via csv.DictReader or the in-memory dict run_bfs.py just
  built. Values may be str or native types; everything is cast explicitly
  below so both sources work unmodified."""
  rounds_completed = int(row["rounds_completed"])
  matrix = row["infile_mtx"]
  pe_grid = row["pe_grid"]
  source = row["source"]
  channels = row["channels"]

  round_duration = parse_cycle_list(row["round_duration_cycles"]).astype(float)
  # profiled_rounds, not rounds_completed, is authoritative for every
  # per-round array's length: ts_buf only has slots for max_rounds rounds,
  # so these arrays are silently truncated once a BFS runs deeper than
  # that (see decode_phase_row's WARNING in bfs_timing.py) -- we plot
  # what was actually profiled, not what was asserted to exist.
  profiled_rounds = len(round_duration)

  local_compute = parse_cycle_list(row["local_compute_max_cycles"]).astype(float)
  local_compute_min = parse_cycle_list(row["local_compute_min_cycles"]).astype(float)
  local_compute_avg = parse_cycle_list_float(row["local_compute_avg_cycles"])
  local_term_cond = parse_cycle_list(row["local_term_cond_max_cycles"]).astype(float)
  # Remainder: everything else in the round (see module docstring).
  communication = np.clip(round_duration - local_compute - local_term_cond, 0.0, None)

  # transpose_structure()'s one-time cost (zero except on the switch round).
  transpose_heights = parse_cycle_list(row["transpose_max_cycles"]).astype(float)
  assert len(transpose_heights) == profiled_rounds, (
      f"transpose_max_cycles has {len(transpose_heights)} entries, expected "
      f"profiled_rounds={profiled_rounds}")

  # h2d_matrix/h2d_seed/d2h: sync-corrected cross-PE span only (see
  # bfs_timing.read_sync_corrected_span) -- no per-PE min/avg to show.
  h2d_span = {p: int(row[f"{p}_span_cycles"]) for p in H2D_PARTS}
  d2h_span = int(row["d2h_span_cycles"])
  # parent_resolve is on-device only, so it keeps the per-PE min/max/avg.
  parent_resolve_min = int(row["parent_resolve_min_cycles"])
  parent_resolve_max = int(row["parent_resolve_max_cycles"])

  rounds = np.arange(profiled_rounds)

  # h2d/d2h (tens of thousands of cycles) and the per-round breakdown
  # (hundreds to a couple thousand) are two measures of different scale --
  # cramming both onto one linear axis would crush the round bars.
  # Small multiples instead: h2d/d2h share their own y-scale in narrow
  # flanking panels, the round bars keep their own in the wide middle one.
  #
  # Panel widths are fixed INCHES, not just a ratio, so the flanking
  # panels stay a constant width regardless of round count -- otherwise a
  # many-round run would squeeze them down and start failing their label
  # fit-check purely from round count.
  per_bar_w_in = 1.3
  h2d_w_in = per_bar_w_in * len(H2D_PARTS)
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
    if tick_y is not None:
      ax.plot([xpos - bar_width / 2 * 0.7, xpos + bar_width / 2 * 0.7], [tick_y, tick_y],
              color=TEXT_PRIMARY, linewidth=1.4, solid_capstyle="butt", zorder=3)
    txt = ax.text(xpos, height / 2, f"{int(height)}", ha="center", va="center",
                  fontsize=7, color="white", fontweight="bold", zorder=4)
    candidate_labels.append((ax, txt, xpos, 0.0, height, bar_width))

  for i, part in enumerate(H2D_PARTS):
    add_solo_bar(ax_h2d, i, h2d_span[part], None, H2D_COLORS[part], part)
  # parent_resolve first (xpos=0), d2h second: matches their actual
  # chronological order, same left-to-right convention as h2d's matrix->seed.
  add_solo_bar(ax_d2h, 0, parent_resolve_max, parent_resolve_min, PARENT_RESOLVE_COLOR,
               "parent_resolve")
  add_solo_bar(ax_d2h, 1, d2h_span, None, D2H_COLOR, "d2h")
  transfer_ylim = 1.15 * max(*h2d_span.values(), d2h_span, parent_resolve_max)
  ax_h2d.set_ylim(0, transfer_ylim)
  ax_d2h.set_ylim(0, transfer_ylim)
  ax_h2d.set_xlim(-0.8, len(H2D_PARTS) - 0.2)
  ax_d2h.set_xlim(-0.8, 1.8)
  ax_h2d.set_xticks(range(len(H2D_PARTS)))
  ax_h2d.set_xticklabels(["matrix", "seed"])
  ax_d2h.set_xticks([0, 1])
  # Short tick labels ("parent_resolve" would collide with "d2h" at this
  # panel width) -- the legend still carries the full name.
  ax_d2h.set_xticklabels(["resolve", "d2h"])

  bottom = np.zeros(profiled_rounds)
  segment_values = {
      "local_compute": local_compute,
      "local_term_cond": local_term_cond,
      "communication": communication,
      # Stacked last (zero-height, invisible, except on the switch round --
      # see ROUND_SEGMENT_COLORS["transpose"] above).
      "transpose": transpose_heights,
  }
  for name, heights in segment_values.items():
    color = ROUND_SEGMENT_COLORS[name]
    # No edge/gap between stacked segments -- contiguous phase time, not
    # separately-bounded blocks.
    ax_rounds.bar(rounds, heights, width=bar_width, bottom=bottom, color=color,
                  edgecolor=color, linewidth=0, label=ROUND_SEGMENT_LABELS[name], zorder=2)

    # Direct label (cycle count); candidate only, fit-checked below against
    # the segment's actual rendered size.
    total_heights = bottom + heights
    for r in rounds:
      txt = ax_rounds.text(r, bottom[r] + heights[r] / 2, f"{int(heights[r])}",
                            ha="center", va="center", fontsize=7, color="white",
                            fontweight="bold", zorder=4)
      candidate_labels.append((ax_rounds, txt, r, bottom[r], total_heights[r], bar_width))

    if name == "local_compute":
      # local_compute is always first-stacked (bottom == 0), so its
      # min/avg values are already absolute y-positions.
      for r in rounds:
        add_stat_ticks(ax_rounds, r, local_compute_min[r], local_compute_avg[r], bar_width)

    bottom = total_heights

  # Label each round's xtick with which traversal strategy it used, when
  # that column is present (older CSV rows predating this feature won't
  # have it).
  direction_history_row = row.get("direction_history")
  if direction_history_row:
    directions = parse_cycle_list(direction_history_row)
    round_labels = [f"round {r}\n({'BU' if directions[r] else 'TD'})" for r in rounds]
  else:
    round_labels = [f"round {r}" for r in rounds]

  ax_rounds.set_xticks(rounds)
  ax_rounds.set_xticklabels(round_labels)

  # A label survives only if its rendered bounding box fits inside its
  # own segment's rectangle -- otherwise remove it and let the legend +
  # color carry identity for that segment instead of clipping text.
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
  # printed.
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


def parse_args(parser=None):
  """parser: an existing ArgumentParser to add these shared flags to (lets
  a caller like plot_bfs_timing_poster.py add its own flags first) --
  creates its own if not given."""
  if parser is None:
    parser = argparse.ArgumentParser()
  parser.add_argument("--csv", default=None,
                       help="bfs_timing.csv path (default: ../results/sim/bfs_timing.csv)")
  parser.add_argument("--row", type=int, default=-1,
                       help="which CSV row to plot (0-indexed, default: -1 = last/most recent). "
                            "Ignored if --infile_mtx/--pe_grid select exactly one row.")
  parser.add_argument("--infile_mtx", default=None,
                       help="filter to rows whose infile_mtx matches this basename")
  parser.add_argument("--pe_grid", default=None, help="filter to rows with this pe_grid, e.g. 8x8")
  parser.add_argument("--channels", type=int, default=None,
                       help="filter to rows with this --channels value (I/O channel count)")
  parser.add_argument("--out", default=None,
                       help="output PNG path (default: results/<hw|sim>/timing/"
                            "timing_<matrix>_<grid>_src<N>.png, see results_variant)")
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


def select_rows(rows, args):
  """Like select_row, but returns every matching row, not just the most
  recent -- for callers that aggregate repeated runs of the same config
  (e.g. plot_bfs_timing_poster.py's mean/std-across-runs mode). Falls back
  to a single row when no filters are given, same as select_row."""
  filtered = rows
  if args.infile_mtx is not None:
    filtered = [r for r in filtered if r["infile_mtx"] == args.infile_mtx]
  if args.pe_grid is not None:
    filtered = [r for r in filtered if r["pe_grid"] == args.pe_grid]
  if args.channels is not None:
    filtered = [r for r in filtered if int(r["channels"]) == args.channels]
  if args.infile_mtx is not None or args.pe_grid is not None or args.channels is not None:
    assert filtered, "no CSV rows match the given --infile_mtx/--pe_grid/--channels filters"
    return filtered
  assert rows, "CSV has no rows to plot"
  return [rows[args.row]]


def main():
  args = parse_args()

  csv_path = args.csv
  if csv_path is None:
    # results/sim/ is a sibling of this script's own plots/ directory --
    # a hw run always passes --csv explicitly.
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "results", "sim", "bfs_timing.csv")
  with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

  row = select_row(rows, args)
  results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", results_variant(csv_path))
  out_path = args.out or default_out_path(
      results_dir, row["infile_mtx"], row["pe_grid"], row["source"], row["channels"])
  plot_timing_row(row, out_path)


if __name__ == "__main__":
  main()