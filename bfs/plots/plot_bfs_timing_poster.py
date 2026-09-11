#!/usr/bin/env python3
"""Poster-simplified version of plot_bfs_timing.py's per-run stacked
timing chart: two panels instead of four, each on its own ms scale -- one
for per-round device work, one for the one-shot h2d_seed/resolve/d2h bars.

Dropped relative to plot_bfs_timing.py: h2d_matrix (kept only as a
--log-scale total), the local_compute breakdown panel, and every min/avg
stat tick -- bar height (max across PEs) is the only number shown, for a
cleaner poster figure. Round bars fuse local_compute+local_term_cond into
one "compute"
segment and lump the rest into "communication", colored to match
plot_grid_scale_heatmap.py's aqua/orange compute/communication poles.

Two panels, not one shared scale: h2d_seed/resolve/d2h are one-shot costs
2-3 orders of magnitude bigger than a round's device time, so sharing an
axis would crush the round bars to invisible slivers. Each panel is
auto-scaled to its own data; the scale difference is meant to read
visually, not via a caption.

Every bar (including the round stacks, previously drawn from a single
representative run) shows mean +/- std across every matching CSV row --
the round panel's error bars are expected to come out small, not absent;
the host-device/resolve bars are where real hardware jitter actually
shows up (docs/GRAPH500_BENCHMARK.md section 15).

--log-scale: an alternative single-panel, log-scale rendering (6 solo
bars, no per-round breakdown) -- the more common HPC/profiling convention
for a metric spanning many orders of magnitude, at the cost of bar
length no longer encoding magnitude linearly. Both modes are kept side by
side for direct comparison.

SVG only -- vector output meant to be embedded/rescaled into a poster.

How to run (aggregating every run of one config):
   python3 plots/plot_bfs_timing_poster.py --csv results/hw/bfs_timing.csv \\
       --infile_mtx rmat_s17_e16.balanced750x750.mtx --pe_grid 750x750
   python3 plots/plot_bfs_timing_poster.py --csv results/hw/bfs_timing.csv \\
       --infile_mtx rmat_s17_e16.balanced750x750.mtx --pe_grid 750x750 --log-scale
Or a single specific row (no --infile_mtx/--pe_grid):
   python3 plots/plot_bfs_timing_poster.py --csv results/hw/bfs_timing.csv --row -1
"""

import argparse
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "implementation"))
from bfs_timing import CLOCK_FREQ_HZ  # pylint: disable=wrong-import-position
from plot_bfs_timing import (  # pylint: disable=wrong-import-position
    BASELINE, GRIDLINE, H2D_BASE_HEX, PARENT_RESOLVE_COLOR, SURFACE, TEXT_MUTED, TEXT_PRIMARY,
    parse_args, parse_cycle_list, select_rows,
)

# Matches plot_grid_scale_heatmap.py's AQUA/ORANGE compute/communication
# poles exactly, so the two figures share one visual language.
COMPUTE_COLOR = "#1baf7a"  # aqua
COMMUNICATION_COLOR = "#eb6834"  # orange
# h2d_seed/h2d_matrix collapse to one color -- both already name themselves
# on their own x-tick, so color here only needs to say "Host to Device".
# resolve keeps its own PARENT_RESOLVE_COLOR/legend entry rather than
# folding into d2h's: it's on-device work, not an actual host transfer,
# even though it's grouped next to d2h positionally (see log-scale mode).
HOST_TO_DEVICE_COLOR = H2D_BASE_HEX  # magenta -- h2d_seed + h2d_matrix
DEVICE_TO_HOST_COLOR = "#eda100"  # yellow -- d2h only

# Same names/values as plot_grid_scale_heatmap.py/plot_balance_before_after.py
# so the three figure families read as one visual system. This figure's own
# axes-top override to 0.85 below is a tight_layout-incompatibility
# workaround, not a divergent convention.
SUPTITLE_FONTSIZE = 14
PANEL_TITLE_FONTSIZE = 12
LEGEND_FONTSIZE = 11
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 8
SUPTITLE_Y = 0.98
TOP_MARGIN = 0.88


def cycles_to_ms(cycles, clock_freq_hz):
  return cycles / clock_freq_hz * 1000.0


RMAT_SCALE_RE = re.compile(r"^rmat_s(\d+)_e\d+(?:\.balanced\d+x\d+)?\.mtx$")


def poster_title_input(matrix):
  """'Scale <scale>' for an RMAT input; falls back to the bare matrix stem
  for anything else (e.g. a SNAP graph)."""
  m = RMAT_SCALE_RE.match(matrix)
  return f"Scale {m.group(1)}" if m else os.path.splitext(matrix)[0]


def default_out_path(results_dir, matrix, pe_grid, source, channels, log_scale=False):
  matrix_stem = os.path.splitext(matrix)[0]
  timing_dir = os.path.join(results_dir, "hw", "timing-poster")
  suffix = "_logscale" if log_scale else ""
  return os.path.join(
      timing_dir, f"timing_poster_{matrix_stem}_{pe_grid}_src{source}_ch{channels}{suffix}.svg")


def mean_std_ms(rows, cycles_key, clock_freq_hz):
  """Mean + sample std dev (ddof=1, 0.0 for a single row) of an int cycle
  column across every row in `rows`, converted to ms. Used for the
  host-device/parent_resolve bars, which real hardware showed have
  genuine run-to-run variance (docs/GRAPH500_BENCHMARK.md section 15)."""
  values = cycles_to_ms(np.array([int(r[cycles_key]) for r in rows], dtype=np.float64),
                         clock_freq_hz)
  return float(values.mean()), (float(values.std(ddof=1)) if len(values) > 1 else 0.0)


def _mean_std(stack, axis):
  """Like mean_std_ms's mean+std, generalized from a flat per-row array to
  an ndarray along `axis` (per_round_compute_communication's
  per-row-per-round stacks)."""
  mean = stack.mean(axis=axis)
  std = stack.std(axis=axis, ddof=1) if stack.shape[axis] > 1 else np.zeros_like(mean)
  return mean, std


def per_round_compute_communication(rows, clock_freq_hz):
  """Per-row, per-round compute/communication arrays (ms), each shape
  (n_rows, n_rounds) -- the shared derivation (local_compute+local_term_cond,
  clip(round_duration - that, 0, None)) both poster modes build their own
  mean/std from. Every row must report the same round count for this
  column; asserts rather than silently truncating/broadcasting if that
  ever stops holding."""
  compute_per_row, communication_per_row = [], []
  for r in rows:
    local_compute = cycles_to_ms(
        parse_cycle_list(r["local_compute_max_cycles"]).astype(np.float64), clock_freq_hz)
    local_term_cond = cycles_to_ms(
        parse_cycle_list(r["local_term_cond_max_cycles"]).astype(np.float64), clock_freq_hz)
    round_duration = cycles_to_ms(
        parse_cycle_list(r["round_duration_cycles"]).astype(np.float64), clock_freq_hz)
    compute_per_row.append(local_compute + local_term_cond)
    communication_per_row.append(np.clip(round_duration - local_compute - local_term_cond, 0.0, None))
  lengths = {len(a) for a in compute_per_row}
  assert len(lengths) == 1, (
      f"rows have mismatched round counts {sorted(lengths)} -- can't average per-round "
      "compute/communication across runs with different round counts")
  return np.stack(compute_per_row), np.stack(communication_per_row)


def plot_timing_row_poster(rows, out_path, log_scale=False):
  """Dispatches to one of two renderings of the same data -- see
  _plot_timing_row_poster_linear/_log. `rows`: every CSV row for the same
  (infile_mtx, pe_grid) config -- see select_rows."""
  if log_scale:
    _plot_timing_row_poster_log(rows, out_path)
  else:
    _plot_timing_row_poster_linear(rows, out_path)


def _plot_timing_row_poster_linear(rows, out_path):
  """Small multiples: 3 separate linear-scale panels (rounds, h2d_seed/
  resolve/d2h, h2d_matrix), one per order of magnitude -- see module
  docstring. Every bar in every panel is a mean across all of `rows`,
  error bars included."""
  row = rows[-1]
  matrix = row["infile_mtx"]
  pe_grid = row["pe_grid"]
  clock_freq_hz = float(row.get("clock_freq_hz") or CLOCK_FREQ_HZ)

  # Fused: local_compute + local_term_cond, one "compute" segment (see
  # module docstring). compute/communication are per-round means across
  # `rows`; round_total_std is the error bar plotted on top of each
  # round's stack below.
  compute_stack, communication_stack = per_round_compute_communication(rows, clock_freq_hz)
  compute, _ = _mean_std(compute_stack, axis=0)
  communication, _ = _mean_std(communication_stack, axis=0)
  _, round_total_std = _mean_std(compute_stack + communication_stack, axis=0)
  # profiled_rounds, not rounds_completed, is authoritative -- see
  # plot_bfs_timing.py's comment on this truncation subtlety.
  profiled_rounds = len(compute)

  # Sync-corrected cross-PE span for h2d_matrix/h2d_seed/d2h. parent_resolve
  # is on-device only, so it keeps its per-PE max. Mean +/- std across `rows`.
  h2d_matrix_mean, h2d_matrix_std = mean_std_ms(rows, "h2d_matrix_span_cycles", clock_freq_hz)
  h2d_seed_mean, h2d_seed_std = mean_std_ms(rows, "h2d_seed_span_cycles", clock_freq_hz)
  parent_resolve_mean, parent_resolve_std = mean_std_ms(
      rows, "parent_resolve_max_cycles", clock_freq_hz)
  d2h_mean, d2h_std = mean_std_ms(rows, "d2h_span_cycles", clock_freq_hz)

  round_xs = np.arange(profiled_rounds)
  transfer_labels = ["h2d_seed", "resolve", "d2h"]
  transfer_xs = np.arange(len(transfer_labels))

  # h2d_matrix (a one-time, whole-matrix upload) gets its own panel/scale,
  # rightmost, not a 4th bar in ax_transfer -- it's 3-4 orders of magnitude
  # bigger than h2d_seed/resolve/d2h and would crush those three on a
  # shared linear scale.
  round_w_in = max(0.9 * profiled_rounds, 2.6)
  transfer_w_in = 1.3 * len(transfer_labels)
  h2d_matrix_w_in = 1.3
  fig, (ax_rounds, ax_transfer, ax_h2d_matrix) = plt.subplots(
      1, 3, figsize=(round_w_in + transfer_w_in + h2d_matrix_w_in + 1.5, 6.5),
      gridspec_kw={"width_ratios": [round_w_in, transfer_w_in, h2d_matrix_w_in],
                   "wspace": 0.25})

  bar_width = 0.62
  candidate_labels = []  # (ax, txt, xpos, seg_bottom, seg_top, w)

  def add_solo_bar(ax, xpos, height, color, label, yerr=0.0):
    ax.bar([xpos], [height], width=bar_width, color=color, edgecolor=SURFACE,
           linewidth=2, label=label, zorder=2)
    if yerr:
      ax.errorbar([xpos], [height], yerr=yerr, fmt="none", ecolor=TEXT_PRIMARY,
                  elinewidth=1.4, capsize=4, capthick=1.4, zorder=5)
    txt = ax.text(xpos, height / 2, f"{height:.3f}", ha="center", va="center",
                  fontsize=7, color="white", fontweight="bold", zorder=4)
    candidate_labels.append((ax, txt, xpos, 0.0, height, bar_width))

  bottom = np.zeros(profiled_rounds)
  # "Compute"/"Communication" capitalized here only, since these feed
  # straight into the figure legend (this panel's x-ticks are "round
  # 0"/"round 1"/...); h2d_seed/resolve/d2h/h2d_matrix stay lowercase on
  # their own x-ticks.
  segments = [("Compute", compute, COMPUTE_COLOR), ("Communication", communication,
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

  # One error bar per round, centered on that round's total stack height.
  # `bottom` here is each round's total mean height.
  if np.any(round_total_std > 0):
    ax_rounds.errorbar(round_xs, bottom, yerr=round_total_std, fmt="none", ecolor=TEXT_PRIMARY,
                        elinewidth=1.4, capsize=4, capthick=1.4, zorder=5)

  add_solo_bar(ax_h2d_matrix, 0, h2d_matrix_mean, HOST_TO_DEVICE_COLOR,
               "h2d_matrix", yerr=h2d_matrix_std)
  add_solo_bar(ax_transfer, transfer_xs[0], h2d_seed_mean, HOST_TO_DEVICE_COLOR,
               "h2d_seed", yerr=h2d_seed_std)
  add_solo_bar(ax_transfer, transfer_xs[1], parent_resolve_mean, PARENT_RESOLVE_COLOR,
               "resolve", yerr=parent_resolve_std)
  add_solo_bar(ax_transfer, transfer_xs[2], d2h_mean, DEVICE_TO_HOST_COLOR, "d2h", yerr=d2h_std)

  round_max = max((bottom + round_total_std).max() if profiled_rounds else 0.0, 1e-9)
  transfer_max = max(h2d_seed_mean + h2d_seed_std, parent_resolve_mean + parent_resolve_std,
                      d2h_mean + d2h_std, 1e-9)
  h2d_matrix_max = max(h2d_matrix_mean + h2d_matrix_std, 1e-9)
  ax_rounds.set_ylim(0, round_max * 1.25)
  ax_transfer.set_ylim(0, transfer_max * 1.15)
  ax_h2d_matrix.set_ylim(0, h2d_matrix_max * 1.15)
  # h2d_matrix is tens of thousands of ms -- compact tick labels plus a
  # single "x10^4"-style offset text above the axis. That offset text
  # carries real information (misread it and every tick is off by 4
  # orders of magnitude), so it's upsized/bolded/darkened below instead of
  # inheriting the muted default tick style.
  ax_h2d_matrix.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
  offset_text = ax_h2d_matrix.yaxis.get_offset_text()
  offset_text.set_fontsize(11)
  offset_text.set_color(TEXT_PRIMARY)
  offset_text.set_fontweight("bold")

  # No direction (TD/BU) suffix -- these poster plots only ever show
  # top-down rounds, so the label would be redundant on every round.
  round_labels = [f"round {r}" for r in range(profiled_rounds)]

  ax_h2d_matrix.set_xticks([0])
  ax_h2d_matrix.set_xticklabels(["h2d_matrix"])
  ax_h2d_matrix.set_xlim(-0.8, 0.8)
  ax_rounds.set_xticks(round_xs)
  ax_rounds.set_xticklabels(round_labels)
  ax_rounds.set_xlim(-0.8, max(profiled_rounds - 0.2, 0.2))
  ax_transfer.set_xticks(transfer_xs)
  ax_transfer.set_xticklabels(transfer_labels)
  ax_transfer.set_xlim(-0.8, len(transfer_labels) - 0.2)

  # A label survives only if its rendered bounding box fits inside its
  # own segment's rectangle (same fit check as plot_bfs_timing.py).
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

  # No "ms" ylabel on ax_h2d_matrix -- ax_rounds already establishes the
  # unit for the whole figure.
  ax_rounds.set_ylabel("ms", labelpad=8, fontsize=AXIS_LABEL_FONTSIZE)

  fig.suptitle(f"Timing Split on {poster_title_input(matrix)} and {pe_grid} PE Grid",
               fontsize=SUPTITLE_FONTSIZE, y=SUPTITLE_Y)
  for ax in (ax_h2d_matrix, ax_rounds, ax_transfer):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=TICK_LABEL_FONTSIZE)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
  fig.patch.set_facecolor(SURFACE)

  # ax_rounds's own "Compute"/"Communication" handles, plus one proxy
  # swatch per transfer color (HOST_TO_DEVICE_COLOR covers two
  # differently-named bars, so it needs its own legend entry too, same as
  # the other two). Legend labels are capitalized (research-poster
  # convention); the bars' own x-tick labels stay lowercase.
  handles, labels = ax_rounds.get_legend_handles_labels()
  handles += [Patch(color=HOST_TO_DEVICE_COLOR), Patch(color=PARENT_RESOLVE_COLOR),
              Patch(color=DEVICE_TO_HOST_COLOR)]
  labels += ["Host to Device", "Parent Resolve", "Device to Host"]
  fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False,
             fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(0.5, 0.01), columnspacing=1.8,
             handletextpad=0.6, labelspacing=1.0)
  # rect's right=0.97 leaves padding so ax_h2d_matrix's bar/ticks don't
  # butt against the figure edge; top=TOP_MARGIN leaves air between the
  # panel titles and the suptitle above.
  plt.tight_layout(rect=[0, 0.08, 0.97, TOP_MARGIN])

  # tight_layout's own rect top is silently ignored for this figure (a
  # matplotlib "Axes ... not compatible with tight_layout" warning) --
  # every axes' top edge lands at 0.88 regardless of rect[3]. Override
  # each axes' top edge post-hoc instead, so ax_h2d_matrix's bold "x10^4"
  # offset text clears the panel title above without crowding it.
  new_axes_top = 0.85
  for ax in (ax_rounds, ax_transfer, ax_h2d_matrix):
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, new_axes_top - pos.y0])

  # Panel titles as fig.text at one shared, absolute figure-fraction y,
  # positioned after tight_layout (using each axes' final get_position())
  # instead of ax.set_title(): matplotlib's per-axes title placement
  # auto-adjusts to clear each axes' own decorations, which would render
  # the three titles at different heights otherwise.
  title_y = 0.90
  pos_rounds = ax_rounds.get_position()
  fig.text((pos_rounds.x0 + pos_rounds.x1) / 2, title_y, "Per-Round Device Time",
            ha="center", va="bottom", fontsize=PANEL_TITLE_FONTSIZE, color=TEXT_PRIMARY)
  # ax_transfer and ax_h2d_matrix share one title spanning both -- they're
  # the same "host-transfer cost" concept, split into two panels only for
  # the scale mismatch.
  pos_transfer = ax_transfer.get_position()
  pos_h2d_matrix = ax_h2d_matrix.get_position()
  fig.text((pos_transfer.x0 + pos_h2d_matrix.x1) / 2, title_y,
            "Parent Resolve + Host-Device Time",
            ha="center", va="bottom", fontsize=PANEL_TITLE_FONTSIZE, color=TEXT_PRIMARY)

  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  plt.savefig(out_path, dpi=200, bbox_inches="tight")
  print(f"saved poster timing plot to {out_path}")
  plt.close(fig)


def _plot_timing_row_poster_log(rows, out_path):
  """One panel, one log-scale y-axis, 6 solo bars in chronological,
  grouped order: h2d_matrix/h2d_seed ("Host to Device"),
  compute/communication ("Kernel" -- collapsed across every round into
  one total each), resolve/d2h ("Device to Host"). Vertical separator
  lines + group labels above each pair make the 3 phases explicit. See
  the module docstring for the linear-vs-log tradeoff.

  Every bar is a mean across all of `rows`, error bars included."""
  row = rows[-1]
  matrix = row["infile_mtx"]
  pe_grid = row["pe_grid"]
  clock_freq_hz = float(row.get("clock_freq_hz") or CLOCK_FREQ_HZ)

  # Summed across rounds per row first, then mean/std taken across rows,
  # collapsing the linear mode's round-panel stack to 2 numbers. A tiny
  # positive floor on the mean, not 0: log(0) is undefined, and either
  # total can be genuinely ~0.
  compute_stack, communication_stack = per_round_compute_communication(rows, clock_freq_hz)
  compute_row_totals = compute_stack.sum(axis=1)
  communication_row_totals = communication_stack.sum(axis=1)
  compute_total, compute_total_std = _mean_std(compute_row_totals, axis=0)
  communication_total, communication_total_std = _mean_std(communication_row_totals, axis=0)
  compute_total = max(float(compute_total), 1e-6)
  communication_total = max(float(communication_total), 1e-6)

  h2d_matrix_mean, h2d_matrix_std = mean_std_ms(rows, "h2d_matrix_span_cycles", clock_freq_hz)
  h2d_seed_mean, h2d_seed_std = mean_std_ms(rows, "h2d_seed_span_cycles", clock_freq_hz)
  parent_resolve_mean, parent_resolve_std = mean_std_ms(
      rows, "parent_resolve_max_cycles", clock_freq_hz)
  d2h_mean, d2h_std = mean_std_ms(rows, "d2h_span_cycles", clock_freq_hz)

  # Chronological, grouped order: the 3 phases a search actually goes
  # through -- Host to Device, Kernel, Device to Host. resolve is
  # on-device only, but it's the step that produces exactly what d2h then
  # reads off, so it sits in the "Device to Host" group by position, not
  # by color (it keeps its own distinct PARENT_RESOLVE_COLOR). See
  # group_spans below for the separators/labels.
  bars = [
      ("h2d_matrix", h2d_matrix_mean, HOST_TO_DEVICE_COLOR, h2d_matrix_std),
      ("h2d_seed", h2d_seed_mean, HOST_TO_DEVICE_COLOR, h2d_seed_std),
      ("compute", compute_total, COMPUTE_COLOR, compute_total_std),
      ("communication", communication_total, COMMUNICATION_COLOR, communication_total_std),
      ("resolve", parent_resolve_mean, PARENT_RESOLVE_COLOR, parent_resolve_std),
      ("d2h", d2h_mean, DEVICE_TO_HOST_COLOR, d2h_std),
  ]
  group_spans = [("Host to Device", 0, 1), ("Kernel", 2, 3), ("Device to Host", 4, 5)]

  fig, ax = plt.subplots(figsize=(1.5 * len(bars) + 1.5, 5.5))
  bar_width = 0.62
  xs = np.arange(len(bars))
  for x, (name, height, color, yerr) in zip(xs, bars):
    ax.bar([x], [height], width=bar_width, color=color, edgecolor=SURFACE, linewidth=2,
           label=name, zorder=2)
    if yerr:
      ax.errorbar([x], [height], yerr=yerr, fmt="none", ecolor=TEXT_PRIMARY,
                  elinewidth=1.4, capsize=4, capthick=1.4, zorder=5)
    # Label above the bar, not centered inside: bar heights span several
    # decades here, so a short bar has no room for inside text.
    ax.text(x, height * 1.2, f"{height:.3f}", ha="center", va="bottom", fontsize=8,
            color=TEXT_PRIMARY, fontweight="bold", zorder=4)

  # Vertical separators between the 3 phase groups -- full-height
  # regardless of the log-scale ylim, since axvline's y range is
  # axes-fraction by default, not data space.
  for sep_x in (1.5, 3.5):
    ax.axvline(sep_x, color=BASELINE, linewidth=1, zorder=1)

  ax.set_yscale("log")
  # Explicit ylim, not matplotlib's own log-scale auto-margin (too
  # generous across the ~6 decades this chart spans) -- a tight,
  # data-driven range: just below the smallest bar, above the tallest label.
  min_height = min(height for _, height, _, _ in bars)
  max_label_top = max(height * 1.2 for _, height, _, _ in bars)
  ax.set_ylim(min_height * 0.5, max_label_top * 1.3)
  ax.set_xticks(xs)
  ax.set_xticklabels([name for name, _, _, _ in bars])
  ax.set_xlim(-0.8, len(bars) - 0.2)
  ax.set_ylabel("ms (log scale)", labelpad=8, fontsize=AXIS_LABEL_FONTSIZE)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_color(BASELINE)
  ax.spines["bottom"].set_color(BASELINE)
  ax.tick_params(colors=TEXT_MUTED, labelsize=TICK_LABEL_FONTSIZE)
  ax.yaxis.grid(True, which="major", color=GRIDLINE, linewidth=1, zorder=0)
  ax.set_axisbelow(True)
  ax.set_facecolor(SURFACE)
  fig.patch.set_facecolor(SURFACE)

  # Group labels above their own span -- y in axes fraction
  # (get_xaxis_transform), so they stay above the bars regardless of the
  # log-scale y-range.
  group_transform = ax.get_xaxis_transform()
  for label, i0, i1 in group_spans:
    ax.text((i0 + i1) / 2, 1.06, label, ha="center", va="bottom",
             fontsize=PANEL_TITLE_FONTSIZE, color=TEXT_PRIMARY, transform=group_transform)

  fig.suptitle(f"Timing Split on {poster_title_input(matrix)} and {pe_grid} PE Grid "
               "(log scale)", fontsize=SUPTITLE_FONTSIZE, y=SUPTITLE_Y)
  # top=0.92, not TOP_MARGIN (0.88): a single axes plus a lighter
  # group-label band needs less top headroom than the linear mode's
  # separate fig-wide panel-title row.
  plt.tight_layout(rect=[0, 0.02, 1, 0.92])

  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  plt.savefig(out_path, dpi=200, bbox_inches="tight")
  print(f"saved poster timing plot to {out_path}")
  plt.close(fig)


def parse_poster_args():
  """This file's own --log-scale flag, plus every shared flag
  plot_bfs_timing.py's parse_args() already defines."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-scale", action="store_true",
                       help="single log-scale panel (6 solo bars) instead of the default "
                            "3-panel linear small-multiples layout")
  return parse_args(parser)


def main():
  args = parse_poster_args()

  csv_path = args.csv
  if csv_path is None:
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "results", "bfs_timing.csv")
  with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

  matched = select_rows(rows, args)
  last = matched[-1]
  results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results")
  out_path = args.out or default_out_path(
      results_dir, last["infile_mtx"], last["pe_grid"], last["source"], last["channels"],
      log_scale=args.log_scale)
  plot_timing_row_poster(matched, out_path, log_scale=args.log_scale)


if __name__ == "__main__":
  main()
