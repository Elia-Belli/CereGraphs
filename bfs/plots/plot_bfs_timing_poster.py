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

  Reuses plot_bfs_timing.py's CLI row-selection (select_rows -- every
  matching row, not just the most recent one) and h2d_seed/resolve base
  colors so this stays behaviorally consistent with the original -- only
  the layout and fused/recolored round segments are new.

  Every bar in every panel -- including the per-round compute/communication
  stacks, which used to be drawn from a single representative run on the
  theory that on-device work showed no comparable run-to-run variance --
  shows mean +/- std across every matching row instead of a single sample,
  so no panel looks methodologically different from any other. The
  host-device/parent_resolve/h2d_matrix bars are where real hardware jitter
  actually shows up as a visibly nonzero error bar (see
  docs/GRAPH500_BENCHMARK.md section 15); the round panel's error bars are
  expected to come out small, not absent. The mean +/- std methodology
  itself isn't captioned on the figure -- it's explained externally,
  wherever this poster gets used.

  --log-scale: an alternative to the whole small-multiples approach above
  -- ONE panel, ONE log-scale y-axis, 6 solo bars (compute + communication,
  each collapsed across every round into one total -- no per-round
  breakdown in this mode -- plus h2d_matrix/h2d_seed/resolve/d2h). Log
  scale is the more common convention for "one metric spans many orders of
  magnitude" in HPC/profiling papers specifically, at the known cost that
  bar *length* stops encoding magnitude the intuitive (linear) way once
  the axis is log-scaled. Both modes are kept side by side (not one
  replacing the other) so they're easy to compare directly -- see
  _plot_timing_row_poster_linear/_log's own docstrings.

  SVG only, no PNG -- this is vector output meant to be embedded/rescaled
  into a poster, not viewed as a standalone raster image.

  How to run (from bool_diag_spmv/), aggregating every run of one config:
     python3 plots/plot_bfs_timing_poster.py --csv results/hw/bfs_timing.csv \\
         --infile_mtx rmat_s17_e16.balanced750x750.mtx --pe_grid 750x750
     python3 plots/plot_bfs_timing_poster.py --csv results/hw/bfs_timing.csv \\
         --infile_mtx rmat_s17_e16.balanced750x750.mtx --pe_grid 750x750 --log-scale
  Or a single specific row, same as before (no --infile_mtx/--pe_grid):
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bfs_timing import CLOCK_FREQ_HZ  # pylint: disable=wrong-import-position
from plot_bfs_timing import (  # pylint: disable=wrong-import-position
    BASELINE, GRIDLINE, H2D_BASE_HEX, PARENT_RESOLVE_COLOR, SURFACE, TEXT_MUTED, TEXT_PRIMARY,
    parse_args, parse_cycle_list, select_rows,
)

# Matches plot_grid_scale_heatmap.py's AQUA (compute-bound pole) and ORANGE
# (communication-bound pole) exactly, so this plot and that heatmap share
# one visual language for "compute" vs "communication".
COMPUTE_COLOR = "#1baf7a"  # aqua
COMMUNICATION_COLOR = "#eb6834"  # orange
# h2d_seed/h2d_matrix collapse to ONE color (instead of plot_bfs_timing.py's
# own H2D_COLORS, 2 magenta tints) -- both already name themselves on their
# own x-tick, so color here only needs to say "Host to Device", not
# distinguish the two individually. resolve keeps plot_bfs_timing.py's own
# PARENT_RESOLVE_COLOR blue and its own "Parent Resolve" legend entry rather
# than folding into d2h's color/group: it's on-device work, not an actual
# host transfer (see the log-scale grouping comment below for why it still
# gets grouped NEXT TO d2h positionally even though its color now says
# otherwise). Every one of these 3 colors gets its own legend swatch -- see
# the legend-assembly comment below.
HOST_TO_DEVICE_COLOR = H2D_BASE_HEX  # magenta -- h2d_seed + h2d_matrix
DEVICE_TO_HOST_COLOR = "#eda100"  # yellow -- d2h only

# Same names/values as plot_grid_scale_heatmap.py/plot_balance_before_after.py
# -- the "hw/heatmap", "timing poster", and "balancing" figure families are
# meant to read as one visual system, not three scripts each with their own
# ad hoc sizing. SUPTITLE_Y (0.98, matplotlib's own suptitle default) and
# TOP_MARGIN (0.88) match those two files' tight_layout rect top too; this
# file's own further axes-top override to 0.85 below is this figure's own
# documented tight_layout-incompatibility workaround, not a divergent
# convention.
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
  """'Scale <scale>' for an RMAT input (the poster's own input family); falls
  back to the bare matrix stem for anything else (e.g. a SNAP graph) so this
  script doesn't break on non-RMAT rows."""
  m = RMAT_SCALE_RE.match(matrix)
  return f"Scale {m.group(1)}" if m else os.path.splitext(matrix)[0]


def default_out_path(plots_dir, matrix, pe_grid, source, channels, log_scale=False):
  matrix_stem = os.path.splitext(matrix)[0]
  # NOT "timings" -- every poster actually gets saved under plots/hw/
  # timing-poster/ (see the tracked files themselves); this default just
  # never matched that until now, so every real invocation so far has
  # passed --out explicitly instead of relying on it.
  timing_dir = os.path.join(plots_dir, "hw", "timing-poster")
  suffix = "_logscale" if log_scale else ""
  return os.path.join(
      timing_dir, f"timing_poster_{matrix_stem}_{pe_grid}_src{source}_ch{channels}{suffix}.svg")


def mean_std_ms(rows, cycles_key, clock_freq_hz):
  """Mean + sample std dev (ddof=1, undefined for a single row -> 0.0) of an
  int cycle column across every row in `rows`, converted to ms. Used for the
  host-device/parent_resolve bars, which real hardware showed have genuine
  run-to-run variance (see docs/GRAPH500_BENCHMARK.md section 15) -- the
  point of running the same input multiple times and plotting error bars,
  not just a single sample."""
  values = cycles_to_ms(np.array([int(r[cycles_key]) for r in rows], dtype=np.float64),
                         clock_freq_hz)
  return float(values.mean()), (float(values.std(ddof=1)) if len(values) > 1 else 0.0)


def _mean_std(stack, axis):
  """Sample mean + std dev (ddof=1, 0.0 when there's only 1 entry along
  `axis`) along `axis` of an ndarray -- mean_std_ms's same one-row-std-is-
  undefined edge case, generalized from a flat per-row array to an ndarray
  (per_round_compute_communication's per-row-per-round stacks)."""
  mean = stack.mean(axis=axis)
  std = stack.std(axis=axis, ddof=1) if stack.shape[axis] > 1 else np.zeros_like(mean)
  return mean, std


def per_round_compute_communication(rows, clock_freq_hz):
  """Per-row, per-round compute/communication arrays (ms), each shape
  (n_rows, n_rounds) -- the shared derivation (local_compute+local_term_cond,
  clip(round_duration - that, 0, None)) both poster modes' round/compute/
  communication bars build their own mean/std from: per-round for
  _plot_timing_row_poster_linear's round panel, summed-across-rounds-per-row
  for _plot_timing_row_poster_log's single compute/communication totals.
  Every row must report the same round count for this column (true for
  every repeated run of the same (infile_mtx, pe_grid) config seen so far);
  asserts rather than silently truncating/broadcasting if that ever stops
  holding."""
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
  """Dispatches to one of two entirely different renderings of the same
  underlying data -- see _plot_timing_row_poster_linear/_log's own
  docstrings for what each one shows and why. `rows`: every CSV row for
  the SAME (infile_mtx, pe_grid) config (one run each) -- see
  select_rows."""
  if log_scale:
    _plot_timing_row_poster_log(rows, out_path)
  else:
    _plot_timing_row_poster_linear(rows, out_path)


def _plot_timing_row_poster_linear(rows, out_path):
  """Small multiples: 3 separate linear-scale panels (rounds, h2d_seed/
  resolve/d2h, h2d_matrix), one per order of magnitude -- see the module
  docstring for why. Every bar in every panel is a mean across ALL of `rows`
  now, error bars included -- the round panel used to be drawn from a
  single representative run on the theory that on-device work showed no
  real run-to-run variance, but that's a reason the error bars should come
  out tiny, not a reason to skip plotting them and be the one panel that
  looks inconsistent with the other two."""
  row = rows[-1]
  matrix = row["infile_mtx"]
  pe_grid = row["pe_grid"]
  clock_freq_hz = float(row.get("clock_freq_hz") or CLOCK_FREQ_HZ)

  # fused: local_compute + local_term_cond, one "compute" segment (see
  # module docstring). compute/communication are per-round MEANS across
  # `rows`; round_total_std (mean+std centered on the same total each bar's
  # stack already sums to, same convention as every other bar in this
  # figure) is the error bar plotted on top of each round's stack below.
  compute_stack, communication_stack = per_round_compute_communication(rows, clock_freq_hz)
  compute, _ = _mean_std(compute_stack, axis=0)
  communication, _ = _mean_std(communication_stack, axis=0)
  _, round_total_std = _mean_std(compute_stack + communication_stack, axis=0)
  # profiled_rounds (not rounds_completed) is authoritative -- see
  # plot_bfs_timing.py's own comment on this same truncation subtlety.
  profiled_rounds = len(compute)

  # sync-corrected cross-PE span (see bfs_timing.read_sync_corrected_span) --
  # the only h2d_matrix/h2d_seed/d2h timing this project records now.
  # parent_resolve is on-device only (no host transfer, no sync bracket),
  # so it keeps its original per-PE max. Mean +/- std across all of `rows`.
  h2d_matrix_mean, h2d_matrix_std = mean_std_ms(rows, "h2d_matrix_span_cycles", clock_freq_hz)
  h2d_seed_mean, h2d_seed_std = mean_std_ms(rows, "h2d_seed_span_cycles", clock_freq_hz)
  parent_resolve_mean, parent_resolve_std = mean_std_ms(
      rows, "parent_resolve_max_cycles", clock_freq_hz)
  d2h_mean, d2h_std = mean_std_ms(rows, "d2h_span_cycles", clock_freq_hz)

  round_xs = np.arange(profiled_rounds)
  transfer_labels = ["h2d_seed", "resolve", "d2h"]
  transfer_xs = np.arange(len(transfer_labels))

  # h2d_matrix (a one-time, whole-matrix upload) gets its OWN panel/scale,
  # rightmost, not a 4th bar in ax_transfer -- it's 3-4 orders of magnitude
  # bigger than h2d_seed/resolve/d2h (seconds vs. low-single-digit ms),
  # which on a shared linear scale would crush those three to invisible
  # slivers, the exact "two measures of different scale" anti-pattern this
  # file's own docstring already avoids once (rounds vs. transfer panels).
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
  # "Compute"/"Communication" (capitalized) here ONLY -- these two labels
  # feed straight into the figure legend below (this panel's own x-ticks are
  # "round 0"/"round 1"/..., not these names), so they follow the legend's
  # own capitalization convention, not this file's lowercase
  # variable-name-as-bar-label convention (h2d_seed/resolve/d2h/h2d_matrix,
  # each still lowercase on ITS OWN x-tick, are unaffected).
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

  # One error bar per round, centered on that round's total (compute +
  # communication) stack height -- same elinewidth/capsize/capthick/ecolor
  # as add_solo_bar's own error bars below, for a consistent look across
  # every bar in the figure. `bottom` here is each round's total mean
  # height (the stack loop above just finished summing up to it).
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
  # h2d_matrix is tens of thousands of ms -- compact 1-5-digit tick labels
  # (e.g. "1", "2", "3") plus a single "x10^4"-style offset text above the
  # axis, instead of a full "20000"/"40000" on every tick. That offset text
  # carries real information (misread it and every tick label on this panel
  # is off by 4 orders of magnitude), so it can't inherit tick_params'
  # TEXT_MUTED color/default small size the way an ordinary tick label can --
  # explicitly upsized, bolded, and darkened below. useMathText=True renders
  # it as an actual "x10^4" superscript instead of the terser, easier-to-miss
  # default "1e4" plain text.
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
  # Same "-0.8, N - 0.2" margin convention as ax_rounds/ax_transfer below (N=1
  # category here) -- this panel previously hardcoded the literal "0.2" from
  # that pattern's own N=1 case instead of computing N - 0.2, so the right
  # edge sat at 0.2 while the bar itself (width 0.62, centered at x=0) already
  # extends out to 0.31: the bar's right side was clipped against the axis
  # limit. That clipping was also why the error bar cap looked off-center --
  # it's centered on the bar's true x=0, but the visibly-clipped bar reads
  # narrower on its right side, so the cap appears shifted right relative to
  # what's actually visible.
  ax_h2d_matrix.set_xlim(-0.8, 0.8)
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

  # No "ms" ylabel on ax_h2d_matrix -- ax_rounds (leftmost) already
  # establishes the unit for the whole figure; repeating it on the
  # rightmost panel is redundant.
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

  # ax_rounds's own "Compute"/"Communication" handles (the only bars whose
  # identity isn't already spelled out by their own x-tick label -- round
  # bars are ticked "round 0"/"round 1"/..., not "Compute"/"Communication"),
  # plus one proxy swatch per transfer color -- HOST_TO_DEVICE_COLOR covers
  # 2 differently-named bars (h2d_seed+h2d_matrix), so unlike
  # compute/communication the color alone no longer maps to one obvious
  # name; PARENT_RESOLVE_COLOR/DEVICE_TO_HOST_COLOR are each already
  # 1-bar-1-color again, but get their own swatch too for the same reason
  # h2d_matrix/h2d_seed/resolve/d2h lost theirs -- consistency: every color
  # actually used in this figure gets exactly one legend entry, no more, no
  # fewer. Every legend label capitalized -- research-poster convention (a
  # legend is read as prose, like an axis label, not as a literal
  # variable/field name) -- while the bars' OWN x-tick labels underneath
  # stay lowercase, matching this codebase's field-name convention for
  # those (h2d_seed/resolve/d2h/h2d_matrix/round N).
  handles, labels = ax_rounds.get_legend_handles_labels()
  handles += [Patch(color=HOST_TO_DEVICE_COLOR), Patch(color=PARENT_RESOLVE_COLOR),
              Patch(color=DEVICE_TO_HOST_COLOR)]
  labels += ["Host to Device", "Parent Resolve", "Device to Host"]
  fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False,
             fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(0.5, 0.01), columnspacing=1.8,
             handletextpad=0.6, labelspacing=1.0)
  # No mean/std-methodology caption on the figure itself -- explained
  # externally, wherever this poster gets used (see module docstring).
  # rect's right=0.97 (not 1.0) leaves a sliver of padding so ax_h2d_matrix's
  # bar/ticks don't butt right up against the figure's own right edge;
  # top=TOP_MARGIN (not 0.93) leaves clear air between the panel titles below
  # and the suptitle above (see title_y), instead of the two crowding
  # together -- same TOP_MARGIN value plot_grid_scale_heatmap.py/
  # plot_balance_before_after.py reserve for their own suptitle headroom.
  plt.tight_layout(rect=[0, 0.08, 0.97, TOP_MARGIN])

  # tight_layout's own rect top above is silently ignored for this specific
  # figure -- this file's own "Axes ... not compatible with tight_layout"
  # warning is exactly that incompatibility, and every axes' top edge lands
  # at 0.88 regardless of what rect[3] says (confirmed by printing
  # ax.get_position() before/after changing it). So the real headroom this
  # panel needs for ax_h2d_matrix's bold "x10^4" offset text (see its own
  # comment) to clear "...Host-Device Time" above without crowding it has to
  # come from directly overriding each axes' top edge post-hoc instead --
  # applied to all 3 (not just ax_h2d_matrix) so their plot areas stay
  # aligned. title_y below is unchanged, so the gap ABOVE the panel titles
  # (before the suptitle) is untouched too.
  new_axes_top = 0.85
  for ax in (ax_rounds, ax_transfer, ax_h2d_matrix):
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, new_axes_top - pos.y0])

  # Panel titles as fig.text at one shared, absolute figure-fraction y,
  # positioned AFTER tight_layout (using each axes' own now-final
  # get_position()) instead of ax.set_title(): matplotlib's per-axes title
  # placement auto-adjusts to clear each axes' own decorations (h2d_matrix's
  # scientific-notation offset text needs more clearance than the other two
  # panels have), so ax.set_title() alone renders the three titles at
  # different heights. One shared figure-space y sidesteps that entirely.
  title_y = 0.90
  pos_rounds = ax_rounds.get_position()
  fig.text((pos_rounds.x0 + pos_rounds.x1) / 2, title_y, "Per-Round Device Time",
            ha="center", va="bottom", fontsize=PANEL_TITLE_FONTSIZE, color=TEXT_PRIMARY)
  # ax_transfer and ax_h2d_matrix share ONE title spanning both, centered
  # over their combined width, instead of each getting its own -- they're
  # the same "host-transfer cost" concept split across two panels only for
  # the scale mismatch (see module docstring), not two different things.
  pos_transfer = ax_transfer.get_position()
  pos_h2d_matrix = ax_h2d_matrix.get_position()
  fig.text((pos_transfer.x0 + pos_h2d_matrix.x1) / 2, title_y,
            "Parent Resolve + Host-Device Time",
            ha="center", va="bottom", fontsize=PANEL_TITLE_FONTSIZE, color=TEXT_PRIMARY)

  # SVG only -- no PNG. Vector output for a poster figure that gets
  # embedded/rescaled, not viewed as a standalone raster image.
  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  plt.savefig(out_path, dpi=200, bbox_inches="tight")
  print(f"saved poster timing plot to {out_path}")
  plt.close(fig)


def _plot_timing_row_poster_log(rows, out_path):
  """One panel, one log-scale y-axis, 6 solo bars in chronological, grouped
  order (not alphabetical): h2d_matrix/h2d_seed ("Host to Device"),
  compute/communication ("Kernel" -- collapsed across every round into one
  total each, no per-round breakdown here), resolve/d2h ("Device to Host").
  Vertical separator lines + group labels above each pair make the 3
  phases explicit. The alternative to _plot_timing_row_poster_linear's
  small-multiples split: a single shared scale reads naturally to an
  audience used to log-scale benchmark charts, at the cost of bar *length*
  no longer encoding magnitude the intuitive (linear) way -- equal-looking
  gaps between bars are multiplicative, not additive. See the module
  docstring for the fuller tradeoff discussion.

  Every bar is a mean across ALL of `rows`, error bars included -- same
  convention as the linear mode's every panel now (see its own docstring)."""
  row = rows[-1]
  matrix = row["infile_mtx"]
  pe_grid = row["pe_grid"]
  clock_freq_hz = float(row.get("clock_freq_hz") or CLOCK_FREQ_HZ)

  # Summed across rounds PER ROW first, then mean/std taken across rows --
  # these totals stand in for the same whole-search "on-device time" the
  # linear mode's round panel shows as a stack, just collapsed to 2 numbers
  # (so summed across rounds), while still averaging across repeated runs of
  # this same config like every other bar here (so meaned across rows). A
  # tiny positive floor on the mean, not 0: log(0) is undefined, and
  # communication_total (or compute_total, for a single-round search) can be
  # genuinely ~0.
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

  # Chronological, grouped order (not alphabetical/panel-inherited): the
  # 3 phases a search actually goes through -- Host to Device (h2d_matrix,
  # h2d_seed), Kernel (compute, communication), Device to Host (resolve,
  # d2h -- resolve is on-device only, but it's the on-device step that
  # produces exactly what d2h then reads off, so it belongs with d2h's
  # phase POSITIONALLY, not the Kernel's). See group_spans below for the
  # separators/labels that make these 3 groups visually explicit -- same
  # HOST_TO_DEVICE_COLOR/PARENT_RESOLVE_COLOR/DEVICE_TO_HOST_COLOR colors as
  # _plot_timing_row_poster_linear now: h2d_matrix/h2d_seed share one color
  # (both already named on their own x-tick, color only needs to say which
  # phase), but resolve keeps its own distinct PARENT_RESOLVE_COLOR rather
  # than d2h's -- it sits in the "Device to Host" group by position/label,
  # not by color, since it isn't actually a host transfer.
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
    # Label ABOVE the bar (a multiplicative offset, not additive -- matches
    # log-scale semantics), not centered inside: bar heights span several
    # decades here, so a short bar (e.g. compute) has no room for inside
    # text the way a linear-scale bar does.
    ax.text(x, height * 1.2, f"{height:.3f}", ha="center", va="bottom", fontsize=8,
            color=TEXT_PRIMARY, fontweight="bold", zorder=4)

  # Vertical separators between the 3 phase groups (after h2d_seed, after
  # communication) -- full-height regardless of the log-scale ylim, since
  # axvline's y range is axes-fraction (0-1) by default, not data space.
  for sep_x in (1.5, 3.5):
    ax.axvline(sep_x, color=BASELINE, linewidth=1, zorder=1)

  ax.set_yscale("log")
  # Explicit ylim, not matplotlib's own log-scale auto-margin (which pads
  # generously in log space -- across the ~6 decades this chart spans,
  # that default margin leaves a lot of dead space above the tallest bar's
  # label). A tight, data-driven range instead: just below the smallest
  # bar, just above the tallest label.
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

  # Group labels above their own span -- x in data coords, y in axes
  # fraction (get_xaxis_transform), so they stay put above the bars
  # regardless of the log-scale y-range.
  group_transform = ax.get_xaxis_transform()
  for label, i0, i1 in group_spans:
    ax.text((i0 + i1) / 2, 1.06, label, ha="center", va="bottom",
             fontsize=PANEL_TITLE_FONTSIZE, color=TEXT_PRIMARY, transform=group_transform)

  fig.suptitle(f"Timing Split on {poster_title_input(matrix)} and {pe_grid} PE Grid "
               "(log scale)", fontsize=SUPTITLE_FONTSIZE, y=SUPTITLE_Y)
  # top=0.92, not TOP_MARGIN (0.88) -- this figure has a single axes plus a
  # lighter group-label band via axes-transform text (see group_transform
  # above), not the linear mode's separate fig-wide panel-title row, so it
  # needs less top headroom reserved.
  plt.tight_layout(rect=[0, 0.02, 1, 0.92])

  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  plt.savefig(out_path, dpi=200, bbox_inches="tight")
  print(f"saved poster timing plot to {out_path}")
  plt.close(fig)


def parse_poster_args():
  """This file's own --log-scale flag, plus every shared flag plot_bfs_timing.py's
  parse_args() already defines (--csv/--row/--infile_mtx/--pe_grid/--channels/--out)."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-scale", action="store_true",
                       help="single log-scale panel (compute/communication collapsed to one "
                            "total each, plus h2d_matrix/h2d_seed/resolve/d2h -- 6 solo bars "
                            "total) instead of the default 3-panel linear small-multiples layout")
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
  plots_dir = os.path.dirname(os.path.abspath(__file__))
  out_path = args.out or default_out_path(
      plots_dir, last["infile_mtx"], last["pe_grid"], last["source"], last["channels"],
      log_scale=args.log_scale)
  plot_timing_row_poster(matched, out_path, log_scale=args.log_scale)


if __name__ == "__main__":
  main()
