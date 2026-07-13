""" Per-PE timing heatmap for bool_diag_spmv -- reads back a .npz saved by
  run_bfs.py's --dump-pe-timing (device_io.py-adjacent, but standalone: this
  never touches the device, only the saved (round, height, width) cycle
  grids from bfs_timing.save_pe_phase_cycles()).

  Every run gets its own subfolder under plots/heatmap/<matrix>_<grid>_
  src<N>/, and every phase/group SELECTION within that run gets its OWN
  further subfolder (`overview/`, `relay/`, `<phase1>_<phase2>/`,
  `skew_<phase1>_<phase2>/`, ...) -- a run directory that's been explored
  with several different --phase/--relay/--skew combinations would
  otherwise accumulate dozens of same-pattern files with no grouping.
  Aggregating rounds together (within one selection's subfolder) hides
  real round-to-round variation -- which vertices are active in a given
  round is itself a function of the specific graph's structure and the
  chosen --source, not just the collective-communication protocol, so
  looking at one round at a time is what actually separates "protocol
  bias" from "this round's workload happened to land here":
    - `<group>/round_<r>.png`: one file per profiled round, all selected
      phases side by side -- the round number is in every title.
    - `<group>/summary_avg.png`: mean-over-rounds per phase, for a single
      at-a-glance overview of typical (not single-worst-round) cost across
      the whole run.
    - `sparsity.png`: the matrix's own per-PE partition structure
      (local_nnz/local_nnz_cols/local_nnz_rows, if run_bfs.py saved them --
      see bfs_timing.save_pe_phase_cycles()'s structural_grids), fixed for
      the whole run (no round axis) and not phase-specific, so it stays at
      the run directory's own top level, not nested in any one selection's
      subfolder -- put next to the timing heatmaps to check by eye whether
      a phase's imbalance (e.g. local_compute) tracks the matrix's own
      sparsity distribution or comes from somewhere else.

  Which phases: default is bfs_timing.LEAF_PHASES, each with its OWN
  independent color scale (local_compute and local_term_cond differ by an
  order of magnitude -- a shared scale would wash out the smaller ones).
  `--phase name1,name2,...` or `--relay` (shorthand for the 4-phase
  termination relay, bfs_timing.RELAY_PHASES) instead select a specific
  subset -- and whenever more than one phase is explicitly selected this
  way, they share ONE color scale, since the point of picking a subset is
  almost always to compare their magnitudes directly.

  Skew-adjusted by default: PEs are never explicitly synchronized at a
  phase boundary, so a phase's raw duration can be almost entirely idle
  wait on a slower PE in the same row/column rather than real cost (see
  GRAPH500_BENCHMARK.md section 10) -- every plot above (default/--relay/
  --phase) therefore shows each phase's bfs_timing.compute_skew_adjusted
  `_adjusted` grid when the .npz has one (every phase except
  local_compute/local_term_cond, which are real local work with no group
  to adjust against), falling back to raw otherwise (e.g. an .npz saved
  before compute_skew_adjusted existed). Panel titles stay the plain phase
  name either way -- this is meant to be the number you look at by
  default, not a separate mode.

  `--skew PHASE[,PHASE...]` (any key(s) in bfs_timing.PHASE_GROUP_AXIS)
  shows [raw, wait, adjusted] per phase instead, all phases' triplets on
  one shared scale -- the breakdown behind the adjustment above, raw ==
  wait + adjusted always.

  Sequential magnitude data over a 2D grid -- default colormap is magma
  (perceptually uniform, colorblind-safe, like viridis but a different
  look) rather than a rainbow map, per the dataviz skill's color-by-job
  rule; no separate hex-palette validation needed since these are
  matplotlib's own built-in, already-vetted colormaps, not a custom brand
  palette. Pass --cmap to use any other matplotlib colormap name.

  How to run (paths relative to bool_diag_spmv/)
     python plots/plot_pe_heatmap.py --npz plots/heatmap/rmat_s8_e4_8x8_src0/rmat_s8_e4_8x8_src0.npz --relay
     python plots/plot_pe_heatmap.py --npz plots/heatmap/rmat_s8_e4_8x8_src0/rmat_s8_e4_8x8_src0.npz \\
        --phase relay_col_reduce,relay_row_reduce
     python plots/plot_pe_heatmap.py --npz plots/heatmap/rmat_s8_e4_8x8_src0/rmat_s8_e4_8x8_src0.npz \\
        --skew relay_col_bcast,relay_col_reduce
     python plots/plot_pe_heatmap.py --npz plots/heatmap/rmat_s8_e4_8x8_src0/rmat_s8_e4_8x8_src0.npz
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# bfs_timing.py lives one directory up (bool_diag_spmv/, this script's own
# parent) -- add it to sys.path so this import works whether this script is
# run standalone or imported by run_bfs.py (which already adds plots/ to
# its own sys.path, see its own top-of-file comment).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bfs_timing import LEAF_PHASES, PHASE_GROUP_AXIS, PHASES, RELAY_PHASES  # pylint: disable=wrong-import-position

# matrix-structure grids run_bfs.py --dump-pe-timing saves alongside the
# per-round phase cycles (see its own save_pe_phase_cycles() call) -- not
# timing data at all, but the same (height, width) shape, so sparsity.png
# can be laid out side by side with the timing heatmaps to check by eye
# for correlation (e.g. does local_compute's imbalance track local_nnz's).
STRUCTURAL_GRID_NAMES = ["local_nnz", "local_nnz_cols", "local_nnz_rows"]

# non-grid scalars save_pe_phase_cycles() may have folded into phase_cycles
# (bfs_timing.compute_skew_adjusted's own end-to-end sanity number) -- a
# (rounds,) array, not a (rounds, height, width) grid, so it can't be
# plotted as a heatmap panel and must be excluded from the generic grid
# pickup below.
_NON_GRID_KEYS = {"relay_critical_path_cycles"}


def load_pe_phase_cycles(npz_path):
  """Read back one run's per-PE-per-round-per-phase cycle grids, its
  per-PE matrix-structure grids (if --dump-pe-timing's run saved any), and
  its small metadata scalars, as saved by bfs_timing.save_pe_phase_cycles().

  phase_cycles picks up EVERY (rounds, height, width) grid in the file --
  not just names literally in bfs_timing.PHASES -- so compute_skew_adjusted's
  saved f"{phase}_wait"/f"{phase}_adjusted" grids are plottable the same way
  as the raw phases, with no extra wiring needed here when new derived
  grids are added on the run_bfs.py side.

  Returns (phase_cycles, structural_grids, metadata)."""
  data = np.load(npz_path)
  structural_grids = {name: data[name] for name in STRUCTURAL_GRID_NAMES if name in data.files}
  phase_cycles = {}
  metadata = {}
  for k in data.files:
    if k in STRUCTURAL_GRID_NAMES or k in _NON_GRID_KEYS:
      continue
    arr = data[k]
    if arr.ndim == 3:  # (rounds, height, width) -- a plottable per-PE grid
      phase_cycles[k] = arr
    else:
      metadata[k] = arr.item()
  return phase_cycles, structural_grids, metadata


def _mark_diagonal_and_root(ax, height, width):
  """Outline every diagonal cell (row==col -- special throughout the whole
  algorithm, not just the relay) and additionally star the (MID, MID) root
  cell (the relay's own aggregation point, see bool_pe.csl's MID comment) --
  harmless to draw for non-relay phases too, just not meaningful there."""
  P = min(height, width)
  for p in range(P):
    ax.add_patch(plt.Rectangle((p - 0.5, p - 0.5), 1, 1, fill=False,
                                edgecolor="white", linewidth=1.4))
  mid = P // 2
  ax.plot(mid, mid, marker="*", color="white", markersize=10,
          markeredgecolor="black", markeredgewidth=0.6)


def _run_title(metadata):
  return f"{metadata.get('infile_mtx')}, {metadata.get('pe_grid')} grid, source={metadata.get('source')}"


def _shared_scale(grids):
  """vmin/vmax shared across `grids` (a list of arrays) if there's more
  than one -- explicit multi-phase selection means "compare these
  directly", so they should share one color scale. A single grid (or the
  default whole-LEAF_PHASES overview, handled by the caller instead) gets
  (None, None) -- imshow's own per-panel autoscale."""
  if len(grids) <= 1:
    return None, None
  return int(min(g.min() for g in grids)), int(max(g.max() for g in grids))


def _plot_grid_panel(ax, grid, name, vmin, vmax, cmap):
  """Just the phase name as the title -- the round/aggregation context
  already lives in the figure's own suptitle, no need to repeat it per
  panel, and no worst-PE annotation cluttering it either (visible directly
  from the heatmap itself)."""
  im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
  _mark_diagonal_and_root(ax, grid.shape[0], grid.shape[1])
  ax.set_title(name, fontsize=9)
  ax.set_xticks([])
  ax.set_yticks([])
  return im


def _plot_phase_grids(phase_cycles_by_name, phases, shared_scale, cmap, suptitle, out_path,
                       cbar_label="cycles"):
  """Shared layout logic for plot_one_round/plot_summary_avg/plot_sparsity:
  phase_cycles_by_name maps phase -> the single (height, width) grid to
  plot for it (already reduced to one round or one aggregate by the
  caller). shared_scale (explicit --phase/--relay selection) lays every
  phase out in ONE ROW sharing ONE color scale and ONE colorbar column, so
  they're directly, visually comparable -- not a separate colorbar per
  panel. The default whole-algorithm overview (shared_scale=False) keeps
  the multi-row grid with each phase's own independent scale/colorbar,
  since e.g. local_compute and local_term_cond differ by an order of
  magnitude and a shared bar would wash the smaller ones out."""
  grids = [phase_cycles_by_name[p] for p in phases]

  if shared_scale:
    vmin, vmax = _shared_scale(grids)
    fig, axes = plt.subplots(1, len(phases), figsize=(3.2 * len(phases), 3.2), squeeze=False)
    im = None
    for i, ax in enumerate(axes[0]):
      im = _plot_grid_panel(ax, grids[i], phases[i], vmin, vmax, cmap)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label=cbar_label)
  else:
    ncols = min(len(phases), 3)
    nrows = math.ceil(len(phases) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows), squeeze=False)
    for i in range(nrows * ncols):
      ax = axes[i // ncols][i % ncols]
      if i >= len(phases):
        ax.axis("off")
        continue
      im = _plot_grid_panel(ax, grids[i], phases[i], None, None, cmap)
      fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)

  fig.suptitle(suptitle)
  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  plt.savefig(out_path, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"saved per-PE heatmap to {out_path}")


_SKEW_STATS = ["raw", "wait", "adjusted"]


def _plot_skew_grids(phase_cycles_by_key, skew_phases, cmap, suptitle, out_path,
                      cbar_label="cycles"):
  """--skew's own layout: one ROW per stat (raw, wait, adjusted), one COLUMN
  per phase -- scan a row to compare the same stat across phases (e.g. is
  relay_col_bcast's wait bigger than relay_col_reduce's), or a column to see
  one phase's own raw = wait + adjusted split. All panels share one scale
  (this is an explicit multi-phase comparison, same rule _shared_scale
  applies elsewhere). Phase name is the column header (row 0 only, not
  repeated per row); the stat name is the row label (column 0 only, via
  ylabel -- independent of the ticks _plot_grid_panel-style panels turn
  off)."""
  grids = [phase_cycles_by_key[p if stat == "raw" else f"{p}_{stat}"]
           for stat in _SKEW_STATS for p in skew_phases]
  vmin, vmax = _shared_scale(grids)

  ncols = len(skew_phases)
  fig, axes = plt.subplots(len(_SKEW_STATS), ncols, figsize=(3.0 * ncols, 2.8 * len(_SKEW_STATS)),
                            squeeze=False)
  im = None
  for row_idx, stat in enumerate(_SKEW_STATS):
    for col_idx, phase in enumerate(skew_phases):
      key = phase if stat == "raw" else f"{phase}_{stat}"
      grid = phase_cycles_by_key[key]
      ax = axes[row_idx][col_idx]
      im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
      _mark_diagonal_and_root(ax, grid.shape[0], grid.shape[1])
      ax.set_xticks([])
      ax.set_yticks([])
      if row_idx == 0:
        ax.set_title(phase, fontsize=9)
      if col_idx == 0:
        ax.set_ylabel(stat, fontsize=9)

  fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label=cbar_label)
  fig.suptitle(suptitle)
  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  plt.savefig(out_path, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"saved per-PE heatmap to {out_path}")


def _resolve_grid(phase_cycles, phase, use_adjusted):
  """Which array to actually plot for `phase`: its skew-adjusted grid
  (bfs_timing.compute_skew_adjusted's f"{phase}_adjusted", real fabric/
  compute cost with cross-PE wait excluded) when `use_adjusted` and that
  grid exists, else the raw grid -- raw is the only option for
  local_compute/local_term_cond (no group axis, see PHASE_GROUP_AXIS) and
  for any npz saved before compute_skew_adjusted existed."""
  if use_adjusted:
    adjusted_key = f"{phase}_adjusted"
    if adjusted_key in phase_cycles:
      return phase_cycles[adjusted_key]
  return phase_cycles[phase]


def plot_one_round(phase_cycles, metadata, round_idx, phases, shared_scale, cmap, out_path,
                    use_adjusted=True, skew_phases=None):
  """One file for a single round, all `phases` side by side. The round
  number is in the figure's own suptitle only (not repeated per panel).
  Titles stay the plain phase name regardless of use_adjusted -- see
  _resolve_grid(). skew_phases (set by --skew) switches to _plot_skew_grids'
  raw/wait/adjusted-per-row layout instead -- `phases` is still needed by
  the caller to know which keys to read from `phase_cycles`, but the
  row/column layout choice lives here."""
  if skew_phases:
    grids_by_key = {p: phase_cycles[p][round_idx] for p in phases}
    suptitle = f"round {round_idx} -- per-PE cycles -- {_run_title(metadata)}"
    _plot_skew_grids(grids_by_key, skew_phases, cmap, suptitle, out_path)
    return
  grids_by_name = {p: _resolve_grid(phase_cycles, p, use_adjusted)[round_idx] for p in phases}
  note = " (skew-adjusted where available)" if use_adjusted else ""
  suptitle = f"round {round_idx} -- per-PE cycles{note} -- {_run_title(metadata)}"
  _plot_phase_grids(grids_by_name, phases, shared_scale, cmap, suptitle, out_path)


def plot_summary_avg(phase_cycles, metadata, phases, shared_scale, cmap, out_path,
                      use_adjusted=True, skew_phases=None):
  """mean-over-rounds overview -- typical cost per PE, not one worst round."""
  num_rounds = next(iter(phase_cycles.values())).shape[0]
  if skew_phases:
    grids_by_key = {p: phase_cycles[p].mean(axis=0) for p in phases}
    suptitle = f"average over {num_rounds} rounds, per-PE cycles -- {_run_title(metadata)}"
    _plot_skew_grids(grids_by_key, skew_phases, cmap, suptitle, out_path)
    return
  grids_by_name = {p: _resolve_grid(phase_cycles, p, use_adjusted).mean(axis=0) for p in phases}
  note = " (skew-adjusted where available)" if use_adjusted else ""
  suptitle = f"average over {num_rounds} rounds, per-PE cycles{note} -- {_run_title(metadata)}"
  _plot_phase_grids(grids_by_name, phases, shared_scale, cmap, suptitle, out_path)


def plot_sparsity(structural_grids, metadata, cmap, out_path):
  """The matrix's own per-PE partition structure (local_nnz/_cols/_rows),
  own independent scale per panel (nnz can run much larger than distinct
  cols/rows touched) -- meant to sit next to round_<r>.png/summary_avg.png
  in the same folder for an eyeball correlation check against the timing
  heatmaps, not to imply these counts vary by round (they don't -- fixed
  once the matrix is partitioned)."""
  names = [n for n in STRUCTURAL_GRID_NAMES if n in structural_grids]
  suptitle = f"matrix partition structure (per-PE) -- {_run_title(metadata)}"
  _plot_phase_grids(structural_grids, names, False, cmap, suptitle, out_path, cbar_label="count")


def default_run_dir(npz_path):
  """The .npz's own directory -- run_bfs.py --dump-pe-timing now saves it
  inside plots/heatmap/<run_id>/ already, so PNGs land right next to the
  raw data as one self-contained per-run bundle, whatever the exact naming
  convention (or an explicit --pe-timing-out override) happened to be."""
  return os.path.dirname(os.path.abspath(npz_path))


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--npz", required=True, help="path to a run_bfs.py --dump-pe-timing .npz")
  parser.add_argument("--phase", default=None,
                       help="comma-separated phase name(s) (see bfs_timing.PHASES). Omit to show "
                            "all phases (bfs_timing.LEAF_PHASES), each with its own scale. 2+ "
                            "phases here share one color scale for direct comparison")
  parser.add_argument("--relay", action="store_true",
                       help="shorthand for --phase=<the 4-phase termination relay's sub-phases> "
                            "(bfs_timing.RELAY_PHASES), shared scale")
  parser.add_argument("--skew", default=None, metavar="PHASE[,PHASE...]",
                       help="decompose one or more comma-separated communication phases (any "
                            "key(s) in bfs_timing.PHASE_GROUP_AXIS) into [raw, wait, adjusted] "
                            "per phase, one shared scale across all of them -- raw = wait + "
                            "adjusted (see bfs_timing.compute_skew_adjusted). Requires the .npz "
                            "to have been saved by a run_bfs.py that computed the skew split; "
                            "mutually exclusive with --phase/--relay")
  parser.add_argument("--out-dir", default=None,
                       help="run directory (default: wherever --npz's file lives) -- this "
                            "selection's own PNGs land one level deeper, in a subfolder named "
                            "for the phase/group selected (overview/, relay/, skew_<phase>/, "
                            "...); sparsity.png stays directly in this directory")
  parser.add_argument("--cmap", default="magma",
                       help="any matplotlib colormap name (default: magma -- perceptually "
                            "uniform, colorblind-safe, like viridis but a different look)")
  return parser.parse_args()


def main():
  args = parse_args()
  assert sum(bool(x) for x in (args.phase, args.relay, args.skew)) <= 1, (
      "--phase, --relay and --skew are mutually exclusive")
  phase_cycles, structural_grids, metadata = load_pe_phase_cycles(args.npz)

  skew_phases = None
  if args.skew:
    skew_phases = args.skew.split(",")
    for sp in skew_phases:
      assert sp in PHASE_GROUP_AXIS, (
          f"--skew {sp!r} isn't a communication phase -- choices: {list(PHASE_GROUP_AXIS)}")
    phases = [name for sp in skew_phases for name in (sp, f"{sp}_wait", f"{sp}_adjusted")]
  elif args.relay:
    phases = RELAY_PHASES
  elif args.phase:
    phases = args.phase.split(",")
  else:
    phases = [p for p in LEAF_PHASES if p in phase_cycles]
  for p in phases:
    assert p in phase_cycles, (
        f"phase {p!r} not found in {args.npz} -- available: {list(phase_cycles)}"
        + (" (was this .npz saved before compute_skew_adjusted existed?)" if args.skew else ""))
  shared_scale = bool(args.relay or args.phase or args.skew)

  run_dir = args.out_dir or default_run_dir(args.npz)
  # one subfolder per phase/group selection (not a flat filename prefix) --
  # otherwise a run directory that's been explored with several different
  # --phase/--relay/--skew combinations accumulates dozens of same-named-
  # pattern files at its top level with no grouping. sparsity.png is the
  # one exception: it's not phase-specific (the matrix's own fixed
  # partition structure), so it stays directly in run_dir, not nested.
  if args.skew:
    group_name = f"skew_{'_'.join(skew_phases)}"
  elif args.relay:
    group_name = "relay"
  elif args.phase:
    group_name = "_".join(phases)
  else:
    group_name = "overview"
  out_dir = os.path.join(run_dir, group_name)

  # --skew's own panels are literally raw/wait/adjusted per phase --
  # substituting the adjusted grid in for its own "raw" panel would just
  # duplicate another panel, so use_adjusted only applies to the default/
  # --relay/--phase views, where raw would otherwise conflate real cost with
  # cross-PE wait (see GRAPH500_BENCHMARK.md section 10). skew_phases (only
  # set for --skew) switches plot_one_round/plot_summary_avg to the
  # raw/wait/adjusted-per-row layout instead of one-row-per-phase.
  use_adjusted = not args.skew
  num_rounds = next(iter(phase_cycles.values())).shape[0]
  for r in range(num_rounds):
    plot_one_round(phase_cycles, metadata, r, phases, shared_scale, args.cmap,
                    os.path.join(out_dir, f"round_{r}.png"), use_adjusted=use_adjusted,
                    skew_phases=skew_phases)

  plot_summary_avg(phase_cycles, metadata, phases, shared_scale, args.cmap,
                    os.path.join(out_dir, "summary_avg.png"), use_adjusted=use_adjusted,
                    skew_phases=skew_phases)

  if structural_grids:
    plot_sparsity(structural_grids, metadata, args.cmap, os.path.join(run_dir, "sparsity.png"))


if __name__ == "__main__":
  main()
