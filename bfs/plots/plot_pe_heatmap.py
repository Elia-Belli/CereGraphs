""" Per-PE timing heatmap for bool_diag_spmv -- reads back a .npz saved by
  run_bfs.py's --dump-pe-timing (device_io.py-adjacent, but standalone: this
  never touches the device, only the saved (round, height, width) cycle
  grids from bfs_timing.save_pe_phase_cycles()).

  Every run gets its own subfolder under plots/heatmap/<matrix>_<grid>_
  src<N>/, and every phase SELECTION within that run gets its OWN further
  subfolder (`overview/`, `<phase1>_<phase2>/`, ...) -- a run directory
  that's been explored with several different --phase combinations would
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

  Which phases: default is bfs_timing.PHASES (local_compute, local_term_cond
  -- the only two phases this repo still tracks per-PE, per-round; every
  individual communication phase -- visited_bcast/vertical_bcast/reduce/the
  4-phase termination relay -- and the skew-adjustment machinery that used
  to decompose them into wait-vs-real-cost were removed as unreliable, see
  docs/GRAPH500_BENCHMARK.md), each with its OWN independent color scale
  (local_compute and local_term_cond differ by an order of magnitude -- a
  shared scale would wash out the smaller one). `--phase name1,name2,...`
  selects a specific subset instead -- and whenever more than one phase is
  explicitly selected this way, they share ONE color scale, since the point
  of picking a subset is almost always to compare their magnitudes directly.

  Sequential magnitude data over a 2D grid -- default colormap is magma
  (perceptually uniform, colorblind-safe, like viridis but a different
  look) rather than a rainbow map, per the dataviz skill's color-by-job
  rule; no separate hex-palette validation needed since these are
  matplotlib's own built-in, already-vetted colormaps, not a custom brand
  palette. Pass --cmap to use any other matplotlib colormap name.

  How to run (paths relative to bool_diag_spmv/)
     python plots/plot_pe_heatmap.py --npz plots/heatmap/rmat_s8_e4_8x8_src0/rmat_s8_e4_8x8_src0.npz
     python plots/plot_pe_heatmap.py --npz plots/heatmap/rmat_s8_e4_8x8_src0/rmat_s8_e4_8x8_src0.npz \\
        --phase local_compute
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
from bfs_timing import PHASES  # pylint: disable=wrong-import-position

DEFAULT_PHASES = [name for name, _, _ in PHASES]

# matrix-structure grids run_bfs.py --dump-pe-timing saves alongside the
# per-round phase cycles (see its own save_pe_phase_cycles() call) -- not
# timing data at all, but the same (height, width) shape, so sparsity.png
# can be laid out side by side with the timing heatmaps to check by eye
# for correlation (e.g. does local_compute's imbalance track local_nnz's).
STRUCTURAL_GRID_NAMES = ["local_nnz", "local_nnz_cols", "local_nnz_rows"]


def load_pe_phase_cycles(npz_path):
  """Read back one run's per-PE-per-round-per-phase cycle grids, its
  per-PE matrix-structure grids (if --dump-pe-timing's run saved any), and
  its small metadata scalars, as saved by bfs_timing.save_pe_phase_cycles().

  phase_cycles picks up EVERY (rounds, height, width) grid in the file --
  not just names literally in bfs_timing.PHASES -- so raw_round_start/
  raw_round_end (also saved by run_bfs.py's --dump-pe-timing, see its own
  comment) are plottable the same way via an explicit --phase selection,
  with no extra wiring needed here.

  Returns (phase_cycles, structural_grids, metadata)."""
  data = np.load(npz_path)
  structural_grids = {name: data[name] for name in STRUCTURAL_GRID_NAMES if name in data.files}
  phase_cycles = {}
  metadata = {}
  for k in data.files:
    if k in STRUCTURAL_GRID_NAMES:
      continue
    arr = data[k]
    if arr.ndim == 3:  # (rounds, height, width) -- a plottable per-PE grid
      phase_cycles[k] = arr
    else:
      metadata[k] = arr.item()
  return phase_cycles, structural_grids, metadata


def _mark_diagonal_and_root(ax, height, width):
  """Outline every diagonal cell (row==col -- special throughout the whole
  algorithm) and additionally star the (MID, MID) root cell (the
  termination relay's own aggregation point, see bool_pe.csl's MID
  comment) -- harmless to draw regardless of which phase is shown."""
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
  default whole-DEFAULT_PHASES overview, handled by the caller instead)
  gets (None, None) -- imshow's own per-panel autoscale."""
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
  caller). shared_scale (explicit --phase selection) lays every phase out
  in ONE ROW sharing ONE color scale and ONE colorbar column, so they're
  directly, visually comparable -- not a separate colorbar per panel. The
  default whole-algorithm overview (shared_scale=False) keeps the
  multi-row grid with each phase's own independent scale/colorbar, since
  local_compute and local_term_cond differ by an order of magnitude and a
  shared bar would wash the smaller one out."""
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


def plot_one_round(phase_cycles, metadata, round_idx, phases, shared_scale, cmap, out_path):
  """One file for a single round, all `phases` side by side. The round
  number is in the figure's own suptitle only (not repeated per panel)."""
  grids_by_name = {p: phase_cycles[p][round_idx] for p in phases}
  suptitle = f"round {round_idx} -- per-PE cycles -- {_run_title(metadata)}"
  _plot_phase_grids(grids_by_name, phases, shared_scale, cmap, suptitle, out_path)


def plot_summary_avg(phase_cycles, metadata, phases, shared_scale, cmap, out_path):
  """mean-over-rounds overview -- typical cost per PE, not one worst round."""
  num_rounds = next(iter(phase_cycles.values())).shape[0]
  grids_by_name = {p: phase_cycles[p].mean(axis=0) for p in phases}
  suptitle = f"average over {num_rounds} rounds, per-PE cycles -- {_run_title(metadata)}"
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
                       help="comma-separated phase name(s) (see bfs_timing.PHASES: local_compute, "
                            "local_term_cond). Omit to show both, each with its own scale. 2+ "
                            "phases here share one color scale for direct comparison")
  parser.add_argument("--out-dir", default=None,
                       help="run directory (default: wherever --npz's file lives) -- this "
                            "selection's own PNGs land one level deeper, in a subfolder named "
                            "for the phase(s) selected (overview/, <phase>/, ...); sparsity.png "
                            "stays directly in this directory")
  parser.add_argument("--cmap", default="magma",
                       help="any matplotlib colormap name (default: magma -- perceptually "
                            "uniform, colorblind-safe, like viridis but a different look)")
  return parser.parse_args()


def main():
  args = parse_args()
  phase_cycles, structural_grids, metadata = load_pe_phase_cycles(args.npz)

  if args.phase:
    phases = args.phase.split(",")
  else:
    phases = [p for p in DEFAULT_PHASES if p in phase_cycles]
  for p in phases:
    assert p in phase_cycles, f"phase {p!r} not found in {args.npz} -- available: {list(phase_cycles)}"
  shared_scale = bool(args.phase)

  run_dir = args.out_dir or default_run_dir(args.npz)
  # one subfolder per phase selection (not a flat filename prefix) --
  # otherwise a run directory that's been explored with several different
  # --phase combinations accumulates dozens of same-named-pattern files at
  # its top level with no grouping. sparsity.png is the one exception: it's
  # not phase-specific (the matrix's own fixed partition structure), so it
  # stays directly in run_dir, not nested.
  group_name = "_".join(phases) if args.phase else "overview"
  out_dir = os.path.join(run_dir, group_name)

  num_rounds = next(iter(phase_cycles.values())).shape[0]
  for r in range(num_rounds):
    plot_one_round(phase_cycles, metadata, r, phases, shared_scale, args.cmap,
                    os.path.join(out_dir, f"round_{r}.png"))

  plot_summary_avg(phase_cycles, metadata, phases, shared_scale, args.cmap,
                    os.path.join(out_dir, "summary_avg.png"))

  if structural_grids:
    plot_sparsity(structural_grids, metadata, args.cmap, os.path.join(run_dir, "sparsity.png"))


if __name__ == "__main__":
  main()
