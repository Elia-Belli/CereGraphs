""" Per-PE timing heatmap for bool_diag_spmv -- reads back a .npz saved by
  run_bfs.py's --dump-pe-timing (device_io.py-adjacent, but standalone: this
  never touches the device, only the saved (round, height, width) cycle
  grids from bfs_timing.save_pe_phase_cycles()).

  Every run gets its own subfolder under plots/heatmap/<matrix>_<grid>_
  src<N>/, since aggregating rounds together hides real round-to-round
  variation -- which vertices are active in a given round is itself a
  function of the specific graph's structure and the chosen --source, not
  just the collective-communication protocol, so looking at one round at a
  time is what actually separates "protocol bias" from "this round's
  workload happened to land here":
    - `round_<r>.png`: one file per profiled round, all selected phases
      side by side -- the round number is in every title.
    - `summary_avg.png`: mean-over-rounds per phase, for a single at-a-
      glance overview of typical (not single-worst-round) cost across the
      whole run.
    - `sparsity.png`: the matrix's own per-PE partition structure
      (local_nnz/local_nnz_cols/local_nnz_rows, if run_bfs.py saved them --
      see bfs_timing.save_pe_phase_cycles()'s structural_grids), fixed for
      the whole run (no round axis) -- put next to the timing heatmaps to
      check by eye whether a phase's imbalance (e.g. local_compute) tracks
      the matrix's own sparsity distribution or comes from somewhere else.

  Which phases: default is bfs_timing.LEAF_PHASES, each with its OWN
  independent color scale (local_compute and local_term_cond differ by an
  order of magnitude -- a shared scale would wash out the smaller ones).
  `--phase name1,name2,...` or `--relay` (shorthand for the 4-phase
  termination relay, bfs_timing.RELAY_PHASES) instead select a specific
  subset -- and whenever more than one phase is explicitly selected this
  way, they share ONE color scale, since the point of picking a subset is
  almost always to compare their magnitudes directly.

  Sequential magnitude data over a 2D grid -- default colormap is magma
  (perceptually uniform, colorblind-safe, like viridis but a different
  look) rather than a rainbow map, per the dataviz skill's color-by-job
  rule; no separate hex-palette validation needed since these are
  matplotlib's own built-in, already-vetted colormaps, not a custom brand
  palette. Pass --cmap to use any other matplotlib colormap name.

  How to run
     python plot_pe_heatmap.py --npz pe_timing/rmat_s8_e4_8x8_src0.npz --relay
     python plot_pe_heatmap.py --npz pe_timing/rmat_s8_e4_8x8_src0.npz \\
        --phase relay_col_reduce,relay_row_reduce
     python plot_pe_heatmap.py --npz pe_timing/rmat_s8_e4_8x8_src0.npz
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from bfs_timing import LEAF_PHASES, PHASES, RELAY_PHASES

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
  Returns (phase_cycles, structural_grids, metadata)."""
  data = np.load(npz_path)
  all_phase_names = [name for name, _, _ in PHASES]
  phase_cycles = {name: data[name] for name in all_phase_names if name in data.files}
  structural_grids = {name: data[name] for name in STRUCTURAL_GRID_NAMES if name in data.files}
  non_metadata = set(all_phase_names) | set(STRUCTURAL_GRID_NAMES)
  metadata = {k: data[k].item() for k in data.files if k not in non_metadata}
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
                       help="comma-separated phase name(s) (see bfs_timing.PHASES). Omit to show "
                            "all phases (bfs_timing.LEAF_PHASES), each with its own scale. 2+ "
                            "phases here share one color scale for direct comparison")
  parser.add_argument("--relay", action="store_true",
                       help="shorthand for --phase=<the 4-phase termination relay's sub-phases> "
                            "(bfs_timing.RELAY_PHASES), shared scale")
  parser.add_argument("--out-dir", default=None,
                       help="output subfolder (default: wherever --npz's file lives)")
  parser.add_argument("--cmap", default="magma",
                       help="any matplotlib colormap name (default: magma -- perceptually "
                            "uniform, colorblind-safe, like viridis but a different look)")
  return parser.parse_args()


def main():
  args = parse_args()
  assert not (args.phase and args.relay), "--phase and --relay are mutually exclusive"
  phase_cycles, structural_grids, metadata = load_pe_phase_cycles(args.npz)

  if args.relay:
    phases = RELAY_PHASES
  elif args.phase:
    phases = args.phase.split(",")
  else:
    phases = [p for p in LEAF_PHASES if p in phase_cycles]
  for p in phases:
    assert p in phase_cycles, f"phase {p!r} not found in {args.npz} -- available: {list(phase_cycles)}"
  shared_scale = bool(args.relay or args.phase)

  out_dir = args.out_dir or default_run_dir(args.npz)
  if args.relay:
    prefix = "relay_"
  elif args.phase:
    prefix = "_".join(phases) + "_"
  else:
    prefix = ""

  num_rounds = next(iter(phase_cycles.values())).shape[0]
  for r in range(num_rounds):
    plot_one_round(phase_cycles, metadata, r, phases, shared_scale, args.cmap,
                    os.path.join(out_dir, f"{prefix}round_{r}.png"))

  plot_summary_avg(phase_cycles, metadata, phases, shared_scale, args.cmap,
                    os.path.join(out_dir, f"{prefix}summary_avg.png"))

  if structural_grids:
    plot_sparsity(structural_grids, metadata, args.cmap, os.path.join(out_dir, "sparsity.png"))


if __name__ == "__main__":
  main()
