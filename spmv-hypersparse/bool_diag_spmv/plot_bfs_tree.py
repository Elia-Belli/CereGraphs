#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args
""" plot bool_diag_spmv's on-device BFS tree (f_spmv_iter) side-by-side with
  an independently-computed host-side reference, to check correctness
  visually rather than just via test_iterative.py's numeric mismatch counts.

  Unlike test_iterative.py (which seeds a random ~50%-density initial
  frontier to stress-test masking across many simultaneous discoveries at
  once), this script seeds a SINGLE source node, so both trees are an actual
  single BFS tree rather than a forest -- the standard, recognizable shape
  for this kind of picture. host_parent[source] and device_parent[source]
  are both set to `source` itself afterward (bfs_spmv/run_bfs.py's own
  convention for the root) since neither side ever assigns the source a
  "discovered via an edge" parent.

  Left panel:  host-driven reference -- sequential f_spmv launches, with
               update_parent_reference() (imported from test_iterative.py)
               tracking parent/visited state on the host exactly the way
               bool_pe.csl's compute() does it on-device.
  Right panel: on-device reference -- one f_spmv_iter launch; parent
               assembled from the full parent_local_buf rectangle via
               extract_parent_result() (also from test_iterative.py).

  Both panels draw the full graph in light gray for context, with the
  parent->child BFS tree edges bolded on top; any node/edge where the two
  sides disagree (parent choice or visited status) is drawn in red instead
  of the usual color, so a real bug would jump out visually.

  How to compile and run
     cs_python plot_bfs_tree.py --arch=wse3 --num_pe_cols=8 --num_pe_rows=8
        --channels=1 --driver=<path to cslc> --infile_mtx=<path to mtx file>
        --source=0 --out=bfs_tree.png
"""

import argparse
import math
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from preprocess_bool import preprocess
from run_bool import (csl_compile_core, dist_x_to_diag_hwl, extract_diag_result,
                       hwl_to_oned_colmajor, oned_to_hwl_colmajor)
from scipy.io import mmread
from test_iterative import extract_parent_result, update_parent_reference

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--infile_mtx", required=True, help="the sparse matrix in MTX format")
  parser.add_argument("--num_pe_cols", type=int, required=True, help="width of the core rectangle")
  parser.add_argument("--num_pe_rows", type=int, required=True, help="height of the core rectangle")
  parser.add_argument("--fabric-dims", help="Fabric dimension, i.e. <W>,<H>")
  parser.add_argument("--compile-only", action="store_true", help="Compile only")
  parser.add_argument("--run-only", action="store_true", help="Run only")
  parser.add_argument("--width-west-buf", default=0, type=int, help="width of west buffer")
  parser.add_argument("--width-east-buf", default=0, type=int, help="width of east buffer")
  parser.add_argument("--channels", default=1, type=int, help="number of I/O channels, 1-16")
  parser.add_argument("-d", "--driver", help="path to the CSL compiler")
  parser.add_argument("--cmaddr", help="CM address and port, i.e. <IP>:<port>")
  parser.add_argument("--arch", help="wse2 or wse3 (default wse2)")
  parser.add_argument("--latestlink", default="latest", help="folder for the compiled ELFs")
  parser.add_argument("--source", type=int, default=0, help="single BFS source vertex")
  parser.add_argument("--out", default="bfs_tree_comparison.png", help="output image path")
  return parser.parse_args()


def build_digraph(A_csr):
  """A_csr is row=dest/col=source (bool_diag_spmv's convention -- see
  generate_boolean_reference in run_bool.py): edge col->row is the real
  adjacency direction. Self-loops are dropped, they'd just clutter the
  picture and never contribute to a BFS tree."""
  G = nx.DiGraph()
  G.add_nodes_from(range(A_csr.shape[0]))
  coo = A_csr.tocoo()
  for r, c in zip(coo.row.tolist(), coo.col.tolist()):
    if r != c:
      G.add_edge(c, r)
  return G


def compute_layout(G, used_nodes):
  """Force-directed layout for `used_nodes` only (the actually-interesting
  set -- visited by either side), rescaled to fill [-1, 1] x [-1, 1] so it
  uses the whole panel regardless of how many other, never-visited nodes
  the full graph has. A plain spring_layout(G, ...) over ALL nodes wastes
  most of the canvas on those -- their mutual repulsion pushes them away
  from the interesting cluster, but does nothing to make that cluster
  itself bigger. Everything else (nodes nothing ever reaches) is placed on
  a ring around the outside, present for context but not competing for
  space."""
  used_nodes = list(used_nodes)
  other_nodes = [v for v in G.nodes() if v not in used_nodes]

  if len(used_nodes) > 1:
    sub = G.subgraph(used_nodes)
    pos = nx.spring_layout(sub, seed=42, k=2.5 / math.sqrt(len(used_nodes)), iterations=300)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pos = {
        v: (2 * (p[0] - xmin) / (xmax - xmin) - 1 if xmax > xmin else 0.0,
            2 * (p[1] - ymin) / (ymax - ymin) - 1 if ymax > ymin else 0.0)
        for v, p in pos.items()
    }
  else:
    pos = {v: (0.0, 0.0) for v in used_nodes}

  if other_nodes:
    radius = 1.5
    for i, v in enumerate(other_nodes):
      angle = 2 * math.pi * i / len(other_nodes)
      pos[v] = (radius * math.cos(angle), radius * math.sin(angle))

  return pos


def plot_panel(ax, G, pos, parent, visited, source, mismatch, title):
  nx.draw_networkx_edges(G, pos, ax=ax, edge_color="lightgray", arrows=True,
                          arrowsize=6, width=0.5, node_size=250)

  node_colors = []
  for v in G.nodes():
    if mismatch[v]:
      node_colors.append("red")
    elif v == source:
      node_colors.append("gold")
    elif visited[v]:
      node_colors.append("skyblue")
    else:
      node_colors.append("whitesmoke")
  nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, edgecolors="black",
                          linewidths=0.5, node_size=250)
  nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

  tree_edges = [(parent[v], v) for v in G.nodes() if v != source and parent[v] >= 0]
  tree_edge_colors = ["red" if mismatch[v] else "darkblue" for (_, v) in tree_edges]
  nx.draw_networkx_edges(G, pos, ax=ax, edgelist=tree_edges, edge_color=tree_edge_colors,
                          width=2.0, arrows=True, arrowsize=10, node_size=250)

  # redraw the source node on top, explicitly -- in a dense cluster a
  # tightly-packed layout can otherwise draw a later node right over it,
  # hiding the gold marker entirely (matplotlib draws scatter points in
  # call order, so "drawn earlier" can mean "covered up").
  source_color = "red" if mismatch[source] else "gold"
  nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[source], node_color=source_color,
                          edgecolors="black", linewidths=1.2, node_size=320)

  ax.set_title(f"{title}\n{int(np.sum(visited))}/{len(visited)} visited")
  ax.axis("off")


def main():
  """Main method to run the example code."""

  args = parse_args()

  cslc = "cslc"
  if args.driver is not None:
    cslc = args.driver

  width_west_buf = args.width_west_buf
  width_east_buf = args.width_east_buf
  channels = args.channels
  assert 1 <= channels <= 16, "number of I/O channels must be between 1 and 16"

  dirname = args.latestlink

  np_cols = args.num_pe_cols
  np_rows = args.num_pe_rows
  assert np_cols == np_rows, "diagonal-reduce design requires a square PE grid"
  P = np_cols
  width = np_cols
  height = np_rows

  infile_mtx = args.infile_mtx
  source = args.source
  print(f"infile_mtx = {infile_mtx}, source = {source}")

  A_coo = mmread(infile_mtx)
  A_csr = A_coo.tocsr(copy=True)
  A_csr = A_csr.sorted_indices()
  assert A_csr.has_sorted_indices == 1, "Error: A is not sorted"

  [nrows, ncols] = A_csr.shape
  assert nrows == ncols, "boolean diagonal-reduce SpMV requires a square matrix"
  n = nrows
  nnz = A_csr.nnz
  assert 0 <= source < n, f"--source={source} out of range [0, {n})"

  print(f"Load matrix A, {nrows}-by-{ncols} with {nnz} nonzeros (structural, boolean)")

  A_csc = A_csr.tocsc(copy=True)
  A_csc = A_csc.sorted_indices()
  assert A_csc.has_sorted_indices == 1, "Error: A is not sorted"

  matrix_info = preprocess(
      nrows, ncols, nnz, np_cols, np_rows,
      A_csr.indptr, A_csr.indices, A_csc.indptr, A_csc.indices,
  )

  max_local_nnz = matrix_info["max_local_nnz"]
  max_local_nnz_cols = matrix_info["max_local_nnz_cols"]
  max_local_nnz_rows = matrix_info["max_local_nnz_rows"]
  mat_rows_buf = matrix_info["mat_rows_buf"]
  mat_col_idx_buf = matrix_info["mat_col_idx_buf"]
  mat_col_loc_buf = matrix_info["mat_col_loc_buf"]
  mat_col_len_buf = matrix_info["mat_col_len_buf"]
  y_rows_init_buf = matrix_info["y_rows_init_buf"]
  local_nnz = matrix_info["local_nnz"]
  local_nnz_cols = matrix_info["local_nnz_cols"]
  local_nnz_rows = matrix_info["local_nnz_rows"]

  blk = math.ceil(n / P)

  # single-source seed, NOT test_iterative.py's random ~50% frontier -- see
  # the module docstring for why a single tree is the point here.
  x_bool0 = np.zeros(n, dtype=bool)
  x_bool0[source] = True
  x_hwl0 = dist_x_to_diag_hwl(n, x_bool0, blk, P)

  fabric_offset_x = 1
  fabric_offset_y = 1
  core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
  core_fabric_offset_y = fabric_offset_y
  min_fabric_width = core_fabric_offset_x + width + 2 + 1 + width_east_buf
  min_fabric_height = core_fabric_offset_y + height + 1

  fabric_width = 0
  fabric_height = 0
  if args.fabric_dims:
    w_str, h_str = args.fabric_dims.split(",")
    fabric_width = int(w_str)
    fabric_height = int(h_str)
  if fabric_width == 0 or fabric_height == 0:
    fabric_width = min_fabric_width
    fabric_height = min_fabric_height
  assert fabric_width >= min_fabric_width
  assert fabric_height >= min_fabric_height

  code_csl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "layout_bool.csl")

  start = time.time()
  csl_compile_core(
      cslc, code_csl, dirname, fabric_width, fabric_height,
      core_fabric_offset_x, core_fabric_offset_y, args.run_only, args.arch,
      np_cols, np_rows, blk, max_local_nnz, max_local_nnz_cols, max_local_nnz_rows,
      channels, width_west_buf, width_east_buf,
  )
  print(f"Compilation done in {time.time()-start}s", flush=True)

  if args.compile_only:
    print("COMPILE ONLY: EXIT")
    return

  runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

  sym_x_buf = runner.get_id("x_buf")
  sym_y_buf = runner.get_id("y_buf")
  sym_visited_buf = runner.get_id("visited_buf")
  sym_parent_local_buf = runner.get_id("parent_local_buf")
  sym_mat_rows_buf = runner.get_id("mat_rows_buf")
  sym_mat_col_idx_buf = runner.get_id("mat_col_idx_buf")
  sym_mat_col_loc_buf = runner.get_id("mat_col_loc_buf")
  sym_mat_col_len_buf = runner.get_id("mat_col_len_buf")
  sym_y_rows_init_buf = runner.get_id("y_rows_init_buf")
  sym_local_nnz = runner.get_id("local_nnz")
  sym_local_nnz_cols = runner.get_id("local_nnz_cols")
  sym_local_nnz_rows = runner.get_id("local_nnz_rows")

  runner.load()
  runner.run()

  mat_rows_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz, mat_rows_buf, np.uint32)
  runner.memcpy_h2d(sym_mat_rows_buf, mat_rows_buf_1d, 0, 0, width, height, max_local_nnz,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  mat_col_idx_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_cols, mat_col_idx_buf,
                                            np.uint32)
  runner.memcpy_h2d(sym_mat_col_idx_buf, mat_col_idx_buf_1d, 0, 0, width, height,
                     max_local_nnz_cols, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  mat_col_loc_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_cols, mat_col_loc_buf,
                                            np.uint32)
  runner.memcpy_h2d(sym_mat_col_loc_buf, mat_col_loc_buf_1d, 0, 0, width, height,
                     max_local_nnz_cols, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  mat_col_len_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_cols, mat_col_len_buf,
                                            np.uint32)
  runner.memcpy_h2d(sym_mat_col_len_buf, mat_col_len_buf_1d, 0, 0, width, height,
                     max_local_nnz_cols, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  y_rows_init_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_rows, y_rows_init_buf,
                                            np.uint32)
  runner.memcpy_h2d(sym_y_rows_init_buf, y_rows_init_buf_1d, 0, 0, width, height,
                     max_local_nnz_rows, streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  local_nnz_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz, np.uint32)
  runner.memcpy_h2d(sym_local_nnz, local_nnz_1d, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  local_nnz_cols_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz_cols, np.uint32)
  runner.memcpy_h2d(sym_local_nnz_cols, local_nnz_cols_1d, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)
  local_nnz_rows_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz_rows, np.uint32)
  runner.memcpy_h2d(sym_local_nnz_rows, local_nnz_rows_1d, 0, 0, width, height, 1,
                     streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                     order=MemcpyOrder.COL_MAJOR, nonblock=True)

  def seed_x(x_hwl):
    x_buf_1d = hwl_to_oned_colmajor(height, width, blk, x_hwl, np.float32)
    runner.memcpy_h2d(sym_x_buf, x_buf_1d, 0, 0, width, height, blk,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)

  def read_buf(sym):
    buf_1d = np.zeros(height * width * blk, np.float32)
    runner.memcpy_d2h(buf_1d, sym, 0, 0, width, height, blk,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    return oned_to_hwl_colmajor(height, width, blk, buf_1d, np.float32)

  def read_parent_local_buf():
    buf_1d = np.zeros(height * width * blk, np.uint32)
    runner.memcpy_d2h(buf_1d, sym_parent_local_buf, 0, 0, width, height, blk,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    return np.reshape(buf_1d, (height, width, blk), order="F")

  print("on-device: one f_spmv_iter launch")
  seed_x(x_hwl0)
  runner.launch("f_spmv_iter", nonblock=False)
  device_visited = extract_diag_result(n, blk, P, read_buf(sym_visited_buf))
  device_parent = extract_parent_result(n, blk, P, read_parent_local_buf())
  device_parent[source] = source  # root, not "undiscovered" -- see module docstring

  print("host-driven: sequential f_spmv launches")
  x_hwl = x_hwl0
  frontier_bool = x_bool0
  visited = x_bool0.copy()
  host_parent = np.full(n, -1, dtype=np.int64)
  for _ in range(n):
    update_parent_reference(host_parent, A_csr, frontier_bool)
    seed_x(x_hwl)
    runner.launch("f_spmv", nonblock=False)
    candidate = extract_diag_result(n, blk, P, read_buf(sym_y_buf))
    new_mask = candidate & ~visited
    visited |= new_mask
    if not new_mask.any():
      break
    frontier_bool = new_mask
    x_hwl = dist_x_to_diag_hwl(n, new_mask.astype(np.float32), blk, P)
  else:
    raise RuntimeError(f"host-driven baseline did not converge within {n} rounds -- "
                        "this should be impossible (bounded by node count); likely a bug")
  host_parent[source] = source

  runner.stop()

  mismatch = (host_parent != device_parent) | (visited != device_visited)
  n_mismatch = int(np.sum(mismatch))
  print(f"[[ mismatches (parent or visited): {n_mismatch} / {n} ]]")

  print("building the plot...")
  G = build_digraph(A_csr)
  # lay out nodes visited by EITHER side (the union, so a mismatch still
  # lands in the main cluster instead of getting flung to the unvisited
  # ring) to fill the whole panel -- see compute_layout()'s docstring.
  used_nodes = set(np.nonzero(visited | device_visited)[0].tolist()) | {source}
  pos = compute_layout(G, used_nodes)

  fig, axes = plt.subplots(1, 2, figsize=(16, 9))
  plot_panel(axes[0], G, pos, host_parent, visited, source, mismatch,
             "Host reference BFS tree")
  plot_panel(axes[1], G, pos, device_parent, device_visited, source, mismatch,
             "Device (f_spmv_iter) BFS tree")

  legend_handles = [
      plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gold",
                 markeredgecolor="black", markersize=10, label="source"),
      plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="skyblue",
                 markeredgecolor="black", markersize=10, label="visited"),
      plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="whitesmoke",
                 markeredgecolor="black", markersize=10, label="unvisited"),
      plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                 markeredgecolor="black", markersize=10, label="mismatch"),
      plt.Line2D([0], [0], color="lightgray", lw=1, label="graph edge"),
      plt.Line2D([0], [0], color="darkblue", lw=2, label="BFS tree edge"),
  ]
  fig.legend(handles=legend_handles, loc="lower center", ncol=6, frameon=False)

  status = "0 mismatches" if n_mismatch == 0 else f"{n_mismatch} MISMATCHES"
  fig.suptitle(f"BFS tree comparison -- {os.path.basename(infile_mtx)}, "
               f"{np_cols}x{np_rows} grid, source={source} -- {status}")
  plt.tight_layout(rect=[0, 0.05, 1, 0.95])
  plt.savefig(args.out, dpi=150)
  print(f"saved plot to {args.out}")


if __name__ == "__main__":
  main()
