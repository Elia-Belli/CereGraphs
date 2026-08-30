""" BFS tree comparison plotting (scipy reference vs. on-device f_spmv_iter)
  for bool_diag_spmv -- shared by run_bfs.py. Full graph faint gray for
  context, tree edges bolded, any node where the device's visited set
  disagrees with scipy's drawn in red. See render_tree_comparison()'s own
  docstring for the full picture; this used to be plot_bfs_tree.py's
  standalone main(), now split into importable pieces.
"""

import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def build_digraph(A_csr):
  """A_csr is row=dest/col=source (bool_diag_spmv's convention: edge
  col->row is the real adjacency direction). Self-loops are dropped,
  they'd just clutter the picture and never contribute to a BFS tree."""
  G = nx.DiGraph()
  G.add_nodes_from(range(A_csr.shape[0]))
  coo = A_csr.tocoo()
  for r, c in zip(coo.row.tolist(), coo.col.tolist()):
    if r != c:
      G.add_edge(c, r)
  return G


def compute_radial_layout(G, source):
  """Concentric rings by BFS level (distance from `source`, computed
  independently of scipy_parent/device_parent via plain graph BFS), instead
  of a force-directed layout -- this directly visualizes the thing that
  actually matters for a BFS tree (hop count), and rings are the standard
  convention for it.

  Within a ring, nodes are angularly sorted by their CANONICAL BFS parent's
  angle (a parent computed fresh via nx.bfs_tree(), guaranteed to be exactly
  one level up), not scipy_parent/device_parent directly -- both are
  genuine one-hop-closer BFS parents (bool_pe.csl's compute() gates parent
  tracking on visited_bitmap so a row can only ever be assigned a parent
  during its own true discovery round -- see its module docstring), but
  nx.bfs_tree() is still used here as the canonical layout reference since
  it's independent of either implementation's own tie-break among multiple
  equally-valid one-hop predecessors. Using device_parent for layout would
  make that tie-break (not the actual BFS structure) drive the clustering
  this function is trying to do.

  Nodes with no path from `source` at all (not merely unvisited by one
  algorithm run -- genuinely unreachable in the graph) go on a final outer
  ring, evenly spaced, past the last real BFS level.
  """
  dist = nx.single_source_shortest_path_length(G, source)
  levels = {}
  for v, d in dist.items():
    levels.setdefault(d, []).append(v)
  max_level = max(levels) if levels else 0
  unreached = [v for v in G.nodes() if v not in dist]

  bfs_tree = nx.bfs_tree(G, source)
  layout_parent = {v: next(bfs_tree.predecessors(v)) for v in bfs_tree.nodes() if v != source}

  angle = {source: 0.0}
  pos = {source: (0.0, 0.0)}

  # EQUAL radius steps between levels -- rings should correspond 1:1 to hop
  # count, not be squeezed or stretched based on how populous each level
  # happens to be. Sizing the (single, shared) step from the largest level
  # means a small level right after a huge one still gets pushed out by a
  # full step, not swallowed by a step sized for its own tiny population
  # (which would put it imperceptibly close to the ring before it, relative
  # to that ring's already-large radius). Sparser levels just get unused
  # circumference; nothing overlaps in the crowded one.
  max_count = max((len(nodes) for nodes in levels.values()), default=0)
  step = max(1.5, max_count / 10.0)
  radius = {L: L * step for L in range(0, max_level + 1)}

  for L in range(1, max_level + 1):
    nodes_here = sorted(levels.get(L, []))
    if not nodes_here:
      continue
    if L == 1:
      # only parent is `source` itself -- no angle to inherit, just spread
      # evenly in node-id order for determinism.
      for i, v in enumerate(nodes_here):
        angle[v] = 2 * math.pi * i / len(nodes_here)
    else:
      groups = {}
      for v in nodes_here:
        groups.setdefault(layout_parent[v], []).append(v)
      ordered_parents = sorted(groups, key=lambda p: angle[p])
      total = len(nodes_here)
      start = 0.0
      for p in ordered_parents:
        children = sorted(groups[p])
        width = 2 * math.pi * len(children) / total
        for i, v in enumerate(children):
          angle[v] = start + width * (i + 0.5) / len(children)
        start += width
    for v in nodes_here:
      pos[v] = (radius[L] * math.cos(angle[v]), radius[L] * math.sin(angle[v]))

  if unreached:
    outer_r = radius[max_level] + max(1.5, len(unreached) / 10.0)
    for i, v in enumerate(sorted(unreached)):
      a = 2 * math.pi * i / len(unreached)
      pos[v] = (outer_r * math.cos(a), outer_r * math.sin(a))

  return pos


def plot_panel(ax, G, pos, parent, visited, source, mismatch, title, extra_label,
                scipy_diff=None):
  """extra_label: a pre-formatted trailing string for the title, e.g.
  "4 rounds" (device panel -- rounds_completed) or "3 levels" (scipy panel
  -- max BFS depth reached, which has no discrete-round concept of its
  own).

  scipy_diff: optional per-node boolean array (only meaningful when
  --show-parent-mismatch is passed) -- True where THIS side's parent choice
  differs from scipy's own breadth_first_order pick. Colored orange,
  distinct from red, since these are expected tie-break differences (see
  run_bfs.py's --show-parent-mismatch help), not bugs -- mismatch (a real
  disagreement: visited-set mismatch against scipy) still takes priority in
  the color/z-order if both happen to apply to the same node."""
  if scipy_diff is None:
    scipy_diff = np.zeros(len(visited), dtype=bool)

  nx.draw_networkx_edges(G, pos, ax=ax, edge_color="lightgray", arrows=True,
                          arrowsize=6, width=0.5, node_size=250)

  node_colors = []
  for v in G.nodes():
    if mismatch[v]:
      node_colors.append("red")
    elif v == source:
      node_colors.append("gold")
    elif scipy_diff[v]:
      node_colors.append("orange")
    elif visited[v]:
      node_colors.append("skyblue")
    else:
      node_colors.append("whitesmoke")
  nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, edgecolors="black",
                          linewidths=0.5, node_size=250)
  nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

  tree_edges = [(parent[v], v) for v in G.nodes() if v != source and parent[v] >= 0]

  def edge_color(v):
    if mismatch[v]:
      return "red"
    if scipy_diff[v]:
      return "orange"
    return "darkblue"

  tree_edge_colors = [edge_color(v) for (_, v) in tree_edges]
  nx.draw_networkx_edges(G, pos, ax=ax, edgelist=tree_edges, edge_color=tree_edge_colors,
                          width=2.0, arrows=True, arrowsize=10, node_size=250)

  # redraw the source node on top, explicitly -- in a dense cluster a
  # tightly-packed layout can otherwise draw a later node right over it,
  # hiding the gold marker entirely (matplotlib draws scatter points in
  # call order, so "drawn earlier" can mean "covered up").
  source_color = "red" if mismatch[source] else "gold"
  nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[source], node_color=source_color,
                          edgecolors="black", linewidths=1.2, node_size=320)

  ax.set_title(f"{title}\n{int(np.sum(visited))}/{len(visited)} visited, {extra_label}")
  ax.axis("off")


def invalid_parents(parent, visited_arr, A_csr, source):
  """Nodes whose recorded parent isn't a valid BFS predecessor -- visited,
  and a real edge in the ORIGINAL matrix (A_csr[v, u] != 0, per
  bool_diag_spmv's row=dest/col=source convention). This is
  sdk-hypersparse-spmv-bfs/run_bfs.py's own verify_bfs() definition of "valid", deliberately
  NOT an exact-parent match against scipy's breadth_first_order: scipy picks
  its own arbitrary valid predecessor when a node has several, using a
  different tie-break than device_parent's "lowest index", so exact
  agreement isn't expected -- only that whichever parent WE picked is
  actually a real, already-visited predecessor in the true graph."""
  bad = []
  for v in range(len(parent)):
    if v == source or not visited_arr[v]:
      continue
    u = parent[v]
    if u < 0 or not visited_arr[u] or A_csr[v, u] == 0:
      bad.append(v)
  return bad


def render_tree_comparison(A_csr, source, scipy_parent, scipy_visited, scipy_levels,
                            device_parent, device_visited, device_rounds_run,
                            mismatch, n_mismatch, scipy_ok, scipy_diff_device,
                            show_parent_mismatch, infile_mtx, np_cols, np_rows, out_path):
  """Build and save the two-panel (scipy reference | on-device) BFS tree
  comparison plot. All the actual comparison numbers (mismatch, scipy_ok,
  scipy_diff_device, ...) are computed by the caller (run_bfs.py) -- this
  function only draws them."""
  print("building the tree plot...")
  G = build_digraph(A_csr)
  pos = compute_radial_layout(G, source)

  fig, axes = plt.subplots(1, 2, figsize=(16, 9))
  plot_panel(axes[0], G, pos, scipy_parent, scipy_visited, source, mismatch,
             "scipy reference BFS tree", f"{scipy_levels} levels")
  plot_panel(axes[1], G, pos, device_parent, device_visited, source, mismatch,
             "Device (f_spmv_iter) BFS tree", f"{device_rounds_run} rounds",
             scipy_diff=scipy_diff_device if show_parent_mismatch else None)

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
  if show_parent_mismatch:
    legend_handles.append(
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange",
                   markeredgecolor="black", markersize=10, label="differs from scipy"))
  fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), frameon=False)

  status = "0 mismatches" if n_mismatch == 0 else f"{n_mismatch} MISMATCHES"
  status += ", scipy OK" if scipy_ok else ", scipy CHECK FAILED"
  fig.suptitle(f"BFS tree comparison -- {os.path.basename(infile_mtx)}, "
               f"{np_cols}x{np_rows} grid, source={source} -- {status}")
  plt.tight_layout(rect=[0, 0.05, 1, 0.95])

  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  plt.savefig(out_path, dpi=600)
  plt.close(fig)
  print(f"saved tree plot to {out_path}")
