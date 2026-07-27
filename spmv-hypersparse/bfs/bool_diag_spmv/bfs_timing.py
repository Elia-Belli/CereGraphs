""" Shared per-round timing constants/decoders for bool_diag_spmv's
  f_spmv_iter profiling (see src/bool_pe.csl's ts_buf/record_ts()/TS_*
  comment for the on-device side of this). Single source of truth for
  run_bfs.py (which writes bfs_timing.csv) and plot_bfs_timing.py (which
  reads it back) -- previously these lived only in bench_timing.py, with
  plot_bfs_timing.py keeping its own separate, hand-copied phase list that
  could silently drift out of sync.
"""

import os

import numpy as np

# must match bool_pe.csl's TS_* constants / NUM_TS_SLOTS exactly -- these are
# indices into the 10-slot-per-round raw timestamp capture, not phases
# themselves (phases are consecutive-slot differences, see PHASES below).
TS_VBCAST_ISSUE = 0
TS_VBCAST_DONE = 1
TS_COMPUTE_ENTRY = 2
TS_COMPUTE_RESET_DONE = 3
TS_COMPUTE_EXPAND_ENTRY = 4
TS_REDUCE_ISSUE = 5
TS_REDUCE_DONE = 6
TS_RELAY_ISSUE = 7
TS_TERM_COL_DONE = 8
TS_TERM_ROW_DONE = 9
TS_TERM_ROW_BCAST_DONE = 10
TS_TERM_COL_BCAST_DONE = 11
NUM_TS_SLOTS = 12

# (name, start slot, end slot) -- each phase is literally end-minus-start of
# two of the raw captures above; relay_total spans all four relay phases at
# once rather than being their sum, as a direct (not accumulated) check.
# local_compute_reset/local_compute_compact/local_compute_expand are
# local_compute's own three consecutive parts (split at
# TS_COMPUTE_RESET_DONE/TS_COMPUTE_EXPAND_ENTRY, see bool_pe.csl's
# compute()) -- all three purely diagnostic sub-phases, deliberately absent
# from SEARCH_TIME_PHASES below so they don't double-count local_compute's
# own contribution to device_time_cycles. local_compute_reset exists
# because its cost tracks blk (dense y_local_buf/y_buf zeroing), not local
# sparsity -- separating it out is what makes local_compute_compact an
# honest measure of the sparse multiply itself.
PHASES = [
    ("visited_bcast", TS_VBCAST_ISSUE, TS_VBCAST_DONE),
    ("vertical_bcast", TS_VBCAST_DONE, TS_COMPUTE_ENTRY),
    ("local_compute", TS_COMPUTE_ENTRY, TS_REDUCE_ISSUE),
    ("local_compute_reset", TS_COMPUTE_ENTRY, TS_COMPUTE_RESET_DONE),
    ("local_compute_compact", TS_COMPUTE_RESET_DONE, TS_COMPUTE_EXPAND_ENTRY),
    ("local_compute_expand", TS_COMPUTE_EXPAND_ENTRY, TS_REDUCE_ISSUE),
    ("reduce", TS_REDUCE_ISSUE, TS_REDUCE_DONE),
    ("local_term_cond", TS_REDUCE_DONE, TS_RELAY_ISSUE),
    ("relay_col_reduce", TS_RELAY_ISSUE, TS_TERM_COL_DONE),
    ("relay_row_reduce", TS_TERM_COL_DONE, TS_TERM_ROW_DONE),
    ("relay_row_bcast", TS_TERM_ROW_DONE, TS_TERM_ROW_BCAST_DONE),
    ("relay_col_bcast", TS_TERM_ROW_BCAST_DONE, TS_TERM_COL_BCAST_DONE),
    ("relay_total", TS_RELAY_ISSUE, TS_TERM_COL_BCAST_DONE),
]

# chronological order == stack order for plot_bfs_timing.py's bars, bottom
# to top -- PHASES minus relay_total (which would double-count its own 4
# relay_* sub-phases if also drawn as its own segment).
LEAF_PHASES = [name for name, _, _ in PHASES if name != "relay_total"]

# the 4-phase termination relay's own sub-phases (see bool_pe.csl's module
# docstring) -- broken out so callers (plot_pe_heatmap.py's --relay) can
# compare just these four on a shared cycle scale, distinct from
# LEAF_PHASES' full-algorithm overview (which intentionally keeps each
# phase's own independent scale, since e.g. local_compute and
# local_term_cond differ by an order of magnitude).
RELAY_PHASES = ["relay_col_reduce", "relay_row_reduce", "relay_row_bcast", "relay_col_bcast"]

# maps each TS_* slot index (see the constants above) to the exact name
# decode_pe_phase_cycles() saves its raw absolute grid under -- used by
# compute_skew_adjusted() below to look up a phase's start/end slot grids.
_SLOT_NAMES = {
    TS_VBCAST_ISSUE: "ts_vbcast_issue",
    TS_VBCAST_DONE: "ts_vbcast_done",
    TS_COMPUTE_ENTRY: "ts_compute_entry",
    TS_COMPUTE_RESET_DONE: "ts_compute_reset_done",
    TS_COMPUTE_EXPAND_ENTRY: "ts_compute_expand_entry",
    TS_REDUCE_ISSUE: "ts_reduce_issue",
    TS_REDUCE_DONE: "ts_reduce_done",
    TS_RELAY_ISSUE: "ts_relay_issue",
    TS_TERM_COL_DONE: "ts_term_col_done",
    TS_TERM_ROW_DONE: "ts_term_row_done",
    TS_TERM_ROW_BCAST_DONE: "ts_term_row_bcast_done",
    TS_TERM_COL_BCAST_DONE: "ts_term_col_bcast_done",
}

# Per-PE timestamps are NOT taken at synchronized phase boundaries -- a PE
# that has nothing real to contribute to a given collective (e.g. a
# non-diagonal PE before relay_col_reduce, or a non-MID row before
# relay_row_reduce/relay_row_bcast) issues its own call almost immediately,
# then simply blocks inside the collective waiting for whichever PE in its
# dependency group (one row for an mpi_x call, one column for an mpi_y call)
# actually has real work to finish first. That PE's own raw phase duration
# therefore conflates real fabric transit with wait time for the group's
# straggler -- this table says, for each phase that's a real collective
# call, which axis its dependency group spans, so compute_skew_adjusted()
# can split "wait" from "real work" per phase, per PE:
#   "row"    -- mpi_x call, group = the fixed row (all P columns)
#   "column" -- mpi_y call, group = the fixed column (all P rows)
# local_compute and local_term_cond are deliberately absent: neither
# brackets a collective call, and their own per-PE variance is genuine local
# work (data-dependent sparsity, or a diagonal-only masking loop), not idle
# wait -- subtracting a group reference there would remove real signal, not
# noise. See GRAPH500_BENCHMARK.md section 10 for the full reasoning.
PHASE_GROUP_AXIS = {
    "visited_bcast": "row",
    "vertical_bcast": "column",
    "reduce": "row",
    "relay_col_reduce": "column",
    "relay_row_reduce": "row",
    "relay_row_bcast": "row",
    "relay_col_bcast": "column",
}

# Root position for each phase's group -- verified against
# collectives_2d/pe.csl's transfer_data_reduce()/configure_broadcast_network():
# every reduce_fadds/broadcast call splits its group into two INDEPENDENT
# sequential sub-chains meeting at this root position, each flowing one hop
# at a time from its own far edge inward toward root (see
# compute_skew_adjusted's own docstring for why this matters).
#   "diagonal" -- root = this row's/column's own diagonal index (varies:
#                 row i's root is column i, column j's root is row j)
#   "mid"      -- root = the fixed MID index, same for every row/column
PHASE_ROOT_KIND = {
    "visited_bcast": "diagonal",
    "vertical_bcast": "diagonal",
    "reduce": "diagonal",
    # relay_col_reduce's root changed from "mid" to "diagonal" when its
    # underlying primitive changed from mpi_y.reduce_fadds(MID, ...) to
    # mpi_y.broadcast(pcol_id, ...) -- see bool_pe.csl's reduce_done() and
    # its own comment on why (only the diagonal PE ever has real data, so
    # this moves it instead of reducing it). Name kept for continuity with
    # existing CSV columns/heatmaps -- it's still "phase A" of the relay,
    # just a different primitive now.
    "relay_col_reduce": "diagonal",
    "relay_row_reduce": "mid",
    "relay_row_bcast": "mid",
    "relay_col_bcast": "mid",
}

# phases where only row/column MID actually participates for real (see
# bool_pe.csl's row-MID-only relay optimization, GRAPH500_BENCHMARK.md
# section 9) -- every other row's own copy of relay_row_reduce/
# relay_row_bcast is a zero-duration placeholder, not a real group to
# adjust; compute_skew_adjusted leaves those rows as a no-op (entry_ref =
# own start, wait = 0), same convention the raw capture already uses.
PHASE_MID_ONLY = {"relay_row_reduce", "relay_row_bcast"}

_PHASE_SLOTS = {name: (start, end) for name, start, end in PHASES}

# on-device phases that make up the Graph500-style "search time" (see
# GRAPH500_BENCHMARK.md section 3) -- relay_total is used directly instead
# of its own 4 relay_col/row_* sub-phases, to avoid double-counting.
SEARCH_TIME_PHASES = [
    "visited_bcast", "vertical_bcast", "local_compute", "reduce", "local_term_cond", "relay_total",
]

# the two separately-timed h2d brackets (see GRAPH500_BENCHMARK.md section
# 2/3) -- matrix structure (construction-like, one-time) vs seed x
# (per-search).
H2D_PARTS = ["h2d_matrix", "h2d_seed"]

# timestamp.tsc_size_words in bool_pe.csl -- the <time> library's fixed
# [3]u16 timestamp width (see SKILL-LIBRARIES.md's <time> entry).
TSC_WORDS = 3

# WSE clock frequency, for converting search_time_cycles -> seconds for
# GTEPS (GRAPH500_BENCHMARK.md section 5/6).
CLOCK_FREQ_HZ = 875e6


def decode_round_timestamps(ts_hwl_u32, height, width, max_rounds):
  """ts_hwl_u32: (height, width, max_rounds*NUM_TS_SLOTS*3) uint32 array
  fresh off memcpy_d2h (u16 values zero-extended into u32 words, COL_MAJOR
  wire order already resolved into this hwl shape -- same convention as
  device_io.py's other u16 buffers, e.g. extract_parent_result's input).
  The last axis is C-contiguous per-PE in exactly the order bool_pe.csl's
  record_ts() wrote it (round slowest, slot, then word fastest), so a plain
  reshape (no extra transpose) recovers (height, width, max_rounds,
  NUM_TS_SLOTS, 3).

  Returns a (height, width, max_rounds, NUM_TS_SLOTS) int64 array of
  absolute 48-bit timestamps, one per PE per round per capture slot --
  packed low-to-high exactly like cerebras.sdk.sdk_utils.make_u48
  (word0 | word1<<16 | word2<<32)."""
  ts_hwl = ts_hwl_u32.reshape(height, width, max_rounds, NUM_TS_SLOTS, 3)
  w0 = ts_hwl[..., 0].astype(np.int64)
  w1 = ts_hwl[..., 1].astype(np.int64)
  w2 = ts_hwl[..., 2].astype(np.int64)
  return w0 + (w1 << 16) + (w2 << 32)


def read_tic_toc_delta(runner, sym_tsc_start, sym_tsc_end, height, width):
  """Read back tsc_start_buffer/tsc_end_buffer (each PE's own f_tic()/
  f_toc() capture -- see f_enable_tsc()/f_tic()/f_toc() in bool_pe.csl) and
  return the per-PE elapsed-cycle deltas as a flat (height*width,) int64
  array. Same-PE subtraction (this PE's own toc minus this PE's own tic),
  so no cross-PE clock synchronization is needed -- see run_bfs.py's own
  module docstring for why."""
  # local import: only needed here, avoids importing the whole SDK runtime
  # module for callers that only want the pure-python constants above.
  from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
      MemcpyDataType, MemcpyOrder,
  )

  def _read(sym):
    buf_1d = np.zeros(height * width * TSC_WORDS, np.uint32)
    runner.memcpy_d2h(buf_1d, sym, 0, 0, width, height, TSC_WORDS,
                       streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                       order=MemcpyOrder.COL_MAJOR, nonblock=False)
    hwl = np.reshape(buf_1d, (height, width, TSC_WORDS), order="F")
    w0 = hwl[..., 0].astype(np.int64)
    w1 = hwl[..., 1].astype(np.int64)
    w2 = hwl[..., 2].astype(np.int64)
    return (w0 + (w1 << 16) + (w2 << 32)).reshape(-1)

  tic = _read(sym_tsc_start)
  toc = _read(sym_tsc_end)
  return toc - tic


def compute_round_summary(raw_slots, profiled_rounds):
  """Robust, non-decomposed round/runtime timing -- deliberately NOT trying
  to further split a round's communication phases into wait vs real cost
  (see compute_skew_adjusted's own docstring / GRAPH500_BENCHMARK.md
  sections 10-13 for why that turned out to be a hard, still-unresolved
  problem on its own -- put aside rather than fixed further here). Two
  straggler-PE spans instead, both justified as genuine synchronization
  points rather than assumptions about internal chain structure:

  - total_runtime_cycles: max over PEs of (last profiled round's
    TS_TERM_COL_BCAST_DONE - round 0's TS_VBCAST_ISSUE) -- the straggler
    PE's own first-phase-to-last-phase span, a solid estimate of total
    on-device BFS time. Doesn't account for the very first broadcast's own
    initial fabric fill delay -- acceptable, a one-time cost diluted across
    every round that follows, not a per-round recurring one.
  - round_duration_cycles: per round, max over PEs of (that SAME round's
    TS_TERM_COL_BCAST_DONE - TS_VBCAST_ISSUE) -- justified because the
    termination relay forces every PE to agree on nz_total before ANY PE
    can issue the next round's visited_bcast, so this genuinely is a hard
    synchronization boundary (unlike the relay's own internal sub-phases,
    where no such cross-PE guarantee holds).

  Returns (total_runtime_cycles: int, round_duration_cycles: (profiled_rounds,)
  int64 array)."""
  issue = raw_slots["ts_vbcast_issue"][:profiled_rounds]        # (rounds, height, width)
  done = raw_slots["ts_term_col_bcast_done"][:profiled_rounds]  # (rounds, height, width)
  round_duration_cycles = (done - issue).max(axis=(1, 2))
  total_runtime_cycles = int((done[-1] - issue[0]).max())
  return total_runtime_cycles, round_duration_cycles


def decode_phase_row(ts_hwl_u32, height, width, max_rounds, rounds_completed, verbose=True):
  """Decode one search's ts_buf into per-phase min/max/avg CSV columns
  (semicolon-joined per-round strings, same shape run_bfs.py has always
  logged), PLUS the skew-adjusted counterpart for every phase that has one
  (see compute_skew_adjusted -- everything except local_compute/
  local_term_cond, which are real local work, not wait). `device_time_cycles`
  (the on-device portion of GRAPH500_BENCHMARK.md section 3's
  search_time_cycles, h2d_seed added by the caller) is built from the
  ADJUSTED straggler-PE-max for SEARCH_TIME_PHASES where one exists --
  section 10/11 showed the raw per-PE max can be mostly cross-PE wait, not
  real cost, so summing raw max there would overstate device time (and
  understate GTEPS) the same way the old per-PE heatmaps did.
  `relay_total` itself has no single group axis to adjust (it spans two
  different collectives' worth of groups), so its own honest substitute is
  `relay_critical_path_cycles` (see compute_skew_adjusted's docstring) --
  logged as its own column and used in place of relay_total's raw max here.

  Shared by run_bfs.py (one search, verbose=True) and run_graph500.py (64
  searches, verbose=False to avoid flooding stdout with a full phase
  breakdown per search).

  Returns (row_cols, device_time_cycles, profiled_rounds)."""
  if rounds_completed > max_rounds:
    print(f"[[ WARNING: BFS ran {rounds_completed} rounds but max_rounds={max_rounds} -- "
          f"only the first {max_rounds} rounds were timestamped; bump max_rounds to "
          "profile the rest ]]")

  phase_cycles, raw_slots, profiled_rounds = decode_pe_phase_cycles(
      ts_hwl_u32, height, width, max_rounds, rounds_completed)
  skew = compute_skew_adjusted(phase_cycles, raw_slots, height, width)
  relay_critical_path_cycles = skew["relay_critical_path_cycles"]

  total_runtime_cycles, round_duration_cycles = compute_round_summary(raw_slots, profiled_rounds)

  row_cols = {}
  row_cols["round_duration_cycles"] = ";".join(str(int(v)) for v in round_duration_cycles)
  row_cols["total_runtime_cycles"] = total_runtime_cycles
  if verbose:
    print(f"  round_duration_cycles (straggler span, VBCAST_ISSUE->TERM_COL_BCAST_DONE): "
          f"{round_duration_cycles.tolist()}")
    print(f"  total_runtime_cycles (straggler span, round 0 VBCAST_ISSUE -> last round "
          f"TERM_COL_BCAST_DONE): {total_runtime_cycles}  (sum of round_duration_cycles above: "
          f"{int(round_duration_cycles.sum())})")

  device_time_max_by_name = {}  # what device_time_cycles actually sums -- adjusted where possible
  for name, start_slot, end_slot in PHASES:
    cycles = phase_cycles[name].reshape(profiled_rounds, height * width)
    per_round_min = cycles.min(axis=1)
    per_round_max = cycles.max(axis=1)
    per_round_avg = cycles.mean(axis=1)
    row_cols[f"{name}_min_cycles"] = ";".join(str(int(v)) for v in per_round_min)
    row_cols[f"{name}_max_cycles"] = ";".join(str(int(v)) for v in per_round_max)
    row_cols[f"{name}_avg_cycles"] = ";".join(f"{v:.1f}" for v in per_round_avg)
    log_line = (f"  {name:>18s}: min={per_round_min.tolist()} "
                f"max={per_round_max.tolist()} avg={np.round(per_round_avg, 1).tolist()}")

    if name in PHASE_GROUP_AXIS:
      adj_cycles = skew[f"{name}_adjusted"].reshape(profiled_rounds, height * width)
      adj_min = adj_cycles.min(axis=1)
      adj_max = adj_cycles.max(axis=1)
      adj_avg = adj_cycles.mean(axis=1)
      device_time_max_by_name[name] = adj_max
      row_cols[f"{name}_adjusted_min_cycles"] = ";".join(str(int(v)) for v in adj_min)
      row_cols[f"{name}_adjusted_max_cycles"] = ";".join(str(int(v)) for v in adj_max)
      row_cols[f"{name}_adjusted_avg_cycles"] = ";".join(f"{v:.1f}" for v in adj_avg)
      log_line += f" | adjusted max={adj_max.tolist()}"
    else:
      device_time_max_by_name[name] = per_round_max  # local_compute/local_term_cond: real work, no wait to remove

    if verbose:
      print(log_line)

  # relay_total's own raw min/max/avg are already logged above (unchanged);
  # its search-time contribution uses the honest end-to-end span instead --
  # see this function's own docstring.
  device_time_max_by_name["relay_total"] = relay_critical_path_cycles
  row_cols["relay_critical_path_cycles"] = ";".join(str(int(v)) for v in relay_critical_path_cycles)
  if verbose:
    print(f"  relay_critical_path_cycles (skew-adjusted, replaces relay_total's raw max in "
          f"search_time_cycles): {relay_critical_path_cycles.tolist()}")

  device_time_cycles = sum(int(device_time_max_by_name[name].sum()) for name in SEARCH_TIME_PHASES)
  return row_cols, device_time_cycles, profiled_rounds


def decode_pe_phase_cycles(ts_hwl_u32, height, width, max_rounds, rounds_completed):
  """Like decode_phase_row, but keeps the full (height, width) PE-grid shape
  instead of collapsing it to min/max/avg -- for per-PE diagnostics (e.g.
  a heatmap of which PEs are a phase's stragglers, see plot_pe_heatmap.py),
  not the aggregate CSV log.

  Returns (phase_cycles, raw_slots, profiled_rounds): phase_cycles is a dict
  of {phase_name: (profiled_rounds, height, width) int64 array}, one entry
  per PHASES tuple; raw_slots is the same shape per individual TS_* slot
  (keyed by _SLOT_NAMES), needed by compute_skew_adjusted() below since a
  skew correction has to compare DIFFERENT PEs' timestamps for the SAME
  slot, which a plain phase diff (always same-PE, consecutive slots)
  can't express."""
  profiled_rounds = min(rounds_completed, max_rounds)
  ts = decode_round_timestamps(ts_hwl_u32, height, width, max_rounds)
  ts = ts[:, :, :profiled_rounds, :]  # (height, width, profiled_rounds, NUM_TS_SLOTS)

  phase_cycles = {}
  for name, start_slot, end_slot in PHASES:
    cycles = ts[:, :, :, end_slot] - ts[:, :, :, start_slot]  # (height, width, profiled_rounds)
    phase_cycles[name] = np.transpose(cycles, (2, 0, 1))  # (profiled_rounds, height, width)

  raw_slots = {}
  for slot, slot_name in _SLOT_NAMES.items():
    raw_slots[slot_name] = np.transpose(ts[:, :, :, slot], (2, 0, 1))  # (profiled_rounds, height, width)

  return phase_cycles, raw_slots, profiled_rounds


def _chain_entry_ref(start_grid, end_grid, axis, root_of, only_index=None):
  """The real reference point for a reduce_fadds/broadcast group, honoring
  the actual two-sided sequential chain topology (verified against
  collectives_2d/pe.csl's transfer_data_reduce()/configure_broadcast_network()):
  the group splits at `root_of(i)` into a side flowing 0 -> root-1 -> root
  and a side flowing (NUM_PES-1) -> root+1 -> root, each one hop at a time.
  A PE at chain position k can't actually do its own real hop (receive from
  its immediate neighbor, add, forward) until BOTH it has reached its own
  issue point AND that neighbor has genuinely finished -- so its reference
  is `max(own issue, neighbor's own done)`, using the neighbor's ACTUAL
  observed completion (`end_grid`), not the neighbor's issue time. Using
  issue time there (an earlier, and wrong, version of this function) still
  double-counts: a neighbor's own issue can be delayed for a reason that
  has nothing to do with this chain at all (e.g. its own unrelated
  local_compute load that round) while its ACTUAL delivery (done) was
  perfectly on time -- chaining off issue times bled that irrelevant
  upstream delay through every subsequent position, one poisoned straggler
  corrupting the whole side (confirmed on real rmat_s12_e4.mtx data: this
  is what produced wildly round-varying "adjusted" costs for phases that
  move a fixed 1-float payload, and what made `reduce`'s heatmap look like
  only the diagonal PE had real work -- see GRAPH500_BENCHMARK.md section
  13). Chaining off each predecessor's own DONE time is exact: that
  timestamp is a hardware fact that already reflects everything upstream
  of it, so there's nothing further back to separately account for.
  Root's own reference additionally waits on whichever side(s) exist,
  since root can't combine data before both sides have actually delivered.

  only_index restricts computation to a single row/column (for
  relay_row_reduce/relay_row_bcast's row-MID-only optimization -- every
  other row keeps its own raw start unchanged, i.e. wait=0, matching the
  zero-duration placeholder those rows already record)."""
  rounds, height, width = start_grid.shape
  entry_ref = start_grid.copy()

  def _apply(i_or_j, r, size, get_start, get_end, set_ref):
    if r >= 2:
      set_ref(slice(1, r), np.maximum(get_start(slice(1, r)), get_end(slice(0, r - 1))))
    if r <= size - 3:
      set_ref(slice(r + 1, size - 1),
               np.maximum(get_start(slice(r + 1, size - 1)), get_end(slice(r + 2, size))))
    root_neighbors = []
    if r > 0:
      root_neighbors.append(get_end(slice(r - 1, r)))
    if r < size - 1:
      root_neighbors.append(get_end(slice(r + 1, r + 2)))
    if root_neighbors:
      neighbor_done = root_neighbors[0]
      for extra in root_neighbors[1:]:
        neighbor_done = np.maximum(neighbor_done, extra)
      set_ref(slice(r, r + 1), np.maximum(get_start(slice(r, r + 1)), neighbor_done))

  if axis == "row":
    indices = [only_index] if only_index is not None else range(height)
    for i in indices:
      r = root_of(i)
      _apply(i, r, width,
             lambda s: start_grid[:, i, s], lambda s: end_grid[:, i, s],
             lambda s, v: entry_ref.__setitem__((slice(None), i, s), v))
  else:
    indices = [only_index] if only_index is not None else range(width)
    for j in indices:
      r = root_of(j)
      _apply(j, r, height,
             lambda s: start_grid[:, s, j], lambda s: end_grid[:, s, j],
             lambda s, v: entry_ref.__setitem__((slice(None), s, j), v))
  return entry_ref


def compute_skew_adjusted(phase_cycles, raw_slots, height, width):
  """Split each communication phase's raw per-PE duration into `wait`
  (time spent blocked on this PE's own dependency chain -- see
  _chain_entry_ref's docstring) and `adjusted` (what's left: real fabric
  transit once this PE's own side of the chain was actually ready). No
  independent global clock exists to directly timestamp "when did this
  phase really start" -- but every PE in a dependency chain already
  recorded its OWN issue timestamp for the phase, and the WSE is one
  synchronous clock domain across the whole wafer (already relied on
  implicitly everywhere this session compares timestamps across different
  PEs), so those timestamps ARE directly comparable:
      wait[pe]     = chain_ref - own_issue[pe]      (>= 0 always)
      adjusted[pe] = own_done[pe] - chain_ref
      raw[pe]      = own_done[pe] - own_issue[pe] = wait[pe] + adjusted[pe]

  Round-boundary special case: round r's visited_bcast issue time is, for
  every PE, essentially identical to that SAME PE's own relay_col_bcast
  completion from round r-1 (term_col_bcast_done() immediately re-issues
  the next broadcast with no real work in between -- see bool_pe.csl). So
  ALL of visited_bcast's cross-PE issue-time variance at round r>=1 is
  provably already-reported relay_col_bcast variance from the previous
  round, not anything new -- giving it a fresh chain reference here would
  either duplicate that already-reported skew, or (using a literal
  backward-pointing reference) produce negative wait. The honest
  accounting: no NEW wait for visited_bcast at r>=1, its whole raw
  duration is genuinely new cost. Round 0 has no preceding round, so it
  keeps the normal chain reference (see GRAPH500_BENCHMARK.md section 12).

  Returns a dict of {f"{phase}_wait": grid, f"{phase}_adjusted": grid} for
  every phase in PHASE_GROUP_AXIS, plus "relay_critical_path_cycles": a
  (profiled_rounds,) array -- the single honest end-to-end number (latest
  anyone finishes the whole relay, minus the earliest a real diagonal PE
  was ready to start it), independent of the per-phase decomposition."""
  P = min(height, width)
  diag = np.arange(P)
  MID = width // 2

  entry_ref_by_phase = {}
  for phase, axis in PHASE_GROUP_AXIS.items():
    start_slot, end_slot = _PHASE_SLOTS[phase]
    start_grid = raw_slots[_SLOT_NAMES[start_slot]]
    end_grid = raw_slots[_SLOT_NAMES[end_slot]]
    root_of = (lambda idx: idx) if PHASE_ROOT_KIND[phase] == "diagonal" else (lambda idx: MID)
    only_index = MID if phase in PHASE_MID_ONLY else None
    entry_ref_by_phase[phase] = _chain_entry_ref(start_grid, end_grid, axis, root_of, only_index)

  if entry_ref_by_phase["visited_bcast"].shape[0] > 1:
    visited_start_slot, _ = _PHASE_SLOTS["visited_bcast"]
    visited_start = raw_slots[_SLOT_NAMES[visited_start_slot]]
    entry_ref_by_phase["visited_bcast"] = entry_ref_by_phase["visited_bcast"].copy()
    entry_ref_by_phase["visited_bcast"][1:] = visited_start[1:]

  result = {}
  for phase in PHASE_GROUP_AXIS:
    start_slot, end_slot = _PHASE_SLOTS[phase]
    start_grid = raw_slots[_SLOT_NAMES[start_slot]]
    end_grid = raw_slots[_SLOT_NAMES[end_slot]]
    entry_ref = entry_ref_by_phase[phase]

    wait = entry_ref - start_grid
    adjusted = end_grid - entry_ref
    raw = phase_cycles[phase]
    assert np.array_equal(raw, wait + adjusted), (
        f"skew decomposition broke the raw == wait + adjusted identity for {phase!r}")
    assert (wait >= 0).all(), f"{phase!r} produced negative wait -- chain reference is unsound"
    result[f"{phase}_wait"] = wait
    result[f"{phase}_adjusted"] = adjusted

  # end-to-end sanity check: real relay span = latest col_bcast finish
  # anywhere, minus the earliest a real diagonal PE was ready to even
  # start the relay (only diagonal PEs' local-term-cond feeds anything
  # real into relay_col_reduce -- see bool_pe.csl's reduce_done()).
  finish = raw_slots["ts_term_col_bcast_done"]  # (rounds, height, width)
  diag_start = raw_slots["ts_relay_issue"][:, diag, diag]  # (rounds, P)
  result["relay_critical_path_cycles"] = finish.max(axis=(1, 2)) - diag_start.min(axis=1)

  return result


def save_pe_phase_cycles(path, phase_cycles, metadata, structural_grids=None):
  """Save one run's full per-PE-per-round-per-phase cycle grid to a single
  self-contained .npz file, one file per run -- NOT appended across runs the
  way bfs_timing.csv/graph500_searches.csv are. A heatmap needs the cycle
  cost back as a (height, width) grid per phase/round; a CSV would force
  either a wide format that doesn't scale with PE count or a long/tidy
  format that has to be pivoted back into a grid every time it's read,
  neither of which numpy's own binary round-trip needs. `metadata` is a
  dict of small scalars/strings (infile_mtx, pe_grid, source,
  rounds_completed, ...) saved alongside the arrays in the same file --
  read back via load_pe_phase_cycles() in plot_pe_heatmap.py.

  `structural_grids`: optional dict of additional (height, width) 2D
  arrays that aren't per-round timing at all -- e.g. run_bfs.py passes the
  matrix's own per-PE partition counts (local_nnz/local_nnz_cols/
  local_nnz_rows from preprocess_bool.py) here, so plot_pe_heatmap.py's
  sparsity.png can be checked by eye against the timing heatmaps for
  correlation (e.g. does local_compute's imbalance actually track
  local_nnz's imbalance, or is it something else)."""
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  np.savez_compressed(path, **phase_cycles, **(structural_grids or {}), **metadata)


def compute_m_and_gteps(A_coo, device_visited, is_symmetric, search_time_cycles,
                         clock_freq_hz=CLOCK_FREQ_HZ):
  """m (edges traversed) + GTEPS for one search, per GRAPH500_BENCHMARK.md
  sections 4/5. A_coo is the caller's A_csr.tocoo() -- passed in rather than
  recomputed here since run_graph500.py calls this once per search against
  the SAME static matrix.

  is_symmetric picks the counting convention (see GRAPH500_BENCHMARK.md
  section 4): Graph500's own undirected dedup rule, or -- for a directed
  --infile_mtx -- the "edges out of every visited source" convention, still
  real algorithmic work but not a Graph500-spec-comparable number."""
  if is_symmetric:
    m = int(np.sum(device_visited[A_coo.row] & (A_coo.col <= A_coo.row)))
    m_convention = "undirected_dedup"
  else:
    m = int(np.sum(device_visited[A_coo.col]))
    m_convention = "directed_source_visited"
  search_time_seconds = search_time_cycles / clock_freq_hz
  gteps = m / search_time_seconds / 1e9 if search_time_seconds > 0 else float("nan")
  return m, m_convention, search_time_seconds, gteps
