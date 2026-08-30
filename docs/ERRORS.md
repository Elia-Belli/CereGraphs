# Real-hardware error compendium — `bool_diag_spmv` BFS on the ALCF Cerebras appliance

Every distinct error hit while validating and scaling this kernel on real
hardware (`cer-usn-01`), across the RMAT growth sweep and the SNAP graph
sweep. Grouped by whether the cause is confirmed (root-caused, usually
fixed) or still only a probable cause (appliance/compile-farm side, outside
this repo's control). Ordered roughly by how deep into the stack each one
sits: our own kernel code, our own Python driver code, our own
methodology/data prep, then appliance infrastructure.

## Confirmed causes — in our own kernel code

### 1. d2h gRPC message-size ceiling (`parent_local_buf` readback)
```
grpc._channel._InactiveRpcError: StatusCode.RESOURCE_EXHAUSTED
"Sent message larger than max (2147482648 vs. 2147482624)"
```
**Where it first appeared**: RMAT s20 (n=1,049,250), 750x750 grid.
**Cause**: every PE in a BFS row independently records its own valid
one-hop parent candidate for the same local vertex; the design deferred
picking a winner to the host, so all `P` per-row copies (not just one)
were read back — `P*P*blk*2` bytes, growing directly with grid size,
independent of edge count. At P=750 this crosses gRPC's ~2GiB hard
per-message ceiling well before any real WSE-3 compute/memory limit is hit.
**Fix**: new on-device `reduce_select_any` collective
(`src/collectives_2d/pe.csl`) resolves the "any valid winner" pick on-chip
before the transfer, cutting the readback to one value per row position.
Validated on real hardware at s18/s20 post-fix. See
[[appliance-bfs-scale-limit]] for full derivation.

### 2. PE memory overflow from the new collective's own first design
```
ld.lld: error: ran out of PE memory for data (section .bss)
```
**Where**: pokec/topcats, first attempt at wiring in `reduce_select_any`.
**Cause**: the first working version of the new collective used 3 buffers
per PE (send/scratch/recv) at `[blk]u32` each — real static SRAM pressure
at large `blk`, self-inflicted by our own design, not a hardware ceiling
being newly discovered.
**Fix**: redesigned to 2 buffers by reordering the merge (root folds its
own value in as soon as the first side lands, freeing the send buffer's
memory as a landing zone for the second side; non-root PEs reuse their own
otherwise-idle recv buffer). Requested directly by the user ("can you
reduce the amount of space used, doing some ops in place?").

### 3. Task ID collision with memcpy's own reservation
```
task ID '21' bound to more than one task
```
**Where**: first compile attempt after adding the new collective's callback
task.
**Cause**: `layout_bool.csl`'s own comment hedges local task ids "21-23,
27-30, 33-37ish reserved by memcpy" — id 21 turned out to be a real,
non-"ish" collision.
**Fix**: moved to id 24, confirmed free by a real compile succeeding.

## Confirmed causes — in our own Python/methodology code

### 4. h2d gRPC message-size ceiling (matrix-structure upload) — FIXED
```
grpc._channel._InactiveRpcError: StatusCode.RESOURCE_EXHAUSTED
"Sent message larger than max (2147482669 vs. 2147482624)"
```
**Where**: berkstan (confirmed, real edge count, 7,600,595 nnz — matching
SNAP's own documented total exactly), orkut (expected, largest matrix in
the suite).
**Cause, now fully pinned down** (previously only "probable"): the failing
call, `runner.memcpy_h2d(sym_mat_rows_buf, ...)`, uploads the matrix
structure sized by `max_local_nnz` (worst-case per-PE nonzero count after
balancing), not total nnz — web graphs like berkstan have far more extreme
per-PE degree skew than social graphs (pokec) or synthetic RMAT, so
`max_local_nnz` alone blows the ~2GiB ceiling even with a correct,
un-inflated edge count. Reading the actually-installed SDK client
(`cerebras/sdk/client/sdk_appliance_client.py`) confirmed the exact
mechanism: the wire payload is always `4 * width * height * elt_per_pe`
bytes regardless of `data_type` (u16 data gets pre-widened to u32 before
ever reaching the wire), and the SDK's *own* internal chunker has an
off-by-protobuf-envelope bug — it sizes chunk 0 to exactly
`MAX_MESSAGE_LENGTH` (2,147,482,624) raw bytes, then the enclosing protobuf
message serializes ~24-45 bytes larger, tripping the identically-valued
`grpc.max_send_message_length` ceiling. For berkstan @ 750x750
(`max_local_nnz=1102`, computed directly from the on-disk balanced matrix):
`4*750*750*1102 = 2,479,500,000` bytes → the buggy first chunk serializes to
exactly 2,147,482,669 — the literal reported figure.
**Fix**: `device_io.py::memcpy_h2d_chunked` — chunk *our own* calls along
the PE-row axis (not the vendor's buggy internal chunker, and not the
`elt_per_pe` depth axis, which has no precedent/confirmed offset support in
this codebase) into pieces safely under the ceiling (1.5GiB default, real
margin vs. the vendor's ~30-byte margin), so the vendor's internal chunker
never needs to trigger a second chunk. Below the threshold it's a single
unchanged call (zero behavior change for every other graph). Wired into all
six `mat_rows_buf` call sites (`run_bfs.py`, `run_bfs.appliance.py`,
`run_graph500.py`, `run_host_driven_bfs.py`, `run_single_spmv.py`,
`run_transpose_device_test.py`).
**Verified on real hardware**: berkstan @ 750x750 now completes end-to-end
(h2d succeeds, 131 rounds, scipy cross-check OK, 0 mismatches, GTEPS=0.108
incl. transfer / 0.256 excl.). Small-scale regression (`rmat_s10@4x4`)
confirmed byte-for-byte unchanged single-call behavior. Also verified the
**canonical-root case** that used to fail at h2d before this fix: original
SNAP vertex 546279, translated via `snap_berkstan.balanced750x750.operm`
line 546280 → balanced index 353938 (`--shared-perm` balancing preserved
vertex identity correctly) — h2d now succeeds and the scipy cross-check
passes with 0 mismatches, confirming the post-balancing vertex mapping is
still correct with the chunked transfer. (Visited count is small, 18/685500
— expected, not a bug: berkstan is directed, and this particular vertex has
a small out-component from that direction.)

### 4b. ~~`h2d_matrix` cycle-count stat is garbage~~ — CORRECTED: not a bug, a genuinely huge real cost (2026-08-02)
This entry originally claimed the `h2d_matrix: min=/max=/avg=` line printed
implausible cycle counts (29.5B, 72.2B, 595B cycles across berkstan/s18/s20
@ 750x750) due to "likely a 32-bit on-device cycle-counter wraparound," and
marked the issue open/cosmetic. **That diagnosis was wrong.** Investigated
directly (2026-08-02, real ALCF hardware):

1. **No wraparound exists at any bracket duration.** A sleep-sweep
   experiment (`f_sync_hostdevice` → `f_tic` → host `time.sleep(span)` →
   `f_toc`, no real transfer, `span` from 0.05s to 30s against a tiny 4x4
   artifact) showed a flat, proportional `measured_cycles`/`expected_cycles`
   ratio (~0.857, stable) across the *entire* range — no corruption, no
   jump, no sign of a wrap anywhere up to 30 real seconds.
2. **The billions-of-cycles number IS real elapsed time.** Cross-checked
   directly against real RMAT s17 @ 750x750: host-side wall-clock timestamps
   on every printed line showed a **23.77 real-second** gap between the
   "timing h2d: matrix structure upload" print and the next print, while the
   device reported `h2d_matrix` span = 17,700,123,401 cycles — 20.2s at the
   assumed `CLOCK_FREQ_HZ=875MHz`, or 23.6s if the real effective clock is
   closer to 750MHz (matching the ~0.857 ratio from the sleep-sweep almost
   exactly). Either way, the reported cycle count corresponds to genuine
   real time, not garbage.

**Actual finding**: `h2d_matrix` (the matrix-structure upload, 7 separate
appliance-mode `memcpy_h2d` calls per run) genuinely takes on the order of
tens of real seconds for a modest ~3.7M-nonzero matrix — dwarfing every
other cost in the pipeline (`parent_resolve` ~1.8ms, on-device compute
~0.15ms). This is a real, very large performance cost that was previously
mischaracterized as a measurement bug and therefore never investigated as
one. **Does not affect correctness**: `search_time_cycles`/GTEPS still use
the separate, confirmed-correct `round_trip_start_buffer`/
`round_trip_done_buffer` mechanism, unaffected either way.

**Secondary, smaller, less-confirmed finding**: the ~0.857 ratio from the
sleep-sweep hints `CLOCK_FREQ_HZ = 875_000_000.0` (used throughout this
codebase for every reported second/GTEPS figure) may not match the real
device clock (~750MHz?) — plausible given 6/7 × 875MHz = 750MHz exactly, but
not yet isolated from potential gRPC dispatch-latency confounds in the
sleep-sweep methodology. Flagged for a dedicated follow-up, not yet acted on.

**Status**: root cause corrected; the *real* h2d_matrix cost (why 7 appliance
gRPC calls take tens of seconds — per-call dispatch overhead vs. real
bandwidth limits vs. something else) is a genuine, separate optimization
target, not chased further in this session pending user direction.

### 5. Blanket symmetrization of SNAP graphs fabricated edges
**Where**: v2 of the SNAP pipeline (before this fix), all 5 graphs.
**Cause**: `datasets/snap_to_mtx.py` unconditionally computed `A | A^T`
for every SNAP graph. Verified against each dataset's own SNAP
documentation: only `com-orkut` is genuinely undirected. berkstan (web
hyperlinks, SNAP's own page: "25% pattern symmetry"), pokec ("friendships
... are oriented"), topcats, and livejournal are all genuinely **directed**
— symmetrizing them fabricated edges that don't exist in the real graph
(berkstan: 7.6M nnz → 13.3M, nearly doubled). This alone was enough to push
berkstan's h2d transfer over the ceiling in #4, making it look like a scale
limit when it was actually a self-inflicted data error. Caught by the user
directly inspecting the resulting nnz counts and asking "are snap directed
graphs? the edges should stay the same."
**Fix**: symmetrization made opt-in (`--symmetrize`), used only for orkut.
Added `--shared-perm` to `util/analyze.cpp` — same vertex-identity-
preserving row/column permutation sharing as `--symmetric`, but without its
structural-symmetry requirement, so genuinely directed graphs balance
correctly without fabricating symmetry.

### 6. Vertex mapping silently scrambled under load-balancing
**Where**: any SNAP run using a "canonical" source vertex id.
**Cause**: `util/analyze`'s `--rand 0` flag was assumed to disable
permutation entirely; it only gates an *additional* random shuffle on top
of the base load-balancing permutation, which **always** runs. So a source
vertex id meaningful in the original graph pointed at the wrong vertex
after balancing, silently, with no error — caught only because the user
asked "are you running with the correct source?"
**Fix**: added `--operm <file>` to `util/analyze.cpp`, dumping the
permutation (original vertex id → balanced vertex id) so canonical source
vertices can be translated correctly before use.

### 7. Raw/unbalanced SNAP matrices fail to compile
**Where**: attempted per user's "do not balance snap graphs" request.
**Cause**: real, un-mitigated SNAP degree skew (without any load
balancing) blows the same PE static-memory ceiling as #2/#8 — not a bug,
a genuine confirmation that balancing is structurally necessary for these
graphs at this grid size, not just a nice-to-have.
**Resolution**: reverted to a balancing approach, fixed properly via #6's
`--operm`.

### 8. PE static-memory ceiling (compile-time, independent of directedness)
```
ld.lld: error: ran out of PE memory for data (section .bss)
ld.lld: error: ran out of PE memory for task table
ld.lld: error: ran out of PE memory for data (section .data.hi)
```
**Where**: topcats and livejournal, consistently, both before and after
the v3 directedness fix. **Also confirmed 2026-07-29 on as-Skitter
(n=1.70M) and cit-Patents (n=3.77M)** — both new to the pipeline this
session, both hit at 750x750, first attempt.
**Cause**: `blk = ceil(n/P)` grows with `n` at fixed `P=750`; each PE's
static-memory footprint (bitmaps, buffers, task table) scales with `blk`.
topcats (n≈1.79M) and livejournal (n≈4.85M) simply need more per-PE static
memory than a 750x750 grid provides — a real, hardware-imposed compile-time
ceiling, confirmed independent of the symmetrization bug (both graphs
failed identically before and after that fix). Would need a larger PE grid
or a smaller per-PE working set to lift. as-Skitter's n (1.70M) sitting
right at the same threshold as topcats (1.79M) is consistent with this
being an n-driven ceiling, not a per-graph quirk.
**New diagnostic detail (2026-07-29, skitter/patents)**: alongside the
usual `.bss`/task-table/`.data.hi` overflow lines, both also produced
`ld.lld: error: section .bss virtual/load address range overlaps with
.filters` at consistent addresses (`.filters` at `[0xF680, 0xF6DF]` both
times; `.bss` starting at `0x46D8`/`0x4720` and overflowing into that
range) — a genuine static memory layout conflict, not new information
about the failure mode itself, just a second linker diagnostic for the
same overflow. **Important secondary finding**: after these real overflow
errors, the log also fills with `ld.lld: error: cannot open
/tmp/cslc-<hash>/cslc-<hash>.o: No such file or directory` — dozens of
repeats of the SAME single hash within one compile attempt (not a
different hash per attempt, unlike issue #12's pattern). This is a
**downstream symptom of this same PE-memory-overflow failure** (the build
system's retry/cleanup logic re-touching an object file already torn down
after the genuine link failure), NOT an independent instance of issue
#12's remote-scratch-eviction flake, despite the superficially similar
"cannot open ... .o" message. Don't misdiagnose future occurrences of this
message as #12 without first checking for the `.bss`/task-table overflow
lines earlier in the same log — if they're present, it's #8, and retrying
will not help (deterministic capacity, not a transient).
**Status**: real, open, not attempted to fix this session (would require
either more PEs or reducing static memory per PE, e.g. narrower bitmaps).

**New occurrence (2026-08-02): `reduce_select_any_indexed` at RMAT s19,
512x512 (`blk=1024`)**. Same `.bss`/task-table/`.data.hi` overflow
signature, confirmed via the FULL (non-truncated) compile log — the
subsequent "cannot open ... .o"/".ld" lines further down the same log are
the same downstream-cleanup symptom already described above, not a fresh
instance of #12; don't misdiagnose this pattern again. A dense control
compile at the identical config succeeded cleanly in 190s, confirming this
is specific to indexed, not a generic flake at this scale.

Initial hypothesis was that indexed's 6 extra per-PE scratch buffers
(send/recv × bitmap/indices/values, ~12.5KB/PE at `blk=1024`, vs sparse's
~8.4KB 2-buffer workspace) were the cause. **Tested and refuted the same
day**: made all variant-specific buffers' sizes comptime-conditional on
`parent_resolve_variant` (a `const X = if (parent_resolve_variant == N)
real_size else 0;` pattern, guaranteeing zero-size for every variant NOT
selected at a given compile — see #17's own update) and additionally
removed `reduce_select_any_sparse`'s workspace buffers from `bool_pe.csl`
entirely (sparse had no regime where it won, see #16, so no reason to keep
its ~8.4KB of dead-weight-when-unused scratch around at all). Retried the
identical s19-512x512 indexed compile with both changes in place: **failed
identically**, same `.bss`/task-table/`.data.hi` signature. Since this
retest genuinely removed sparse's ~8.4KB from the indexed compile and
nothing changed, the caller-side DATA buffers are not the (sole) driver of
this ceiling — the remaining, more likely cause is CODE size: indexed's
`transfer_data_reduce_select_any_indexed()` FSM has roughly 2x the
sub-transfers/branch states of dense's much simpler transfer function
(see #17), and that code is compiled into the binary unconditionally
regardless of `parent_resolve_variant`'s value (confirmed no `comptime if`
gates it in `collectives_2d/pe.csl`, and the buffer experiment's null
result is itself evidence the compiler isn't eliminating the whole dead
branch, code included, based on a runtime `if` over a `param`).

**Further tested the same day**: also removed `reduce_select_any_sparse()`
entirely from `collectives_2d/pe.csl` (not just its `bool_pe.csl` wiring --
the function, helpers, `Ftype` entry, and FSM, see #16's updated status)
and separately found and removed `scatter()`/`gather()` and their full
supporting infrastructure (10 functions total, see new entry #18) after
confirming they have zero call sites anywhere in this application. Retried
the identical s19-512x512 indexed compile with ALL of today's cleanup in
place (comptime-sized buffers + sparse fully removed + scatter/gather fully
removed): **still failed, but the failure signature changed** —
`.bss` no longer overflows; only `ld.lld: error: ran out of PE memory for
task table` and `ld.lld: error: ran out of PE memory for data (section
.data.hi)` remain. This is genuine partial progress (removing real dead
code did measurably shrink the footprint), just not enough to clear the
two remaining overflows. **Not retried further** — real, deterministic,
and would need a genuine reduction in indexed's own per-hop protocol
complexity (fewer sub-transfers or FSM states, which is what's now driving
both the task-table and `.data.hi` overflow), not further dead-code
removal, to lift.

### 9. `plot_bfs_timing.py` crash on `--max-rounds`-truncated runs
```
AssertionError: round_duration_cycles has 2 entries, expected rounds_completed=10
```
**Cause**: the plotting code assumed `len(round_duration_cycles) ==
rounds_completed` always — true only when every round was profiled in
detail, which breaks as soon as `--max-rounds` is used (deliberately, to
save per-PE `ts_buf` memory) on a run whose true round count exceeds it.
The underlying CSV row was fine; only plot generation crashed.
**Fix**: `profiled_rounds = len(round_duration)` used throughout instead of
trusting `rounds_completed` to match array length.

### 10. `total_runtime_cycles`/GTEPS silently wrong under round truncation
**Cause**: `ts_buf`/`nf_history`/`direction_history` only have
`max_rounds`-many slots; `record_ts()` silently no-ops past that with no
error. Any run whose true round count exceeded `max_rounds` got a
`total_runtime_cycles`/`search_time_cycles`/GTEPS computed by summing only
the *profiled* rounds — silently too low, not obviously wrong, not caught
by any assertion.
**Fix**: added always-correct `round_trip_start_buffer`/
`round_trip_done_buffer` markers, captured unconditionally every round
(overwritten each time, so they hold the true final round's value
regardless of `max_rounds`), used as the authoritative timing source
instead of summing the (possibly truncated) per-round breakdown.

### 11. `preprocess_bool.py` masquerading as "slow compile"
**Cause**: not an error, but a real perf bug worth recording — an
unvectorized, pure-Python O(nnz) triple-nested loop cost ~9-10 minutes on
large SNAP graphs *before* the actual remote compile even started (128 idle
NumPy/BLAS threads observed, not real parallel work), making it look like
the compile itself was slow.
**Fix**: rewritten with vectorized numpy (`np.unique`/`bincount`/`cumsum`
over the full nonzero arrays). Validated bit-for-bit identical against the
original on 8 random test matrices.

## Probable causes only — appliance/compile-farm infrastructure, outside this repo's control

### 12. Remote linker "file vanished" flake
```
ld.lld: error: cannot open /tmp/cslc-<hash>/cslc-<hash>.o: No such file or directory
```
**Where**: livejournal (once), **RMAT s21 (4/4 attempts)**, and **orkut
(3/3 attempts across two separate sessions)** (a different `cslc-<hash>`
every single time, so not a stuck stale artifact — a fresh failure each
attempt).
**Probable cause**: `cslc`'s two-step build (emit per-unit `.o` files into a
per-job scratch dir, then `ld.lld` links them) runs on ALCF's remote
compile-farm container. The `.o` file being gone by link time means the
remote job's scratch space was cleaned, or the container was recycled/
evicted, between the two steps. This has now hit exactly the **three
largest, longest-compiling matrices in the entire test suite** — s21
(largest RMAT attempted), livejournal (2nd-largest SNAP graph, 68.9M nnz),
and orkut (by far the largest matrix overall, 234.4M nnz post-balancing,
3.4x livejournal's size) — and never once a smaller case, which is strong
circumstantial evidence for a **timeout- or resource-linked eviction on the
compile-farm side for long-running/large compiles**, not pure random
flake. This can't be confirmed from the client side (no visibility into
the remote container's lifecycle). Retrying did not resolve it for s21
after 4 attempts; orkut was retried once more in a later session (2
attempts, both failed identically with different hashes) specifically to
test the h2d chunking fix (#4) against it -- never got the chance, since it
still can't get past compile. Treated as an open, size-correlated
infrastructure limitation rather than a transient to retry through
indefinitely, per direct user decision on 2026-07-28 (made for s21, applied
consistently to orkut for the same reason, reaffirmed 2026-07-29 after the
fresh 2/2 failure).

### 13. Transient gRPC `503`/`UNAVAILABLE` during artifact upload
```
grpc._channel._InactiveRpcError: StatusCode.UNAVAILABLE
"Received http2 header with status: 503"
```
**Where**: pokec's first v3 run attempt (compile succeeded; the *run*
step's `SdkRuntime.__enter__` → `upload_artifact` call failed before any
matrix transfer began).
**Probable cause**: a genuine transient connectivity issue between the
client and the appliance's ingress service (`Could not find coordinator
IP:port in cluster details` / `Empty ingress service url` warnings
immediately precede it in the log) — unlike #12, this did NOT reproduce
on an immediate retry (pokec succeeded the very next attempt with no other
changes), which points toward ordinary infra flake rather than anything
scale-correlated.

### 14. "Failed to terminate linker worker processes"
**Where**: topcats, appearing alongside the real PE-memory-ceiling errors
(#8) in the same compiler output.
**Probable cause**: a secondary/cosmetic message from the same failed
compile — the substantive errors in the same output are the `.bss`/task
table/`.data.hi` overflow lines; this is very likely just linker cleanup
choking after the real failure already occurred, not an independent root
cause.

### 15. Directed BFS computed ancestor-reachability, not descendant-reachability — FIXED (code), but reopens berkstan
```
raw vertex 546279: forward (out-edge) BFS reaches 459,847; reverse (in-edge)
BFS reaches 18. Hardware/scipy, sourcing from 546279 (translated), gave 18.
```
**Where**: every genuinely directed SNAP graph (berkstan, pokec, topcats,
livejournal — not orkut, which is symmetrized; not RMAT, which doesn't use
this loader). Invisible until now because every prior check only verified
device-vs-scipy self-consistency, never checked against an independently-
known real-world root vertex's true reachability (caught by the user
directly questioning berkstan's canonical-root visited count against a
published reference table's expected explored-edge count).
**Cause**: `graph_loader.py`'s SNAP edge-list loader built
`coo_matrix((data, (src, dst)))` (row=src, col=dst — the natural
convention). But `bool_pe.csl`'s `compute_topdown()` walks, per local
column `c`, the row-list stored for `c` and marks those rows visited when
`c` is in the frontier — i.e. it computes `y = M @ x`: "row `r` becomes
visited if `r` has an edge **to** some already-frontier column `c`". Given
row=src, that's ancestor-reachability (who points at the frontier), not
descendant-reachability (what the frontier points to). Verified directly:
translating balanced row 353938 back to original vertex ids reproduced
546279's *true out-neighbors* exactly — the matrix and the balancing were
never wrong, only which direction "row=src" gets fed into the kernel's
native (ancestor) SpMV walk.
**Fix**: `graph_loader.py::_load_edgelist` now builds
`coo_matrix((data, (dst, src)))` — i.e. feeds `M = A^T` instead of `A`.
The kernel's native ancestor-of-`M` computation becomes descendant-of-`A`
(standard "vertices reachable via out-edges from source" BFS). Verified:
re-running the exact transpose-then-`breadth_first_order` logic
`run_bfs.appliance.py` already uses for its own scipy check, against the
newly-regenerated berkstan file, now gives 459,847 for vertex 546279 —
matching true forward reachability exactly.
**Consequence — berkstan is temporarily broken again, for a different
reason**: regenerating+rebalancing berkstan with the corrected direction
raised `max_local_nnz` from 1102 to 3481 (the true directed skew is worse
than the accidentally-reversed one) — high enough that it now hits the
*existing* PE static-memory ceiling (#8) at 750×750, the same wall
topcats/livejournal are already stuck behind. This is not a new bug; it's
`max_local_nnz` crossing a threshold that was already known to exist,
independent of source vertex (a compile-time constant from matrix
structure alone) — confirmed by ~100+ repeated `ran out of PE memory for
task table`/`for data (section .data.hi)` errors across many internal
placement attempts in one compile, not the random linker flake (#12).
**This invalidates today's earlier "berkstan h2d fix" success rows**
(`source=0` and the `546279`→`353938` canonical-root run) — both were
computed against the wrong-direction matrix, which happened to have low
enough skew (1102) to fit; the corrected graph doesn't fit at all right
now. Those CSV rows and plots were removed rather than left looking valid.
**Status**: code fix is in and verified correct at the matrix-construction
level; berkstan itself is back to **Open** (needs a bigger grid, joining
#8) until a coarser-than-750×750 option exists.

**Update — real-hardware confirmation, and topcats reconfirmed**:
- **pokec**: regenerated + rebalanced + rerun on real hardware (source=0,
  balanced index) post-fix. Result: `rounds_completed=11`,
  `visited_count=1,504,295/1,633,500`, `m_edges_traversed=30,159,128`.
  This matches an independently-published reference table for pokec
  (diameter 11, ~30.1M explored edges out of 30.6M total) almost exactly —
  genuine on-silicon confirmation of the fix, not just the host-side
  simulation berkstan got. The old pre-fix pokec row (source=1178437,
  computed under the wrong direction) should be treated as superseded.
- **topcats**: regenerated + rebalanced under the fix. `max_local_nnz`
  went from 398 (pre-fix) to **2179** (post-fix) — same "in-degree skew
  far worse than out-degree" pattern as berkstan, confirmed directly this
  time (raw degree check: berkstan max out=249 vs max in=84,208; RMAT s20
  by contrast is exactly out=in=64,701 at every percentile, structurally
  symmetric, hence completely unaffected by this whole bug). Recompiled
  at 750×750: fails, same PE static-memory ceiling as before (246
  `ran out of PE memory` errors this time, more than berkstan's ~100+) —
  not a new failure, an already-failing case failing more decisively.
- **livejournal**: still not regenerated/retested this session — same
  treatment needed, expected to follow the same pattern (already failed
  pre-fix, and directed hyperlink/social graphs so far all show worse
  in-degree skew post-fix).

### 16. Sparse `reduce_select_any_sparse` — correct, but a real ~7.6-8x performance REGRESSION, not a bug
```
parent_resolve avg cycles:  dense    sparse     ratio
  P=8,   blk=128:            6,858.8    52,199.6   7.61x slower
  P=750, blk=175 (real HW):  785,507.8  6,336,928.5  8.07x slower
  P=512, blk=1024 (real HW): 3,045,422.2 23,057,499.9 7.57x slower
```
**Where**: `reduce_select_any` (see #1 above) resolves each row's BFS
parent candidates in a serial ~P/2-hop relay chain toward the diagonal.
Real-hardware instrumentation (a read-only `parent_occupancy` popcount
added to `bool_pe.csl`, zero fabric cost) confirmed occupancy at the point
this collective fires is genuinely low — under 1.2% mean at RMAT s17/s19
scale — the classic setup for a "send fewer bytes" optimization.
**What was built**: a sparse counterpart, `reduce_select_any_sparse()`
(`src/collectives_2d/pe.csl`), with an identical caller-visible dense
`[count]u32`/sentinel contract but an internal bitmap + compact-values wire
format per hop (only real entries cross the fabric, plus a small fixed-size
bitmap; no length word ever travels separately — both sides derive it by
locally popcounting the just-landed bitmap). Two real bugs were found and
fixed via the standalone test harness (`pe_reduce_select_sparse_test.csl`/
`run_reduce_select_test.py --variant=sparse`, all on the free local
simulator, no hardware spent): a fabric-side DSD length that was never
re-set per sub-transfer (surfaced as a simulator "kernel stall" abort), and
a compact-values merge that didn't preserve ascending-bit order (found via
a targeted device-state dump). A third bug — three `Callback`-transition
branches missing the `@activate(ACTIVATE_FSM_TASK_ID)` needed when no
async op is left to trigger it — only surfaced under a very-low-occupancy
P=8 stress test and would have been a near-certain production hang at the
real occupancy this feature targets. All three fixed; final validation is
0 mismatches across P=2..8, occupancy 0.005..0.95, multiple roots.
**Then wired into the real BFS kernel** (a `sparse_parent_resolve` A/B
switch in `bool_pe.csl`, off by default) and timed on real hardware at the
same two configs above. Correctness held (0 mismatches both configs) —
but `parent_resolve` was consistently **~7.6-8x slower**, not faster.
**Root cause, quantified**: this is not fixed per-hop round-trip overhead
dominating (the naive first hypothesis) — fitting `avg_cycles/hops` as a
linear function of `blk` across these three very different scales shows
the dense collective's per-hop cost is ~8.1 cycles/word of `blk` (a
hardware-accelerated `@mov32` DMA-style transfer), while the sparse
collective's per-hop cost is ~81.9 cycles/word of `blk` — about **10x
more cycles per word**, while the FIXED (non-`blk`-scaling) per-hop
overhead only differs by ~3.8x between the two. The dominant cost is
`select_merge_sparse()`/`popcount_bitmap()`'s **manual scalar bit-scan**
(`while (bit < 32)`, a branchy CSL loop touching every bit of every word,
by design O(`blk`) *regardless of occupancy* — see those functions' own
comments) running on **every single hop**, replacing what was one
hardware-accelerated bulk transfer in the dense design. Low occupancy
genuinely shrinks the *bytes moved*, but this design never made the
*compute* scale down with occupancy — and that compute, done the scalar
way, costs far more than the bytes it was trying to save.
**Status**: `reduce_select_any_sparse()` and its standalone test harness
have been **removed entirely** (2026-08-02, alongside #17's follow-up
investigation) — not just unwired from `bool_pe.csl` as originally done,
but fully deleted from `collectives_2d/pe.csl` itself (the function, its
helper functions `compress_dense_to_sparse`/`append_merge_sparse`/
`decompress_sparse_to_dense`/`popcount_word`/`popcount_bitmap`, its
`Ftype` enum entry, its `transfer_data_reduce_select_any_sparse()` FSM,
and every dispatch site referencing it) plus its standalone test kernel
(`pe_reduce_select_sparse_test.csl`/`layout_reduce_select_sparse_test.csl`,
deleted; `run_reduce_select_test.py --variant=sparse` removed from its
CLI choices). There is no regime where the sparse variant was the better
choice (#16), so — unlike indexed, which is a real trade-off kept as an
opt-in — there was no reason to keep any of it around, in `bool_pe.csl`
or in the shared collectives library. Regression-tested clean (0
mismatches, dense + indexed, both the real BFS kernel and the standalone
harness) after the removal. If a hardware-accelerated bit-count design is
ever revisited, the concrete next step would be a `@popcnt`+u16-view idiom
(as `bool_pe.csl`'s own `reduce_done()` already uses for `nz_local`) —
blocked previously by `collectives_2d/pe.csl` having no comptime bound on
`count` to size a `@popcnt` DSD scratch buffer against, since `count` is a
runtime function argument there, not a `dim_params` field.

### 17. `reduce_select_any_indexed` — real 4.5-8.9x SPEEDUP over dense, but hits #8's PE-memory ceiling at blk=1024
```
parent_resolve avg cycles:  dense       indexed     ratio
  P=750, blk=175  (real HW, RMAT s17): 785,507.8    173,864.4   4.52x FASTER
  P=750, blk=700  (real HW, RMAT s19): 3,060,275.1  342,748.4   8.93x FASTER
  P=512, blk=1024 (real HW, RMAT s19): indexed cannot compile -- see #8
```
**Where**: same call site as #16 (`reduce_select_any`,
`term_col_bcast_done()`). Following #16's root cause (sparse's regression is
compute-bound — a manual scalar bit-scan costing ~10x more cycles/word than
dense's hardware `@mov32` transfer, not a bandwidth problem), the natural
question was whether avoiding that scan entirely — rather than trying to
accelerate it with `@popcnt` (blocked, see #16's own "next step" note, by
`collectives_2d/pe.csl` having no comptime bound on `count`) — would do
better.
**What was built**: `reduce_select_any_indexed()` (`src/collectives_2d/
pe.csl`), a structurally different design from sparse: instead of a bitmap
+ position-implicit compact array (which still needs an O(`blk`) scan to
merge, regardless of occupancy), each hop carries an explicit `(row_index,
value)` pair list. `append_merge_indexed()` walks only `incoming_count`
entries — genuinely O(popcount), not O(`blk`) — using the bitmap purely as
an O(1) "have I already got this row" membership test, not as the thing
being scanned. This also makes it safe-by-construction (a forward-only
append, no backward-scan proof needed the way sparse's in-place merge
required).
**Correctness**: validated via the same standalone harness pattern as dense/
sparse (`pe_reduce_select_indexed_test.csl`/`run_reduce_select_test.py
--variant=indexed`) across P=2,3,4,8, occupancy 0.005-0.95, multiple roots —
**0 mismatches on the first try, no bugs found** (unlike sparse's 3 real
bugs during its own development), consistent with the append-only design
being structurally simpler to get right. Then validated on `appliance-sim`
(RMAT s10 8x8, free) before real hardware, per this repo's own convention.
**Real hardware results**: wired via the `parent_resolve_variant` A/B
switch (0=dense, 2=indexed) and measured at two configs — correctness held
in both (0 mismatches, scipy cross-check OK): RMAT s17 750x750 (`blk=175`,
**4.52x faster**) and RMAT s19 750x750 (`blk=700`, **8.93x faster** —
dense 3,060,275.1 vs indexed 342,748.4 avg cycles). The margin *grows* with
`blk`, not shrinks — a genuine, substantial, scale-holding win, not just
"not as bad as sparse."
**Could not measure the third config**: RMAT s19 512x512 (`blk=1024`) hits
the real PE static-memory ceiling described in #8.
**Follow-up investigation (2026-08-02, same day) into shrinking indexed's
footprint to fit `blk=1024`**: audited `bool_pe.csl`/`collectives_2d/pe.csl`
for buffers that are declared but dead/only-conditionally-used. Found and
fixed two real things, neither of which lifted the ceiling:
1. Made all three variants' scratch buffer sizes comptime-conditional on
   `parent_resolve_variant` (`const LEN: u16 = if (parent_resolve_variant
   == N) real_size else 0;` — the same "if-expression on a comptime value"
   idiom `collectives_2d/pe.csl`'s own `task_id_COLOR_0/1` already use for
   `@is_arch`), so a compile only pays for its own active variant's
   scratch, not all three unconditionally. Regression-tested clean on the
   free simulator across all variants.
2. Removed `reduce_select_any_sparse`'s workspace buffers and its
   `parent_resolve_variant == 1` branch from `bool_pe.csl` entirely (both
   the CLI choice and the compile params mapping in `run_bfs.py`/
   `run_bfs.appliance.py`) — sparse had no regime where it won (#16), so
   there was no reason to keep its ~8.4KB of scratch wired in at all. The
   collective itself and its standalone test harness remain untouched in
   `collectives_2d/pe.csl`/`pe_reduce_select_sparse_test.csl` as reference.
   Regression-tested clean (dense + indexed, free simulator).
3. Retried the s19-512x512 indexed compile with both fixes in place:
   **failed identically.** Since this genuinely removed sparse's ~8.4KB of
   dead-when-indexed-is-active scratch and nothing changed, the DATA
   buffers are not the (sole) driver of the ceiling. See #8's own updated
   occurrence note: the more likely remaining cause is CODE size —
   `transfer_data_reduce_select_any_indexed()`'s FSM body is roughly 2x the
   sub-transfers/branch-states of dense's, and (confirmed via grep) none of
   `collectives_2d/pe.csl`'s three collective-specific functions are gated
   by `comptime if`, so all of their code compiles into the binary
   regardless of which variant is actually selected at a given compile.
   (Aside, prompted by "does collectives_2d load code for collectives we
   never use": confirmed via `@bind_local_task` grep that hardware task
   *count* is fixed at 2 per module instance — `fsm` + `f_lock`, doubled to
   4 total since `bool_pe.csl` instantiates the module twice as `mpi_x`/
   `mpi_y` — independent of how many `Ftype` variants exist. So the
   "task table" ceiling isn't about task *count*; it's more likely
   proportional to code/dispatch-site complexity inside the one shared
   `fsm` task body, not chased down to an exact byte accounting.)
**Status**: **kept wired into `bool_pe.csl`** as the `parent_resolve_variant
=2` opt-in path (default remains 0/dense) rather than promoted to the new
default — the s17/s19@750x750 wins are real and substantial, but the
blk=1024 ceiling means indexed cannot currently be used unconditionally at
every grid size this kernel targets. Choosing it is a real trade-off
(faster `parent_resolve`, more code+data) to make deliberately per-config,
not a strict improvement over dense the way this collective's own existence
was over the pre-`reduce_select_any` d2h design (#1). If revisited:
reducing indexed's own per-hop protocol complexity (fewer sub-transfers or
FSM states, not a caller-side scratch trim) is the concrete next step to
lift the blk=1024 ceiling.

### 18. Dead `scatter`/`gather` collectives removed from `collectives_2d/pe.csl`

User asked (2026-08-02, same investigation as #17) whether any code/data
structures in the collectives module are dead or only optionally used,
prompted by confirming that unused-variant CODE (not just data buffers)
compiles into the binary regardless of which `parent_resolve_variant` is
selected (see #17's follow-up). Audited every module-scope function and
buffer in `bool_pe.csl`/`collectives_2d/pe.csl` against real call sites.

Found `scatter()`/`gather()` — full, working implementations (`scatter`,
`gather`, `teardown_scatter_network`, `teardown_gather_network`,
`configure_scatter_filter`, `configure_scatter_network`,
`configure_gather_network`, `transfer_data_scatter`,
`transfer_data_gather_root`, `transfer_data_gather` — 10 functions, each
mirroring `reduce_or`/`reduce_select_any`'s own structural complexity) with
**zero call sites anywhere** in this application: `bool_pe.csl` never
calls them, and grepping every standalone test harness (`pe_reduce_or_test
.csl`, `pe_reduce_select_test.csl`, `pe_reduce_select_indexed_test.csl`)
turned up nothing either. This `collectives_2d` copy is `bool_diag_spmv`'s
own private fork (confirmed no other kernel in the repo imports it), so
there was no hidden external caller. Removed entirely — all 10 functions,
their `Ftype.scatter`/`Ftype.gather` enum entries, every dispatch site
(`initiate_teardowns()`, `teardown_handler_0/1`, the main `transfer_data()`
switch), and a now-dead `fabout_adv` DSD variable only `transfer_data_
gather()` had used. A few stale/copy-paste comments referencing scatter or
gather elsewhere in the file (e.g. a `reduce_fadds()` comment that
mistakenly said "Request a gather network") were cleaned up along the way.

Regression-tested clean (0 mismatches, dense + indexed, both the real BFS
kernel via `run_bfs.py` and the standalone harness) after removal — this
was pure dead-code elimination, no behavior anywhere depended on it.
**Status**: removed. Combined with sparse's full removal (#16), this
measurably shrank the s19-512x512 indexed compile's overflow from three
sections (`.bss`/task-table/`.data.hi`) down to two (task-table/`.data.hi`
only, see #8's updated occurrence note) — real progress, but not itself
sufficient to lift the blk=1024 ceiling. A genuine, safe, permanent
code-size reduction regardless.

## Summary table

| # | Error | Where confirmed | Cause | Status |
|---|-------|-----------------|-------|--------|
| 1 | d2h gRPC 2GiB ceiling | RMAT s20 | design flaw (P copies transferred) | **Fixed** (`reduce_select_any`) |
| 2 | PE mem overflow (new collective) | pokec/topcats | 3-buffer design | **Fixed** (2-buffer redesign) |
| 3 | task id collision | compile-time | id 21 not actually free | **Fixed** (moved to 24) |
| 4 | h2d gRPC 2GiB ceiling | berkstan (fix verified logically; blocked again by #15/#8), orkut | `max_local_nnz` skew + vendor SDK chunker envelope-overflow bug | **Fixed** (`memcpy_h2d_chunked`) |
| 4b | ~~`h2d_matrix` stat garbage~~ → real ~24s cost, not a bug | s17 @ 750x750 (wall-clock cross-checked) | none — corrected misdiagnosis; genuine large real transfer time | **Not a bug** (real cost; optimization target, not chased further) |
| 5 | fabricated symmetrization | v2 SNAP pipeline | wrong default (`A\|A^T` for directed graphs) | **Fixed** (opt-in `--symmetrize`) |
| 6 | scrambled source vertex | any SNAP run | `--rand 0` doesn't disable base permutation | **Fixed** (`--operm`) |
| 7 | raw/unbalanced SNAP fails | user experiment | real degree skew, no balancing | N/A (balancing required) |
| 8 | PE static-mem ceiling | topcats, livejournal, as-Skitter, cit-Patents, `reduce_select_any_indexed` @ s19 512x512 (blk=1024) | `blk` too large / too much per-PE static state for the grid; for indexed, confirmed CODE-size-driven not data-buffer-driven -- removing dead scatter/gather+sparse code shrank the overflow from 3 sections to 2 (task-table/`.data.hi`) but didn't clear it | **Open** (needs bigger grid, less per-PE state, or a lower-complexity FSM) |
| 9 | plot crash on truncation | any `--max-rounds` run | wrong length assumption | **Fixed** |
| 10 | silently wrong GTEPS | any `--max-rounds` run | truncated `ts_buf` history | **Fixed** (round-trip markers) |
| 11 | slow "compile" | large SNAP graphs | unvectorized Python preprocessing | **Fixed** (vectorized) |
| 12 | linker file-vanished flake | s21 (4/4), livejournal (1x), orkut (1/1) | compile-farm scratch/container lifecycle (probable), correlates with the 3 largest jobs in the suite | **Open**, not retried further |
| 13 | transient 503 upload error | pokec (1st attempt) | connectivity flake (probable) | Resolved on retry |
| 14 | "failed to terminate linker workers" | topcats | secondary message alongside #8 | Not independent |
| 15 | directed BFS = ancestor not descendant reachability | berkstan (546279 test vs. reference table); pokec confirmed correct on real hardware (matches reference diameter/EE); topcats reconfirmed still fails (max_local_nnz 398→2179) | edge-list loader fed row=src into a kernel that natively computes ancestor-of-frontier | **Fixed** (code, `graph_loader.py`, verified on real hardware via pokec); berkstan/topcats blocked by #8; livejournal untested |
| 16 | `reduce_select_any_sparse` ~7.6-8x slower, not a bug | RMAT s17 (750x750) + s19 (512x512), real hardware | manual O(`blk`) scalar bit-scan every hop, ~10x more cycles/word than dense's `@mov32` DMA transfer, dwarfing the bytes saved by low occupancy | **Not adopted, fully removed** (no regime where it won; deleted entirely from `collectives_2d/pe.csl` and `bool_pe.csl`, not just unwired) |
| 17 | `reduce_select_any_indexed` 4.5-8.9x FASTER than dense, real win | RMAT s17 (750x750, blk=175) + s19 (750x750, blk=700), real hardware; s19 (512x512, blk=1024) blocked by #8 | explicit `(row_index,value)` pairs give O(popcount) merge vs sparse's O(`blk`) scan; ceiling at blk=1024 traced to indexed's own code size, not caller-side buffers (comptime-sizing + removing sparse AND scatter/gather narrowed but didn't clear it) | **Kept as opt-in** (`parent_resolve_variant=2`; default remains dense since the code/data cost blocks the largest-`blk` grids, see #8) |
| 18 | Dead `scatter`/`gather` collectives, zero call sites | `collectives_2d/pe.csl` (bool_diag_spmv's private fork) | vestigial from the original library; 10 functions, never called by this kernel or any of its test harnesses | **Removed** (regression-tested clean; shrank but didn't clear #8's blk=1024 ceiling) |
