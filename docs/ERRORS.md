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

**Update (2026-09-07): this specific instance is now CLEARED.** After #19's
compacted `parent_local_buf` replacement (`.bss` reduction) and #20's fix
(which re-grew the send-side buffers, but only back to `blk`-sized — same
as before #19, not larger), a fresh RMAT s19 512x512 (`blk=1024`) indexed
compile **succeeded**: `Compilation done in 140.4s`, exit 0, real (non-stub)
per-tile output confirmed via `cs_readelf -m` — peak per-tile static memory
**46,032 bytes**, comfortably under the PE ceiling (real margin, not a
razor's-edge pass). Net effect: #19's `.bss` saving (`parent_local_buf`,
`blk`-sized, → ~0) plus #18's dead-code removal (scatter/gather, sparse)
together made up enough headroom to absorb #20's fix growing the send-side
buffers back to `blk`-sized. This does not mean the underlying code-size
driver identified above (indexed's FSM being ~2x dense's sub-transfers/
branch-states) is gone — it means the combined effect of #18+#19+#20 is
currently enough margin at THIS specific (s19, 512x512) config, and the
margin is thin: 46,032B against a ~49,152B ceiling is only ~3.1KB of slack
(~6.3%). **Confirmed NOT universal, same day**: RMAT s20 750x750
(`blk=1399`, `max_local_nnz_rows=92` — ~37% larger `blk` than the s19
512x512 case above) hit the identical `.bss`/task-table/`.data.hi` overflow
again, indexed-variant compile failed outright. So the ceiling moved, it
didn't lift — a larger `blk` (whether from a bigger scale at the same grid,
or the same scale at a smaller grid) can still exceed the thin margin
#18+#19+#20 bought back. RMAT s21 750x750 (`blk≈2797`, #19's own real-scale
reference config) not attempted after this negative result — predictably
worse, not a useful data point. See #20 for the fix details and #19 for the
(revised) memory numbers.

**Update (2026-09-11): this specific s20 750x750 failure is now CLEARED
too.** After #21's bottom-up-only redesign (CSC-by-source structure,
on-device transpose scratch, and — #23 — the entire dense variant all
removed), the identical `blk=1399` config compiles clean: `Compilation
done in 289.2s`, exit 0, peak per-tile memory (`cs_readelf -m`) **47,120
bytes** — real but thin margin against the ~49,152B ceiling, only ~2KB
(~4.2%), tighter than #21's own s19 512x512 result (~8.4KB/17%) since
`blk=1399` is substantially larger. Not yet re-verified for actual BFS
correctness at this scale (compile-only so far) — see #21's own entry for
the s19 512x512 case's full real-hardware verification, which this s20
config hasn't been taken through yet.

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

### 19. Sparse (compacted) parent-candidate storage for `reduce_select_any_indexed` — real `.bss` reduction, but caught a real intra-round duplicate-append bug during development

**Where**: `bool_pe.csl`'s `parent_local_buf` (the per-PE dense `[blk]u32`
send-side buffer `compute_topdown()`/`compute_bottomup()` write into and
`term_col_bcast_done()` feeds to the parent-resolution relay) and, for
`parent_resolve_variant == 2` only, the matching `[blk]u16`/`[blk]u32`
`parent_send_indices`/`parent_send_values` scratch #17's indexed collective
needed to compress it into wire format.

**Root cause of the cost (not a bug, a sizing choice)**: a PE can only ever
record a parent candidate for a local row it has a real incoming structural
edge to — bounded by `max_local_nnz_rows` (already computed by
`preprocess_bool.py`, already threaded as a compile param), not `blk`. At
real scale (RMAT s21 750x750: `blk=2797`, `max_local_nnz_rows=152`) that
bound is a small fraction of `blk` (~5.4% in that config), so the dense
send-side buffers were paying for far more slots than could ever be used —
a real, quantifiable, directly A/B-compiled (`cslc` + `cs_readelf -m`)
contributor to the "ran out of PE memory" failures in #8/#17, on top of the
FSM code-size driver #17 already identified there.

**Important scope limit, checked directly against source before building
anything**: this only applies to `parent_resolve_variant == 2` (indexed).
Dense's own reduce protocol (`reduce_select_any`/
`transfer_data_reduce_select_any`, the production default and what
actually failed on the real scale-21 hardware run referenced in #1/#8) has
no index-array concept anywhere — every hop moves a fixed-size positional
block via DMA and merges it with a plain positional compare
(`select_merge_u32`). It cannot be pointed at a compacted array at any size
without ceasing to be dense. So this does **not** fix the original
scale-21 dense-variant failure — only extends indexed's own reach (e.g.
grids like the documented `blk=1024` case in #8/#17 that indexed currently
can't compile at).

**What was built**: `parent_compact_indices`/`parent_compact_values`
(`bool_pe.csl`), an incrementally-appended, discovery-order compacted pair
sized by `max_local_nnz_rows` instead of `blk`, replacing
`parent_local_buf`/`parent_send_indices`/`parent_send_values` for
`variant == 2` (comptime-zero-sized there instead; dense's own buffers
untouched). Two pure additions to `collectives_2d/pe.csl`:
`build_bitmap_from_indices()` (derives the membership bitmap
`reduce_select_any_indexed`'s wire protocol needs, since an incrementally
built pair has no bitmap for free the way a dense-array compress pass
does) and `reduce_select_any_indexed_precompacted()` (identical contract
and wire protocol to #17's `reduce_select_any_indexed()`, just skipping its
internal `compress_dense_to_indexed()` call and aliasing the caller's
already-compacted arrays straight into the FSM's scratch — reuses the
entire FSM/teardown path unmodified, no new `Ftype`).

**A real correctness bug found and fixed during development, before this
ever reached hardware**: the first version reused
`visited_bitmap`'s existing "not yet discovered" gate as the *only*
duplicate check on `compute_topdown()`'s append site, reasoning that a row
already visited can never be re-appended. True across rounds, but **not**
within one round: `compute_topdown()`'s nested loop (every frontier
column, then every row it touches) can hit the *same* not-yet-visited row
from two different local columns inside one call — harmless for the old
dense scheme (repeated writes to the same `parent_local_buf[k]` slot, last
write wins, no growth) but silently double-appended that row into the
compacted pair, inflating `parent_compact_count` past its true
`max_local_nnz_rows` bound and eventually tripping the bounds guard,
silently dropping later genuine discoveries. Caught immediately by this
repo's own scipy cross-check at the smallest smoke-test scale (RMAT s8,
4x4 grid, source 0): `10/256` visited-set mismatches, `2` invalid parents,
device `visited_count=179` vs scipy's `189` — with the *dense* variant
passing cleanly (`0` mismatches) on the identical matrix/grid/source,
isolating the regression to this change rather than something
pre-existing. Root-caused to the missing within-round dedup described
above (not a relay/collective bug — `reduce_select_any_indexed_
precompacted()`'s cross-PE merge was never reached with corrupted input
until this was fixed). Fixed by adding `parent_round_seen_bitmap`, a
per-round (not per-run) dedup bitmap reset every round alongside
`y_bitmap`/`y_bitmap_reduced`, gating the append so only the first hit on a
given row within a round is recorded (`compute_bottomup()` needs no
equivalent gate — `mat_row_idx_buf` already lists each distinct local row
at most once by construction). Re-verified clean after the fix: RMAT s8
4x4, sources {0, 5, 50}, both plain and `--directional` (confirming
`compute_bottomup()`'s own path, unaffected by the bug, still correct) —
**0 mismatches, 0 invalid parents, scipy cross-check OK** in every case.
**Status of correctness verification**: confirmed only at this small
smoke-test scale (RMAT s8, 4x4 grid) so far, per this session's own
scope — larger-grid A/B (matching #16/#17's own real-hardware scales) and
a real appliance re-attempt at a previously-blocked config (e.g. #8/#17's
`blk=1024` case) are follow-up work, not yet run.

**Memory measured** (direct `cslc` + `cs_readelf -m` A/B, this session's
own smoke-test scale — RMAT s8, 4x4 grid, `blk=64`, `max_local_nnz_rows=55`;
note this config's `max_local_nnz_rows`/`blk` ratio (~86%) is far denser
than #17's real-scale s21 config (~5.4%), so the *absolute* saving below is
correspondingly small — it's a directional confirmation, not the
real-scale number, which needs the larger-grid A/B noted above to measure
directly):
```
                         dense (variant=0)   indexed (variant=2)
before this change:      24,704 B             27,088 B
after this change:       24,704 B (unchanged) 26,320 B  (-768 B, -2.8%)
```
Dense's byte count is bit-for-bit identical before/after (confirmed via
the same A/B) — zero regression there, as expected from the scope limit
above.

**Status**: landed for `parent_resolve_variant == 2` only. Does **not**
supersede #8's residual code-size-driven ceiling at very large `blk` (see
#17's own follow-up investigation — FSM code size, not caller-side
buffers, is the larger remaining driver there), and does **not** fix the
original scale-21 dense-variant failure referenced in #1/#8 — only shrinks
indexed's own per-PE data footprint. Larger-grid measurement (to see the
saving at a `max_local_nnz_rows`/`blk` ratio closer to #17's real-scale
numbers) and a real appliance re-attempt at a previously-blocked config were
the concrete next steps — see #20, which found a second real bug during
exactly that follow-up and revises the send-side memory picture above.

### 20. `reduce_select_any_indexed_precompacted()` send-side aliasing overflow — real crash at the very first mid-scale test, caught before appliance time was spent

**Where**: `reduce_select_any_indexed_precompacted()`'s `my_indices`/
`my_values` parameters (`collectives_2d/pe.csl`), as called from
`bool_pe.csl`'s `term_col_bcast_done()` with `&parent_compact_indices`/
`&parent_compact_values` (#19's new compacted storage) passed straight
through instead of a `blk`-sized scratch copy.

**Symptom**: real crash, not a slow compile or a silent wrong-answer —
confirmed on the very first test in #19's own "not yet run" follow-up list
(RMAT s12, 8x8 grid, source 0, free simulator): a fatal `hcf` (halt and
catch fire) fault at simulated cycle ~485,166, tile `P8.7`, stack trace
through `vflag_set`/`e_process` (`hwtile.c:5647`) — i.e. a hardware-level
fault from an out-of-bounds fabric-side write, not a Python or compile-step
error (compile itself succeeded in ~5s, identical to dense). Bisecting on
`--max-rounds` (1, 2, 3) showed the fault lands at essentially the same
cycle (473,251 / 473,517) regardless of how many total rounds are
configured — it fires within round 1's own execution, independent of
occupancy buildup across rounds, ruling out the "grows until it exceeds
`max_local_nnz_rows`" theory #19's own "what cannot shrink" scope note had
flagged as the risk to watch for.

**Root cause**: #19's `reduce_select_any_indexed_precompacted()` aliases the
caller's `my_indices`/`my_values` directly as the relay's send-side scratch
(`indexed_send_indices`/`fsm_state.send_buf`), sized only to this PE's own
`max_local_nnz_rows`. But `transfer_data_reduce_select_any_indexed()`'s
"middle root" branch (root not at an extreme PE — true here, and always
true in this kernel, since `bool_pe.csl` deliberately roots at `MID`, see
commit `852c838`) reuses that *same* send-side memory as the landing zone
for side B's incoming data once this PE's own contribution has been merged
away (pass 7/8, `fsm_state.counter == 6/7`). Side B is itself already the
merged UNION of every PE further down that relay chain — bounded by
`blk`/`count`, not by any single PE's `max_local_nnz_rows` (the recv-side
buffers already account for exactly this, per #19's own `INDEXED_ARRAY_LEN`
comment — the send-side aliasing just didn't carry the same reasoning
over). Writing up to `blk`-many entries into a `max_local_nnz_rows`-sized
buffer silently overflows into whatever memory follows it on that PE.

**Confirms itself via the crash coordinates**: `MID = P/2 = 4` for this 8x8
grid; with `--fabric-offsets=4,1`, tile `P8.7`'s core column is `8-4=4` —
column `MID`, exactly the root, exactly where the overflowing branch runs.
Not a coincidence.

**Why #19's own smoke-test (RMAT s8, 4x4) didn't catch this**: at P=4,
`MID=P/2=2` is still a genuine middle position, so this isn't a topology
difference — more likely occupancy at that tiny scale/those sources never
pushed a side-B union past `max_local_nnz_rows=55`'s slack before the run
ended, whereas the RMAT s12/8x8 config's higher absolute occupancy did.
Underlines that this class of bug needs a real mid-scale run to surface,
exactly as #19's own follow-up list anticipated in spirit (if not in the
specific mechanism predicted).

**Fix**: restored a dedicated `blk`-sized send-side scratch pair
(`parent_send_indices`/`parent_send_values`, `bool_pe.csl`) — the same
buffers #19 had removed — and copy `parent_compact_indices`/`values`'
`[0, parent_compact_count)` prefix into them (`O(parent_compact_count)`,
still far cheaper than a full `O(blk)` `compress_dense_to_indexed()` pass)
before calling the relay. `parent_compact_indices`/`values` themselves stay
`max_local_nnz_rows`-sized — this PE's own storage footprint keeps #19's
saving — only the buffer actually handed to the collective grew back.
Corrected `reduce_select_any_indexed_precompacted()`'s own doc comment in
`pe.csl`, which had asserted the now-disproven "safe to alias directly"
claim (it only reasoned about `my_bitmap`'s landing-zone reuse, missing
that indices/values get the same treatment).

**Re-verified after the fix**: RMAT s12 8x8, sources {0, 1}, free
simulator — `0` FATAL in `sim.log`, `0/4096` visited-set mismatches, `0`
invalid parents, scipy cross-check OK in both cases (dense-variant parity
confirmed, same visited/parent counts as before this bug existed).

**Revises #19's memory numbers**: #19's projected saving assumed the
send-side buffers could shrink to `max_local_nnz_rows` too. They can't, per
this bug — the send side must stay `blk`-sized for the relay's landing-zone
trick to be safe, same size as #17's original (pre-#19) design. The real
net saving from #19+#20 together is narrower than #19's own writeup
implied: `parent_local_buf` (`blk`, `u32`) → 0, replaced by
`parent_compact_indices`/`values` (`max_local_nnz_rows`-sized) plus one new
`parent_round_seen_bitmap` (`BITMAP_WORDS`) — a real saving whenever
`max_local_nnz_rows < blk`, but smaller than #19's own before/after byte
counts suggested, since those numbers were taken before this bug (and its
fix) existed. A fresh `cslc`+`cs_readelf -m` A/B at real scale (plan item 2,
still not run) is needed for an accurate updated number.

**Status**: **Fixed**, re-verified on free simulator at RMAT s12 8x8 (2
sources). The blk=1024 compile-ceiling re-attempt (#8/#17) this fix put in
doubt was tried immediately after and **succeeded** — RMAT s19 512x512
compiled clean (140.4s, peak per-tile 46,032B, real margin) for the first
time; see #8's own updated occurrence note for the full number. Not yet
re-verified at #19's own real-scale correctness target (RMAT s21 750x750,
full run not just compile) or against a skewed-degree (non-RMAT) graph.

### 21. Bottom-up-only BFS: top-down and the on-device CSC→CSR transpose removed entirely; `parent_resolve_variant==2`'s local storage redefined to `parent_values` (position-aligned to `mat_row_idx_buf`)

**Motivation**: memory capacity, not performance, was made the explicit
priority for this change. #19/#20's discovery-order compacted scheme
(`parent_compact_indices`/`values`/`count` + `parent_round_seen_bitmap`)
exists specifically because `compute_topdown()`'s nested loop can hit the
same undiscovered row twice within one round — remove top-down entirely
and that whole dedup problem disappears, since `compute_bottomup()`'s
single linear pass over its resident row list (`mat_row_idx_buf`) never
revisits a row twice per round. And since bottom-up no longer needs to be
switched into mid-run (Beamer et al.'s adaptive direction-optimizing switch
is gone — accepted, deliberate tradeoff, not an oversight: bottom-up's
per-round cost floor is `local_nnz_rows[0]`, independent of actual frontier
size, so tiny-frontier rounds now pay what only big "frontier" rounds used
to pay; not measured numerically as part of this change), the host can
upload the matrix already transposed (CSR-by-destination), removing the
on-device transpose and its scratch (`row_to_bucket`/`cursor_buf`/
`visited_pos`) and the CSC-by-source structure (`mat_col_idx/loc/len_buf`,
`max_local_nnz_cols`) entirely — nothing reads them anymore.

**Where**: `bool_pe.csl` (removed `compute_topdown()`, `transpose_structure()`,
`col_of()`, `is_bottom_up`/`tau_switch_count`/`direction_history` and their
buffers; `task compute()` now unconditionally calls `compute_bottomup()`;
`term_col_bcast_done()`'s parent-resolve branch rewritten, see below),
`layout_bool.csl` (dropped params/exports to match), `device_io.py`
(`csl_compile_core`/`csl_compile_core_appliance` signatures), `run_bfs.py`/
`run_bfs.appliance.py` (`preprocess()` call-site CSR/CSC argument-pair
swap, symbol renames, dropped `--directional` flag and transpose/direction
readback+CSV columns), `bfs_timing.py` (`check_round_vs_total_communication`
signature), `plot_bfs_timing.py` (dropped the transpose segment/color).
`preprocess_bool.py` itself needed **zero internal changes** — it already
takes independent CSR-role and CSC-role array arguments, and for a square
matrix on a square grid with `.sorted_indices()` applied, `csc(A^T) ==
csr(A)` and `csr(A^T) == csc(A)`, so swapping which physical array goes
into which parameter slot at the call site produces the transposed layout
for free.

**What was built**: `parent_values: [max_local_nnz_rows]u32`, position-aligned
to the resident `mat_row_idx_buf` (`parent_values[i]` is the parent for
local row `mat_row_idx_buf[i]`) — replaces `parent_compact_indices`/`values`/
`count` and `parent_round_seen_bitmap` entirely: `compute_bottomup()`
writes `parent_values[i] = global_c` directly (the loop already has `i` in
hand — no bounds check, no dedup gate needed). `start_spmv()`'s reset
becomes an `O(local_nnz_rows[0])` sentinel fill (real, but bounded, paid
once per run — the discovery-order scheme's reset this replaces was O(1),
since it only needed a running count, not every slot touched).
`term_col_bcast_done()`'s convergence branch (confirmed to fire **exactly
once per run**, never per-round) does a one-time compress-scan over
`parent_values`/`mat_row_idx_buf`, pulling real row numbers (not positions)
for cross-PE correctness, that does double duty as both the membership-
bitmap source and the `blk`-sized copy-into-scratch #20's fix already
required — no new per-call cost class, just a different scan predicate
over the same buffers.

**Host readback needed zero changes**: the relay's own on-device
`decompress_indexed_to_dense()` step already produces a dense `[blk]`
`parent_relay_result` (exported as `"parent_local_buf"`) regardless of how
the send side was populated — this invariant, already true before this
change, is what made the redesign possible without touching `device_io.py`'s
`extract_parent_result()` at all.

**Verification**: kernel compiles clean (both variants, `cslc` direct and
via `run_bfs.py --compile-only`) at the RMAT s12 8x8 smoke scale.
**Correctness re-run initially FAILED — see #22 for a real bug this
surfaced and its fix; #23 for the subsequent removal of the dense
variant entirely.**

**Known, accepted breakage (out of scope, per explicit decision)**:
`run_graph500.py` and `bfs/scripts/commands_wse3_graph500.sh` both assume
the CSC-by-source upload path (always dense/top-down, never
`--directional`) and will fail once compiled against the new kernel —
`run_graph500.py` calls `preprocess()` unswapped and `csl_compile_core()`
with the old positional-argument shape; `commands_wse3_graph500.sh`
hardcodes `--params=...,max_local_nnz_cols:4,...` directly to `cslc`, which
no longer declares that param. Not fixed — left as a documented, deliberate
break, not a regression to chase.

**Status**: **Fully verified**, smoke through real hardware. See #22 for
the real correctness bug this design's own first correctness run
surfaced, and its fix. Mid-scale + real-hardware re-verification (2026-09-11,
RMAT s19 512x512, blk=1024 — the smallest config #8/#17 originally
documented as failing to compile): both the free simulator and real WSE-3
hardware now compile this config cleanly and run it correctly --
`0/524288` visited-set mismatches, `0` invalid parents, scipy cross-check
OK on real hardware (291s execution, 6 rounds). Fresh `cslc`+`cs_readelf -m`
peak per-tile memory at this scale: **40,736 bytes**, down from the
46,032B measured right after #19/#20 alone (before this entry's own
CSC-structure/transpose-scratch/dense removals) -- margin against the
~49,152B ceiling grew from ~3.1KB/6.3% to ~8.4KB/17%. First real
performance numbers for this exact kernel: 0.55 GTEPS (full), 6.23 GTEPS
(excl. h2d/d2h), 44.3 GTEPS (excl. parent_resolve too). One thing to watch:
both the compile and run jobs logged an `InconsistentVersion` warning
(client 1.14.0 vs. cluster server 1.20.2) -- did not block either job this
time, but worth checking first if something looks off on a future
appliance run.

### 22. `preprocess()` call-site swap silently transposed the (px,py) PE-grid axes — real bug caught by #21's own first correctness run

**Where**: `run_bfs.py`/`run_bfs.appliance.py`'s `preprocess()` call site
(the CSR/CSC argument-pair swap #21 introduced to get a CSR-by-destination
layout with zero changes to `preprocess_bool.py`).

**Symptom**: #21's first real correctness run (not just `--compile-only`)
failed identically at every source tried (0, 5, 50), on RMAT s12 8x8:
`303/4096` visited-set mismatches vs. scipy, `visited_count=3169/4096`
(scipy's own true count is `3342/4096`) — the SAME failure signature
regardless of `parent_resolve_variant`, which was the key diagnostic: since
dense's own parent-storage code (`parent_local_buf`) was completely
untouched by #21's redesign, an identical failure on both variants meant
the bug had to be upstream of parent storage entirely, in the shared
traversal/data path.

**Root cause, found via a standalone host-side test** (not the real
kernel — just `preprocess_bool.preprocess()` called directly, checked
against `scipy`'s own CSR ground truth for a small hand-built graph on a
2x2 grid, then confirmed at 4x4/8x8): `preprocess()`'s own block-placement
formula (`block_id = row_b*fabx + col_b`, reshaped as `(faby, fabx, ...)`)
treats whichever array is fed into the `cscColPtr`/`cscRowInd` parameter
role as the one whose per-nonzero "row_b" becomes the array's FIRST
output axis. Under #21's swap, the CSR-role data (`A_csr`, genuinely
indexed by row) was fed into that slot -- its own per-nonzero "row_b"
computation ends up numerically equal to the actual **column**-block index
(px), not the row-block index (py), because of how the mislabeled
row/col-per-nonzero values interact with the `bx`/`by` divisors (this only
avoids an outright shape mismatch because the kernel's grid is always
square, `fabx == faby`, per `bool_pe.csl`'s own `prows == pcols` assert --
it does NOT save the block-placement math from being transposed). Net
effect: `preprocess()` still returns fully correct DATA (confirmed via the
standalone test) but with its returned arrays' first two axes silently
swapped -- `matrix_info[...][px, py, :]` where every other caller convention
(and the un-swapped, pre-#21 code) expects `[py, px, :]`. Every off-
diagonal PE (`px != py`) received its transpose partner's block; diagonal
PEs (`px == py`) were coincidentally unaffected, which is why the bug
wasn't a crash or a gross shape error -- just wrong BFS results.

**Confirmed NOT an artifact of the swap-trick specifically**: also tried
computing the actual transposed matrix (`A_csr.transpose()`) and calling
`preprocess()` UNSWAPPED on it -- algebraically identical to the swap (per
the same `csc(A^T)==csr(A)` identity #21 relied on, confirmed byte-for-byte
via `np.array_equal`), and empirically produces the exact same transposed
result. So this is a genuine, inherent property of using `preprocess()`
this way (feeding it a "logically transposed" input), not a mistake
specific to the manual argument-swap framing.

**Fix**: transpose the first two axes of every array `preprocess()`
returns that's derived from the (now CSR-role-fed) compact-array branch --
`mat_row_idx_buf`/`mat_row_loc_buf`/`mat_row_len_buf`/`mat_rows_buf`/
`local_nnz`/`local_nnz_rows` -- via `np.transpose(..., (1, 0, 2))`
immediately after extraction from `matrix_info`, in both `run_bfs.py` and
`run_bfs.appliance.py`. `local_nnz` itself is no longer read on-device
(its only consumers, `compute_topdown()`/`transpose_structure()`, were
removed in #21) but was transposed too, for consistency with the host-side
`--dump-pe-timing` structural-grid diagnostic.

**Verification**: standalone test (`preprocess()` output vs. scipy ground
truth, per-PE, per-row) passes at 2x2, 4x4, and 8x8 grids after the fix.
Real kernel re-run (RMAT s12 8x8, indexed variant, sources {0, 5, 50}):
`0/4096` mismatches, `0` invalid parents, scipy cross-check OK, at every
source -- matching #19's own pre-#21 clean baseline exactly
(`visited_count=3342/4096` in both).

**Status**: **Fixed**. This was the actual, sole blocker for #21's design --
no further changes to the bottom-up-only traversal or `parent_values`
storage itself were needed.

### 23. Dense (`parent_resolve_variant==0`) removed entirely

**Where**: `bool_pe.csl` (`parent_local_buf`/`PARENT_LOCAL_BUF_LEN`,
`occ_bitmap`/`occ_popcnt_scratch`/`occ_popcnt_src_dsd`/`occ_popcnt_dst_dsd`,
the `param parent_resolve_variant` declaration itself, and every `if
(parent_resolve_variant == 2) ... else ...` branch -- `INDEXED_BITMAP_LEN`/
`INDEXED_ARRAY_LEN`/`PARENT_VALUES_LEN` are now unconditional constants),
`layout_bool.csl` (matching param/wiring removal), `device_io.py`
(`csl_compile_core`/`csl_compile_core_appliance` lost the
`parent_resolve_variant` parameter), `run_bfs.py`/`run_bfs.appliance.py`
(`--parent-resolve-variant` CLI flag and its `{"dense":0,"indexed":2}`
mapping removed -- the indexed/sparse path via
`reduce_select_any_indexed_precompacted()` is now the kernel's only
parent-resolution behavior).

**Reasoning**: once bottom-up became the only traversal strategy (#21),
dense's own `parent_local_buf` scheme had no remaining reason to exist
side-by-side with `parent_values` -- it was never the point of this
session's redesign, just carried along as an unaffected fallback. With
#22's fix confirming the indexed/sparse path is fully correct, keeping
dense around was pure maintenance surface (a second buffer scheme, a
second relay call, `occ_bitmap`'s own separate occupancy-instrumentation
path) for a variant nobody intended to keep using. Explicit user decision,
not a default assumption.

**Verification**: kernel compiles clean with `parent_resolve_variant` fully
absent from both `bool_pe.csl` and `layout_bool.csl`. Real kernel re-run
(RMAT s12 8x8, sources {0, 5, 50}, no variant flag): `0/4096` mismatches,
`0` invalid parents, scipy cross-check OK at every source -- identical
numbers to #22's own post-fix verification, confirming the removal itself
introduced no regression.

**Status**: **Done**. The kernel now has a single parent-resolution path;
`--parent-resolve-variant` no longer exists as a CLI concept.
`docs/COLLECTIVES_TOPOLOGY.md`'s own variant-comparison table is
`collectives_2d/pe.csl`-level (the collectives themselves, `reduce_or`/
`reduce_select_any`/`reduce_select_any_indexed`, are all still present in
that library file as reference/history, per #16/#17/#18's own "kept as
reference" convention) and needs no change -- only `bool_pe.csl`'s own
CALLER-side selection of which collective to use was narrowed to one.

### 24. On-device parent-aggregation relay eliminated entirely; per-row combine moved host-side

**Where**: `bool_pe.csl` (deleted `parent_relay_result`, `parent_send_bitmap`/
`parent_recv_bitmap`, `parent_send_indices`/`parent_recv_indices`,
`parent_send_values`/`parent_recv_values`, `INDEXED_BITMAP_LEN`/
`INDEXED_ARRAY_LEN` and their pointer vars; `term_col_bcast_done()`'s
convergence branch now just scans `parent_values`/`local_nnz_rows` for an
occupancy count, no compress-scan/bitmap-build/relay-call; `parent_values`
itself exported directly under its own name instead of via a
`parent_relay_result` alias), `layout_bool.csl` (`@export_name` updated
to match), `collectives_2d/pe.csl` (deleted both `reduce_select_any` and
`reduce_select_any_indexed` -- the indexed variant because its sole caller
just went away, the dense variant as bonus cleanup since it had zero
callers already, see #23 -- along with their module-scope scratch,
`Ftype` tokens, the now-dead `Network_Config.sentinel` field, and their
FSM bodies/dispatch-switch cases), `device_io.py` (`extract_parent_result()`
rewritten to take every PE's own `(mat_row_idx_buf, parent_values,
local_nnz_rows)` triple and do the per-row combine itself, vectorized with
numpy fancy-indexing scatter-assign), `run_bfs.py`/`run_bfs.appliance.py`
(d2h readback replaced: one `parent_local_buf` column ->
whole-grid `parent_values`; call site updated to the new
`extract_parent_result()` signature), `plot_bfs_timing.py` (docstring
line only), `run_snap_sweep.sh` (stale d2h-size-estimate comment updated,
old per-graph predictions flagged stale/unverified against the new
formula).

**Reasoning**: the buffer recap this entry is named for (prompted by the
user's "i expected more free space" reaction to #21's sparse `parent_values`
redesign alone) found that sparsification never touched the actual
biggest cost: the relay's five `blk`-sized wire buffers
(`parent_relay_result`/`parent_send_values`/`parent_recv_values`/
`parent_send_indices`/`parent_recv_indices`), together **~22.4KB, roughly
half of the whole 47,120B per-tile footprint at RMAT s20 750x750**. These
buffers are `blk`-sized (not `max_local_nnz_rows`-sized) by necessity --
#20's fix already established the relay's middle-root (`MID = P/2`)
cross-PE union is genuinely bounded by `blk`, and real runs routinely
approach that bound (64-82% block occupancy measured), so shrinking them
was never safe. Explicit user decision: stop aggregating on-device at
all, ship each PE's own compact array to the host and combine there --
matching how this exact kernel worked before an on-device relay ("Phase
B") was ever added -- accepting slower d2h transfer since memory capacity
remains the stated priority, not raw speed.

**Host-side combine semantics**: unchanged from every prior on-device
tie-break convention in this kernel -- multiple `(py,px)` can legally
claim the same global row (different column ranges can both have a real
edge into it), and any valid candidate winning is correct, so a plain
numpy scatter-assign (last write wins in whatever order) needs no
additional tie-break logic.

**Verification**: compile-only smoke (RMAT s12 8x8) clean via both raw
`cslc` and `run_bfs.py --compile-only`; full correctness re-run same
scale, sources {0,5,50}: `0/4096` mismatches, `0` invalid device parents,
scipy cross-check OK at every source -- identical to #22/#23's own
post-fix numbers, confirming the host-side combine introduces no
regression. Mid-scale `cslc`+`cs_readelf -m` peak-per-tile measurement,
compared directly against this session's own recorded baselines:

- RMAT s19 512x512 (`blk=1024`): **40,736B -> 16,384B** (-24,352B, -59.8%)
- RMAT s20 750x750 (`blk=1399`, the config that motivated this change):
  **47,120B -> 16,672B** (-30,448B, -64.6%)

Both drops exceed the ~22.4KB relay-buffer-only prediction -- confirms a
real additional code-size win from the `collectives_2d/pe.csl` cleanup
(two entire FSM bodies plus their design-comment blocks removed, not just
their buffers). **Real hardware**, RMAT s19 512x512, source 0 (same
config #21 already validated on real WSE-3 silicon): `0/524288` visited
mismatches, `0` invalid device parents, scipy cross-check OK, 6 rounds,
170.6s execution (vs. #21's 291s at the same scale -- not a controlled
comparison, job-to-job cluster variance not accounted for, not claimed as
a speedup). D2H readback span grew as expected with the larger transfer
(now every PE's own `parent_values` array instead of one relay-produced
column) -- `sync-corrected span=601481218` cycles (~687ms @875MHz) for
this run, still trivially far under the 2GiB gRPC ceiling (#1/#4) at
~198MB actual payload for this scale. Same `InconsistentVersion` warning
(client 1.14.0 vs. cluster server 1.20.2) seen again, still benign.

**Known, accepted trade-off**: d2h transfer volume grows ~47x (one
`blk`-sized column -> every PE's own `max_local_nnz_rows`-sized array,
`P*blk*4B` -> `P*P*max_local_nnz_rows*4B`) -- pure transfer-time cost, no
new failure mode, explicitly accepted since memory capacity is the
stated priority for this whole effort. `run_snap_sweep.sh`'s per-graph
d2h-size predictions (berkstan/pokec/etc.) predate this change and have
not been re-derived against the new formula -- flagged stale in that
script's own header comment, not re-measured here.

**Status**: **Fully verified**, smoke through real hardware. Net result
of #21+#24 together against the pre-#21 baseline: RMAT s19 512x512 peak
per-tile memory went 46,032B (pre-#21) -> 40,736B (#21 alone) ->
16,384B (#21+#24), margin against the ~49,152B ceiling grew from
~3.1KB/6.3% to ~32.8KB/67%.

### 25. Host-side parent-combine had no timer at all -- real cost invisible to every measurement

**Where**: `run_bfs.py`/`run_bfs.appliance.py` (new
`host_parent_combine_seconds` wall-clock timer wrapping
`extract_parent_result()`, new CSV column), `plot_bfs_timing.py` (new
"combine" bar in the `ax_d2h` panel), `plot_bfs_timing_poster.py` (new
"combine" bar in both the linear and `--log-scale` modes,
`mean_std_ms_from_seconds` helper, `has_host_combine_column` guard).

**Symptom, caught by the user**: #24 moved the real per-row parent combine
off-device entirely, but `extract_parent_result()` ran strictly AFTER
`end = time.time()` in both run scripts -- so its cost was outside every
existing measurement: not in `search_time_cycles`/GTEPS (device+transfer-
only, correctly so), not in the CSV, not in either plot. The on-device
"parent_resolve" TSC bracket still existed and still got reported, but by
#24 it only brackets a trivial occupancy-count scan -- confirmed later,
on the s22 real-hardware log-scale poster, at a genuinely negligible
0.005ms. Nothing was actually measuring the real combine cost anywhere.

**Fix**: `time.time()` around the `extract_parent_result()` call in both
scripts, printed and added as a new CSV column
(`host_parent_combine_seconds`) -- deliberately a separate unit/column
from every `*_cycles` field (host wall-clock, not a device TSC span), and
deliberately NOT folded into `search_time_cycles`/`gteps` (those stay
device+transfer-only, comparable across runs the same way they always
were). Both plot scripts draw a new "combine" bar (converted to an
equivalent cycle count, `seconds * CLOCK_FREQ_HZ`, purely so it shares the
existing cycle-scale panels -- not a device measurement) only when the
CSV row actually has the column, so older rows render exactly as before
(no false zero implied for data that predates this fix).

**Verification**: confirmed via the s22@750x750 real-hardware run's own
generated poster plots -- log-scale mode shows `resolve=0.005ms` cleanly
separated from a real, nonzero `combine` bar, visually confirming #24's
relay removal actually made on-device resolve free, while the real
(formerly invisible) cost now has its own honest number.

**Status**: **Fixed**. No regression risk (additive: new column, new
optional plot bars, existing behavior for old CSV rows unchanged).

### 26. `parent_values` shrunk to a local u16 column offset (not a global u32 vertex id); real host-side preprocessing memory work; RMAT s25 @ 750x750 still blocked by a host-side (not WSE-3) memory ceiling

**Where**:
- `bool_pe.csl`/`layout_bool.csl`: `parent_values` is now `u16` (was
  `u32`); `PARENT_NONE` is now `65535` (was `4294967295`);
  `compute_bottomup()` stores the local column `c` directly -- the
  on-device `global_c = pcol_id*blk + c` computation this kernel used to
  do at every write is gone entirely.
- `device_io.py`: `extract_parent_result()` now reconstructs each
  candidate's global vertex id itself (`px_idx*blk + parent_values_hwl`,
  the same way it already reconstructed each row's global index from
  `py_idx*blk + mat_row_idx_buf_hwl`); `PARENT_NONE_LOCAL` (65535)
  replaces `PARENT_NONE_GLOBAL`.
- `run_bfs.py`/`run_bfs.appliance.py`: the `parent_values` d2h read
  switched to `data_type=MemcpyDataType.MEMCPY_16BIT` -- but the HOST-side
  numpy buffer stays `np.uint32` (a real, easy-to-miss SDK requirement:
  passing a `uint16` host buffer throws `RuntimeError: Internal data type
  of any memcpy_d2h()/memcpy_h2d() operation should be 32 bit` at runtime,
  even though `data_type=MEMCPY_16BIT` is what actually controls the
  DEVICE-side transfer width; this codebase's own `rounds_completed`/
  `mat_row_idx_buf`/`ts_buf`/`nf_history` transfers already followed this
  convention -- only the new `parent_values` read initially missed it).
- `preprocess_bool.py` (host-side memory fixes, motivated by an RMAT-s25
  OOM -- see below): removed a genuinely dead array (`col_l_per_nz`,
  computed but never read anywhere); freed 7 other large nnz-sized
  intermediates via explicit `del` as soon as their last real use passed
  (Python locals otherwise live for the WHOLE function frame, not just
  until their last use -- a real, if easy to miss, difference from how
  memory would be freed in a language with proper scoping/lifetime
  analysis); safely downcast `row_b_per_nz`/`col_b_per_nz` to `int32` and
  `row_l_per_nz` to `uint16` (both bounded by `faby`/`fabx`/`by`, all
  already asserted `< uint16::max`). The `int32` downcast was tried once
  and got REVERTED first: `row_b_per_nz.astype(np.int32) * np.int64(ncols)`
  does NOT reliably promote to int64 under numpy 1.25's actual behavior
  (confirmed empirically: `np.array([749], dtype=np.int32) *
  np.int64(4194750)` stays `int32` and silently overflows to a negative
  number) -- real at RMAT-s22 scale and up, caught by this session's own
  regression test before it ever reached real hardware. Fixed correctly
  the second time via an explicit `.astype(np.int64)` on `row_b_per_nz`
  itself at that one call site, not by wrapping the OTHER operand.
  Replaced `np.unique(rowb_col_key, return_inverse=True,
  return_counts=True)` with `np.unique(rowb_col_key, return_counts=True)`
  + a separate `np.searchsorted(unique_rc_key, rowb_col_key)` call
  (`return_inverse=True`'s own implementation builds several MORE
  nnz-sized int64 temporaries internally -- an argsort permutation, a
  sorted copy, a boolean mask, a cumulative rank, and the inverse
  permutation itself -- entirely invisible to any `del` on the Python
  side, since they live and die inside numpy's own C implementation for
  the duration of that one call).
- `graph_loader.py`: `load_graph()` now downcasts `.data` to `uint8` right
  at the source -- structural-only, never read downstream by any caller
  in this repo, but previously carried at `mmread`'s default `float64` (8
  bytes/nonzero for a value nobody looks at) through every later
  `.tocsr()`/`.tocsc()` copy. `run_bfs.py`/`run_bfs.appliance.py`:
  `del A_coo` immediately after building `A_csr` from it, instead of
  holding `A_coo`/`A_csr`/`A_csc` all alive simultaneously (3 full live
  copies of the whole matrix at once, previously).

**Motivation**: an attempt to push this whole effort's memory-capacity
work to RMAT s25 @ 750x750 (`n=33,554,432`, `nnz=1,047,214,494`) hit a
real, reproducible host-side OOM in `preprocess_bool.py` well before ever
reaching the WSE-3 compiler. Every fix above is real and independently
verified, but collectively they did NOT get s25 under the ceiling -- see
Status below.

**Verification**: regression-tested clean at every single step (identical
`max_local_nnz`/`max_local_nnz_rows` at RMAT s19 512x512, s20/s21/s22
750x750 before and after each fix, via a standalone probe script built
this session -- loads a balanced `.mtx`, calls `preprocess()` directly, no
appliance/compile needed). Smoke-scale correctness re-run (RMAT s12 8x8,
sources {0,5,50}) after the kernel-side u16 change: `0/4096` mismatches,
`0` invalid device parents, scipy cross-check OK at every source -- the
host-side global-id reconstruction (`px_idx*blk + local`) is correct.
Real peak-per-tile memory (`cslc`+`cs_readelf -m`, re-measured after this
entry's changes, compared against #24's own numbers):

- RMAT s19 512x512: 16,384B -> **16,192B** (-192B)
- RMAT s20 750x750: 16,672B -> **16,384B** (-288B)
- RMAT s22 750x750: 21,488B -> **20,848B** (-640B)

Each drop matches `2 bytes * max_local_nnz_rows` exactly (96, 92, and 270
respectively) -- confirms the model: this entry's on-device change is a
clean halving of `parent_values` alone, nothing else moved.

**RMAT s25 @ 750x750 -- generated and balanced clean, but blocked on a
host-side ceiling, not the WSE-3 kernel**: `util/analyze` balanced the
raw 18.7GB matrix to a 20.2GB output cleanly (~107 minutes real time).
Host-side `preprocess()` repeatedly hit the **100GiB per-user cgroup
memory limit** enforced on `cer-usn-01`/`02`/`03` (confirmed identical via
`cat /sys/fs/cgroup/memory/user.slice/user-<uid>.slice/memory.limit_in_bytes`
== `107374182400` on all three -- a cluster-wide policy, not local to one
node, and not something to work around by switching hosts) -- this is a
**host-side Python/numpy ceiling, structurally unrelated to the WSE-3
per-PE static-memory ceiling** every other entry in this file targets.
Instrumented checkpoint-by-checkpoint measurement (peak RSS via
`resource.getrusage`, re-run after every fix above) isolated the real
breakdown: after all the fixes in this entry, the graph-loading/
conversion pipeline (`mmread` + `.tocsr()` + `.tocsc()`) stays flat at
**16.63GB** for s25 (matches a linear extrapolation from s22's own
measured 2.11GB at 8.16x less `nnz`) -- `preprocess()` itself then
consumes the remaining ~83GB of budget and gets OOM-killed before
completing, every single time this was tried (5 attempts across this
entry's fixes, each regression-tested clean at smaller scales first). The
likely irreducible-without-a-rewrite cost is `preprocess()`'s own TWO
full-`nnz`-element sorts (one `np.unique()` for the CSC-ordered dedup, one
for the CSR-ordered one) -- each needs several more nnz-sized int64
temporaries alive simultaneously inside numpy's own C implementation,
invisible to any further Python-level `del`.

**Lead for next session, not yet implemented or tested**: tracing what
each of the two sorts actually feeds shows the CSR-ordered one computes
ONLY `local_nzrows` (distinct-row count per PE tile) -- nothing else
downstream depends on it. That count looks derivable from data the
CSC-ordered pass ALREADY has in memory (`row_b_per_nz`/`col_b_per_nz`/
`row_l_per_nz`, or equivalently a combined key
`block_id_per_nz * by + row_l_per_nz`) via one more, much cheaper
`np.unique()` call -- eliminating the second full sort AND the second
scipy sparse representation (`A_csc`) as a caller-side input entirely,
if it holds up under implementation + the same regression tests already
in place. See `bfs/HANDOFF.md`'s own "Next session" section for the full
writeup.

**Status**: **Partial**. The u16 local-column change and the host-side
memory fixes above are real, verified, and landed (not reverted) --
RMAT s19-s22 @ their respective grids all still compile and run clean at
the new, lower peak-memory numbers. RMAT s25 @ 750x750 itself remains
unreached, blocked by a host-side (not on-device) memory ceiling, pending
either the CSR/CSC-redundancy lead above or a more invasive
chunked/blocked rewrite of `preprocess()`.

### 27. CSR/CSC redundancy eliminated from `preprocess_bool.py`; RMAT s25 @ 750x750 gets past the host-side wall, but hits a genuine, separate WSE-3 PE-memory ceiling

**Where**: `preprocess_bool.py` (`preprocess()`'s `csrRowPtr`/`csrColInd`
parameter pair removed entirely, along with the whole CSR-ordered
`np.unique()` pass it fed -- `local_nzrows`/`max_local_nnz_rows` no
longer computed or returned in `matrix_info` at all), `run_bfs.py`/
`run_bfs.appliance.py` (no longer build `A_csc` at all -- `A_csr`'s
arrays alone are passed into `preprocess()`'s remaining `cscColPtr`/
`cscRowInd` slot), `.claude_scratch/probe_preprocess.py`/
`probe_preprocess_instrumented.py` (matching updates).

**The lead from #26's own "next session" writeup, now implemented and
verified**: tracing every live caller (`run_bfs.py`, `run_bfs.appliance.py`
-- `run_graph500.py` is a separate, already-broken caller predating #21's
bottom-up swap, explicitly out of scope per #21/#24's own precedent)
showed NEITHER ever reads `matrix_info["local_nnz_rows"]`/
`["max_local_nnz_rows"]` -- both only ever consume `["local_nnz_cols"]`/
`["max_local_nnz_cols"]` (renamed locally to `"*_rows"` post-#21's
argument swap; each caller's own comment already said as much). The
entire CSR-ordered computation -- a full second `nnz`-element sort, fed
by a whole second scipy sparse representation the caller had to build --
was **provably dead code** for every live path. Not just cheapened (a
derivation from the CSC-ordered pass's own data was drafted and would
have worked, see git history), but deleted outright: zero cost beats any
cost, and one less parameter pair for callers to worry about getting
right (this codebase already has a history of subtle CSR/CSC swap bugs,
#22).

**Verification**: regression-tested clean at every step (identical
`max_local_nnz`/`max_local_nnz_rows` at RMAT s19/s20/s21/s22 before and
after). Smoke-scale correctness re-run (RMAT s12 8x8, sources {0,5,50}):
`0/4096` mismatches, `0` invalid device parents, scipy cross-check OK at
every source -- confirms the signature change and dead-output removal
introduced no regression. Instrumented peak-RSS re-measurement at RMAT
s22 750x750: total dropped from 15.18GB to **11.78GB** (`preprocess()`
itself: 13.07GB -> 9.67GB, ~26% cheaper) -- a real win at every scale,
not just s25.

**RMAT s25 @ 750x750 -- the host-side wall is cleared, but a NEW, genuine
WSE-3 wall is hit immediately after**: with this fix, `preprocess()`
completed successfully for the first time at this scale -- peak RSS
**86.29GB**, comfortably under the 100GiB per-user cgroup ceiling (#26).
`max_local_nnz=2078`, `max_local_nnz_rows=1293`, `blk=44740`. But the
actual `cslc` compile then failed with a REAL, different error:
```
ld.lld: error: ran out of PE memory for data (section .bss)
ld.lld: error: ran out of PE memory for task table
ld.lld: error: ran out of PE memory for data (section .data.hi)
```
(the familiar "linker file-vanished" flake, #12, also appeared
repeatedly in the same log, almost certainly a secondary symptom once
the primary `ran out of PE memory` failures start cascading through the
linker's worker pool -- not evidence this is actually a flake rather
than a real overflow). This is the genuine **WSE-3 per-PE static-memory
ceiling** every other entry in this file targets, structurally distinct
from #26's host-side cgroup wall -- fixing one did not, and could not,
fix the other. At RMAT s25's `max_local_nnz`/`max_local_nnz_rows`/`blk`
values, the per-tile data this kernel needs genuinely exceeds the
~49,152B ceiling at a 750x750 grid (WSE-3's practical max grid size,
per `rmat_grid_sweep.sh`'s own header comment -- there is no bigger grid
to fall back to for this exact matrix).

**Also newly surfaced, not yet reached**: even if the PE-memory ceiling
were somehow cleared, the `parent_values` d2h readback at this scale
(`750*750*1293*4` bytes, confirmed via `device_io.py`'s own documented
invariant that "the SDK's real wire payload is always 4 bytes/element...
regardless of the `data_type` kwarg") is **~2.71GiB -- over the
2,147,482,624-byte hard gRPC message-size ceiling** this project already
hit once on the h2d side (#1/#4, fixed there via
`prepare_h2d_chunked`/`send_h2d_chunked`). No equivalent d2h-side
chunking exists in this codebase yet -- would need implementing before
any RMAT s25-scale real-hardware run could work, independent of the
PE-memory question above.

**Status**: **Done** (the CSR/CSC-redundancy fix itself: verified,
landed, a real improvement at every scale tested). RMAT s25 @ 750x750
remains **blocked**, now by a different, harder, and more fundamental
wall than #26 left it at -- genuine WSE-3 PE-memory overflow at this
scale's `max_local_nnz`/`max_local_nnz_rows`/`blk` values, with a second,
independent d2h gRPC-ceiling problem waiting behind it. Neither is fixed
by anything in this entry. See `bfs/HANDOFF.md`'s "Next session" section
for the current recommendation (fall back to RMAT s23/s24 as the next
real-hardware high-water mark instead of continuing to chase s25).

### 28. RMAT s23/s24 real-hardware verified as new high-water marks; scipy cross-check's own memory cost discovered and auto-disabled above RMAT-s20 scale

**Where**: `run_bfs.py`/`run_bfs.appliance.py` (new `_SCIPY_CHECK_AUTO_DISABLE_N`
module constant, new `--force-scipy` flag, auto-disable check inserted
right after `n` is known).

**RMAT s23/s24 @ 750x750 -- both compile and run clean on real WSE-3
hardware**, following #27's fix:

- **s23** (`blk=11,185`, `max_local_nnz=567`, `max_local_nnz_rows=451`):
  **26,320B** peak per-tile (46% margin). Real hardware: `0/8,388,750`
  mismatches, `0` invalid parents, scipy cross-check OK, 6 rounds,
  **224.5 GTEPS** (excl. transfer).
- **s24** (`blk=22,370`, `max_local_nnz=1075`, `max_local_nnz_rows=755`):
  **36,768B** peak per-tile (25% margin). Real hardware: 6 rounds,
  **273.99 GTEPS** (excl. transfer) -- new high. Correctness NOT
  independently confirmed at this scale (see below) -- compile and
  on-device execution are real and verified; the scipy reference check
  itself is what's missing here, not evidence against correctness.

Both are genuine new real-hardware high-water marks for this whole
memory-capacity effort (previous: RMAT s22, 20,848B, 164 GTEPS).

**New problem found while verifying s24**: the real-hardware run got
**OOM-killed** (`dmesg`-confirmed: `anon-rss:104639284kB`, right at the
100GiB per-user cgroup ceiling) with the process printing **nothing at
all** -- not even `"Run done in Xs"`, which always appears before the
scipy cross-check starts in every prior successful run. This pointed
directly at the scipy correctness-check code (`breadth_first_order()` +
`A_fwd = A_csr.transpose().tocsr()`, a second full CSR copy on top of
the one already held), NOT `preprocess()` (which succeeds fine at this
scale on its own, confirmed by the local compile-only succeeding) and
NOT the on-device execution (confirmed by a `--nocorrectness` re-run
completing cleanly). A **third, distinct host-memory wall** from #26's
`preprocess()` one and #27's on-device one -- this one specific to the
*correctness-verification* code path.

**Fix**: `run_bfs.py`/`run_bfs.appliance.py` now auto-disable the scipy
cross-check (both the printed correctness summary and the tree plot's
own scipy dependency) once the matrix exceeds
`_SCIPY_CHECK_AUTO_DISABLE_N = 1,100,000` vertices -- RMAT s20's own `n`
(1,049,250) is the largest scale this project has verified the scipy
check itself against without incident, so that's the threshold, with
headroom. A new `--force-scipy` flag overrides it for anyone who's
checked host memory headroom themselves. The auto-disable only fires
when `--notree` is also set (tree-plot rendering unconditionally expects
scipy data further downstream; silently passing it `None` there would be
worse than just respecting an explicit tree-plot request at large scale).
Verified: smoke-scale (RMAT s12 8x8, well under threshold) behavior is
byte-for-byte unchanged (`0/4096` mismatches, scipy check still runs);
RMAT s24 real hardware with **no flags at all** now prints the auto-disable
NOTE and completes cleanly, matching the explicit `--nocorrectness`
run's own numbers exactly (`273.99` GTEPS both times).

**Status**: **Done**. RMAT s23 is fully scipy-verified on real hardware;
RMAT s24 is compile+execution-verified on real hardware, with the scipy
cross-check itself now understood to need a memory-cost fix of its own
(not yet done -- would need the same kind of treatment #26/#27 gave
`preprocess()`, e.g. freeing `A_csr` before/while building `A_fwd`, or a
lighter validation approach) before it can safely run at this scale.
Every scale above the new threshold auto-skips it by default now, rather
than risking a silent OOM.

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
| 19 | Indexed's dense send-side parent buffers cost more than needed; also a real intra-round duplicate-append bug found+fixed during this change's own development | RMAT s8 4x4 (smoke-test scale; larger-grid A/B not yet run) | `parent_local_buf`/`parent_send_indices`/`parent_send_values` sized by `blk` when only `max_local_nnz_rows` slots can ever be used; dev bug: `visited_bitmap` alone doesn't dedup two hits on the same row within one round's `compute_topdown()` | **Fixed**, memory numbers revised by #20 (compacted `parent_compact_indices/values` + `reduce_select_any_indexed_precompacted`, `variant==2` only; dev bug fixed via new per-round `parent_round_seen_bitmap`; -768B/-2.8% measured at smoke-test scale predates #20's fix, no longer accurate; does not clear #8's blk=1024 ceiling) |
| 20 | `reduce_select_any_indexed_precompacted()` send-side aliasing overflow — real crash, not silent corruption | RMAT s12 8x8, sources {0,1}, free simulator — first mid-scale test of #19's own follow-up list | send-side scratch aliased directly to `max_local_nnz_rows`-sized compact storage, but the relay's middle-root branch reuses it as a landing zone for a multi-PE union bounded by `blk`, not `max_local_nnz_rows` | **Fixed** (restored `blk`-sized `parent_send_indices/values`, copied from compact storage per call; re-verified 0 mismatches, 2 sources; narrows #19's projected memory saving, see there) |
| 21 | Bottom-up-only BFS: top-down + on-device CSC→CSR transpose removed; `parent_resolve_variant==2`'s storage redefined to `parent_values` | RMAT s12 8x8 (smoke) through s19 512x512/blk=1024 (mid-scale, real WSE-3 hardware) | memory-capacity priority: bottom-up's single linear per-round pass makes #19/#20's discovery-order dedup scheme unnecessary; host now uploads the matrix pre-transposed instead of an on-device transpose | **Fully verified** (smoke through real hardware; 40,736B peak per-tile at s19 512x512, down from 46,032B pre-#21; 0 mismatches on real hardware) |
| 22 | `preprocess()` call-site swap silently transposed the (px,py) PE-grid axes | RMAT s12 8x8, sources {0,5,50} — #21's own first correctness run | `preprocess()`'s block-placement formula treats the CSR-role-fed data's own row_b as the array's first output axis; under #21's swap that ends up holding the column-block index instead, transposing every off-diagonal PE's data with its transpose partner (diagonal PEs unaffected, masking it as a shape-safe compile) | **Fixed** (transpose returned arrays' first two axes post-extraction; 0/4096 mismatches at every source after) |
| 23 | Dense (`parent_resolve_variant==0`) removed entirely | RMAT s12 8x8, sources {0,5,50} | no longer needed once #21 made bottom-up the only strategy and #22 confirmed indexed/sparse fully correct; kept only as an unused fallback until now | **Done** (single parent-resolution path; `--parent-resolve-variant` no longer exists; 0/4096 mismatches, no regression) |
| 24 | On-device parent-aggregation relay (`reduce_select_any_indexed`) + dense `reduce_select_any` both removed; combine moved host-side | RMAT s12 8x8 (smoke) through s19 512x512/blk=1024 (real WSE-3 hardware); s20 750x750/blk=1399 (mid-scale memory) | relay's 5 `blk`-sized wire buffers (~22.4KB) were the actual dominant static-memory cost, untouched by #21's sparse `parent_values` alone, and structurally couldn't shrink (union bound by `blk`, not `max_local_nnz_rows`, per #20) | **Fully verified** (smoke through real hardware; 40,736B->16,384B at s19 512x512, 47,120B->16,672B at s20 750x750, both ~60-65% peak-memory reduction; 0 mismatches on real hardware; ~47x d2h volume increase accepted) |
| 25 | Host-side parent-combine had no timer at all | s22 750x750, real hardware (poster plots) | #24 moved the real combine off-device, but `extract_parent_result()` ran after every existing timer stopped -- invisible to search_time_cycles/GTEPS/CSV/plots | **Fixed** (new `host_parent_combine_seconds` wall-clock column + poster/detail plot bars, additive/backward-compatible) |
| 26 | `parent_values` shrunk to local u16 column offset (was global u32 vertex id); real host-side `preprocess()` memory fixes; RMAT s25 @ 750x750 still blocked | RMAT s12 8x8 (smoke) through s19-s22 @ their own grids (real peak-memory re-measurement); RMAT s25 @ 750x750 (host-side OOM, not reached) | on-device global-id computation was pure host-side-derivable waste once #24 moved the combine off-device; separately, `preprocess_bool.py`/`graph_loader.py` held many more nnz-sized temporaries/copies alive than needed, hitting a 100GiB per-user cgroup ceiling (host-side, NOT the WSE-3 per-PE ceiling) at RMAT-s25 scale | **Partial** (u16 change + memory fixes landed/verified, 16,384B->16,192B at s19, 16,672B->16,384B at s20, 21,488B->20,848B at s22; RMAT s25 still blocked -- CSR/CSC-redundancy lead flagged for next session, not yet implemented) |
| 27 | CSR/CSC redundancy eliminated from `preprocess_bool.py` (`local_nzrows`/`max_local_nnz_rows` was dead code for every live caller); RMAT s25 @ 750x750 clears the host-side wall but hits a genuine WSE-3 PE-memory overflow | RMAT s19-s22 @ their own grids (regression + real memory re-measurement, ~26% cheaper preprocess() at every scale); RMAT s25 @ 750x750 (preprocess() succeeds at 86.29GB RSS; `cslc` fails with real `ran out of PE memory`) | a whole second full-nnz sort + second scipy sparse representation was computing an output neither live caller ever read; separately, RMAT s25's own `max_local_nnz`/`max_local_nnz_rows`/`blk` values genuinely exceed the ~49,152B WSE-3 per-PE ceiling at the largest available (750x750) grid | **Done** (CSR/CSC fix itself, verified/landed); RMAT s25 **still blocked** -- now by a real, harder, on-device ceiling (not host-side), plus an unaddressed ~2.71GiB d2h payload over the 2GiB gRPC ceiling waiting behind it |
| 28 | RMAT s23/s24 verified as new real-hardware high-water marks; a THIRD host-memory wall found in the scipy correctness-check code itself (distinct from #26's `preprocess()` wall and #27's on-device wall) | RMAT s23 @ 750x750 (26,320B, 0/8,388,750 mismatches, 224.5 GTEPS); RMAT s24 @ 750x750 (36,768B, 273.99 GTEPS, OOM-killed at ~99.8GiB anon-rss when the scipy check ran) | `breadth_first_order()` + rebuilding a transposed CSR copy of A is itself memory-hungry at large `n`, independent of `preprocess()`/on-device costs -- undiscovered until RMAT s24 was the first scale big enough to hit it | **Done** (scipy cross-check auto-disabled above RMAT-s20 scale by default, `--force-scipy` to override; verified byte-identical smoke-scale behavior and a clean RMAT s24 real-hardware run with no flags at all) |
