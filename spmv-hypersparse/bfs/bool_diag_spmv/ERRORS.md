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
confirmed byte-for-byte unchanged single-call behavior.

### 4b. `h2d_matrix` cycle-count stat is garbage — separate, pre-existing bug (open)
While cross-checking timing after the #4 fix (real appliance runs of
berkstan, s18, s20 @ 750x750, host-side `time` vs. the script's own printed
stats), the `h2d_matrix: min=/max=/avg=` line printed a nonsense cycle count
in *all three* runs (29.5B, 72.2B, and 595B cycles respectively — none
plausible at 875MHz). Critically, **s18 and s20 never touch the new
chunking path at all** (single-call, `max_local_nnz` far below the
threshold) and still show garbage — proving this is a pre-existing bug in
the `h2d_matrix` timing readout itself (most likely a 32-bit on-device
cycle-counter wraparound not handled by the delta computation), unrelated
to chunking, just never noticed before because every prior large-graph run
either failed outright (this same #4 ceiling) or was small enough to finish
before anyone looked closely at this one diagnostic line. **Does not affect
correctness**: `search_time_cycles`/GTEPS use a different, confirmed-correct
mechanism (`round_trip_start_buffer`/`round_trip_done_buffer`), and the
script's own `WARNING` already notes `total_runtime_cycles`/`GTEPS` "remain
CORRECT regardless." Old CSV rows/plots for berkstan/s18/s20 @ 750x750 that
carried this garbage value were replaced with fresh reruns (cleanup, not a
fix). **Status**: open, cosmetic/diagnostic-only, not chased further this
session.

### 5. Blanket symmetrization of SNAP graphs fabricated edges
**Where**: v2 of the SNAP pipeline (before this fix), all 5 graphs.
**Cause**: `benchmarks/snap_to_mtx.py` unconditionally computed `A | A^T`
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
the v3 directedness fix.
**Cause**: `blk = ceil(n/P)` grows with `n` at fixed `P=750`; each PE's
static-memory footprint (bitmaps, buffers, task table) scales with `blk`.
topcats (n≈1.79M) and livejournal (n≈4.85M) simply need more per-PE static
memory than a 750x750 grid provides — a real, hardware-imposed compile-time
ceiling, confirmed independent of the symmetrization bug (both graphs
failed identically before and after that fix). Would need a larger PE grid
or a smaller per-PE working set to lift.
**Status**: real, open, not attempted to fix this session (would require
either more PEs or reducing static memory per PE, e.g. narrower bitmaps).

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

## Summary table

| # | Error | Where confirmed | Cause | Status |
|---|-------|-----------------|-------|--------|
| 1 | d2h gRPC 2GiB ceiling | RMAT s20 | design flaw (P copies transferred) | **Fixed** (`reduce_select_any`) |
| 2 | PE mem overflow (new collective) | pokec/topcats | 3-buffer design | **Fixed** (2-buffer redesign) |
| 3 | task id collision | compile-time | id 21 not actually free | **Fixed** (moved to 24) |
| 4 | h2d gRPC 2GiB ceiling | berkstan (verified fixed), orkut | `max_local_nnz` skew + vendor SDK chunker envelope-overflow bug | **Fixed** (`memcpy_h2d_chunked`) |
| 4b | `h2d_matrix` stat garbage | berkstan/s18/s20 @ 750x750 | likely 32-bit cycle-counter wraparound in that stat's readout | **Open** (cosmetic; GTEPS unaffected) |
| 5 | fabricated symmetrization | v2 SNAP pipeline | wrong default (`A\|A^T` for directed graphs) | **Fixed** (opt-in `--symmetrize`) |
| 6 | scrambled source vertex | any SNAP run | `--rand 0` doesn't disable base permutation | **Fixed** (`--operm`) |
| 7 | raw/unbalanced SNAP fails | user experiment | real degree skew, no balancing | N/A (balancing required) |
| 8 | PE static-mem ceiling | topcats, livejournal | `blk` too large for 750x750 grid | **Open** (needs bigger grid) |
| 9 | plot crash on truncation | any `--max-rounds` run | wrong length assumption | **Fixed** |
| 10 | silently wrong GTEPS | any `--max-rounds` run | truncated `ts_buf` history | **Fixed** (round-trip markers) |
| 11 | slow "compile" | large SNAP graphs | unvectorized Python preprocessing | **Fixed** (vectorized) |
| 12 | linker file-vanished flake | s21 (4/4), livejournal (1x), orkut (1/1) | compile-farm scratch/container lifecycle (probable), correlates with the 3 largest jobs in the suite | **Open**, not retried further |
| 13 | transient 503 upload error | pokec (1st attempt) | connectivity flake (probable) | Resolved on retry |
| 14 | "failed to terminate linker workers" | topcats | secondary message alongside #8 | Not independent |
