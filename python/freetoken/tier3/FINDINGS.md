# Phase A findings — NVMe expert tier (Qwen3.8-Flash-Next)

Measured 2026-09-02 on the 5090 box, **without the GPU** (replay of the existing
`/media/nuko/New Volume/route-dumps-m1` dumps: 6 chunks, 9,952 tokens, 48 layers, top-10).
Reproduce with `scripts/tier3-compact-routes.py` → `scripts/tier3-sim.py` /
`scripts/tier3-bench-nvme.py`.

## Verdict: GO, with a large margin

The success bar was 5–10 tok/s. The disk term at the planned geometry costs **~9–15 ms/step**,
against a ~12.5 ms base step — i.e. the tier lands in the **80–130 tok/s** region, not single
digits. Two of the plan's assumptions were wrong in our favour, and one design branch is
cancelled outright.

## G1 — disk law: **PASS**

Sweep at the 16 GiB-card operating point (`--gpu-slots 3000`, 48 layers, 22.54 distinct active
experts/layer/step):

| host slots/layer | residency | RAM hit | reads/step | MB/step |
|---|---|---|---|---|
| 128 | 25.0 % | 48.5 % | 160.7 | 444.2 |
| 256 | 50.0 % | 87.7 % | 38.4 | 106.2 |
| 320 | 62.5 % | 93.8 % | 19.4 | 53.7 |
| **356** | **69.5 %** | **95.6 %** | **13.6** | **37.6** |
| 448 | 87.5 % | 98.0 % | 6.1 | 17.0 |

356 slots/layer ≈ the 44 GiB pool the 64 GiB budget affords. A short trace charges cold-start
to every geometry: at full residency (512, nothing ever evicted) the sim still reports
**5.00 reads/step**, which is pure compulsory miss. **Steady-state capacity misses at 356
slots are therefore ~8.6/step = 23.8 MB/step** — the 37.6 figure is conservative.

**`MB/step` is essentially independent of GPU slot count** (37.6 at 1650, 2500, 3000 and 3500
slots). The RAM tier absorbs the VRAM squeeze, so the tight 16 GiB budget does *not* propagate
to disk traffic. This decouples the two halves of the profile.

## G2 — skew: **PASS, and it cancels the policy work**

At 69.5 % residency the RAM tier hits **95.6 %**, not the 69.5 % that uniform routing would
give. Routing is strongly skewed; the pessimistic corner of the plan does not apply.

More usefully: **LRU is already within ~1.5 points of the frequency oracle** (95.64 % vs
97.2 % at 356 slots; 93.78 vs 95.2 at 320), and at 200 slots LRU *beats* the static oracle
(84.0 vs 82.2) because recency carries information frequency does not. **Ship plain LRU and
delete the hot-set / LFU / TinyLFU branch (plan B7).** There is under 2 points on the table and
a static profile would cost a calibration pass, a serialized artifact and a staleness failure
mode to chase it.

## G3 — queue depth: **SOFT PASS** (the one place read-ahead still pays)

p50 reads issuable within an 8-layer window at the operating point is **1.0** — the warm cache
misses ~0.28 experts per layer, so without read-ahead the reader runs at **QD1**.

That is survivable only because the request is large. Measured on the NTFS3 volume at the true
tier-3 shape (2,764,800 B aligned O_DIRECT reads = one expert record):

| QD | GB/s | p50 ms | p99 ms |
|---|---|---|---|
| **1** | **2.58** | 1.031 | 1.543 |
| 2 | 4.28 | 1.196 | 2.860 |
| 4 | 5.51 | 1.854 | 5.913 |
| **8** | **5.90** | 3.306 | 14.289 |
| 16 | 5.38 | 6.475 | 31.905 |
| 32 | 4.50 | 12.756 | 59.384 |

So the disk term is **13.6 × 1.03 ms ≈ 14 ms/step at QD1**, or **~6.8 ms at QD4–8**.
Read-ahead (plan B6) is worth ~2.3× *on the disk term only* — real, but an optimisation, not
the correctness-of-performance requirement the plan assumed. Note QD ≥ 16 is
counter-productive; cap the reader at 8 in flight.

## G4 — prefill / TTFT: **PASS at 8 K, quantified beyond**

`_prefill_routed` materialises every expert of every layer per forward, and prefill deliberately
does not admit into the LRU, so each chunk re-reads the non-resident **30.5 % of the corpus =
19.3 GiB**. These are whole-layer sequential reads, so they run at the ~5.9 GB/s end:

| prompt | chunks @ 8192 | disk bytes | added TTFT @ 5.9 GB/s |
|---|---|---|---|
| 8 K | 1 | 19.3 GiB | **~3.3 s** |
| 32 K | 4 | 77 GiB | ~13 s |
| 128 K | 16 | 309 GiB | ~52 s |

Acceptable at short context; at long context it roughly doubles an already-long TTFT. Keep the
prefill chunk at the measured 8192 maximum — halving it doubles this cost.

## G5 — stationarity: **PASS**

Same geometry over disjoint windows of the stream (gpu 3000 / host 356):

| window | RAM hit | reads/step |
|---|---|---|
| first half | 95.3 % | 15.3 |
| second half | 94.2 % | 17.6 |
| Q1 / Q2 / Q3 / Q4 | 93.5 / 93.7 / 91.9 / 92.8 % | 20.9 / 20.4 / 24.5 / 21.6 |

Stable to ~3 points, and shorter windows look *worse* (more cold-start weight), so the
full-trace number is the conservative one.

## Corrections to the plan

1. **NTFS3 is not the bottleneck the plan assumed.** The earlier 2.4 GB/s figure was a `dd`
   artifact (process spawn + small blocks). At expert granularity NTFS3 delivers 2.58 GB/s at
   QD1 and 5.90 GB/s at QD8. The ext4 migration (risk #4) drops from "largest single lever" to
   "not currently worth doing".
2. **Routing is skewed, not uniform.** The 11–15 tok/s pessimistic corner is off the table.
3. **Cancel B7's policy work** — LRU ≈ oracle.
4. **Demote B6 (read-ahead)** from mandatory to a ~2.3×-on-the-disk-term optimisation, and cap
   the reader at 8 in-flight reads.

## Caveats

- 9,952 tokens is a short trace; pools never fully warm, which biases every number
  **pessimistic** (the 5.0 reads/step compulsory floor). A longer dump would sharpen it, but
  needs the GPU.
- The dumps are teacher-forced prefill routing, which is the same distribution decode sees over
  the same text, but they are *these six texts*: prose, code, Japanese, replayed generation.
- The sim models 48 main-tower layers. The 49th (MTP draft) bank layer adds residency but is
  not routed in these dumps.

## Bank-split decision (measured, not assumed)

`host_bank_split` pages a bank only if it is ≥ 256 KiB *and* a whole number of 4096-B blocks.
For Qwen NVFP4 that selects only the two **packed** banks (2,457,600 B = 600 blocks, 88.6 % of
the expert); the scale banks are aligned but under the threshold, so they stay resident
together with the tiny unaligned `*_global` banks — **314,880 B/expert = 7.2 GiB** across the
model.

7.2 GiB of a 64 GiB box is not nothing, so the alternative was simulated rather than argued,
at a fixed 52 GiB pool budget:

| config | resident | slots/layer | residency | reads/step | MB/step | **IOs/step** | QD1 ms |
|---|---|---|---|---|---|---|---|
| **page packed only** | 7.21 GiB | 407 | 79.5 % | 8.54 | 21.0 | **17.1** | **9.84** |
| page packed + scales | 0.18 GiB | 419 | 81.8 % | 7.68 | 21.2 | 30.7 | 11.30 |

Paging the scales buys 2.3 points of residency and no reduction in bytes, while nearly
doubling the request count — and at QD1 the request count is what the step pays for. **Keep the
scales resident.** Their 7.2 GiB also buys them a 100 % hit rate, which is why the byte traffic
does not move.

## Tier-3 reader, measured end to end (A5)

FTW conversion of Qwen3.8-Flash-Next: **72.70 GiB, 10 shards, 288 expert-bank tensors**
(48 layers × 6 banks, per-layer entries), `quant_format: nvfp4`, in **67 s**.

`NvmeExpertStore` addressing was validated against the **production `FTWReader`** on that
checkpoint — 24 expert rows across 3 layers and both paged banks, **byte-identical, zero bounce
reads**. Paged geometry confirmed on real data: `gate_up_packed` 1,638,400 B +
`down_packed` 819,200 B = **2,457,600 B/expert**.

Random `(layer, expert)` reads over the full 72 GiB file, per-row issue:

| readers | GB/s | ms/expert | ms/step @ 13.6 reads | est. tok/s (12.5 ms base, 2.4 acc/step) |
|---|---|---|---|---|
| 1 | 0.99 | 2.476 | 33.7 | 52 |
| 4 | 2.74 | 0.897 | 12.2 | 97 |
| 8 | 4.19 | 0.587 | 8.0 | 117 |
| **16** | **5.61** | **0.438** | **5.9** | **130** |
| 24 | 6.31 | 0.389 | 5.3 | 135 |
| 32 | 5.80 | 0.424 | 5.8 | 131 |

Broad knee at 16–24 readers, turning over by 32. `DEFAULT_READERS = 16`.

Note the QD1 figure (0.99 GB/s) is *worse* than the 2.58 GB/s the single-block probe gave,
because one expert is two reads: at one reader they serialise. Issuing per **(expert, bank)
row** rather than per expert makes in-flight depth equal `readers` regardless of bank count —
throughput-neutral in measurement (4.69 vs 4.19 GB/s at 8 readers, run-to-run noise), kept for
the predictable contract rather than for speed.

**This is the whole design validated on real bytes**: ~6 ms/step of disk against a ~12.5 ms
base step, i.e. the tier costs roughly a third of a step, not the 10× the pessimistic corner
feared.

## B1 — the VRAM half, measured with no new code

`scripts/serve-qwen38.sh` with `FREETOKEN_GPU_RESERVE_MB=14500`,
`FREETOKEN_KV_RESERVE_TOKENS=16384`, `FREETOKEN_KV_MAX_TOKENS=32768`. Free memory before load
was 30.10 GiB, and the reserve is subtracted after the ratio, so this emulates a
**~14.4 GiB card** (`X = 30.10 - R/0.9`) — slightly *tighter* than the 16 GiB target. For an
exact 16 GiB emulation use `R = 0.9 x (30.10 - 16) = 12.7 GiB ~ 12995 MB`.

Resolved geometry: **`moe_cache_size=2821`, `num_pages=257`, `prefill_overlap=True`**, KV
0.22 GiB at 16,448 tokens.

| task | decode tok/s (3 runs) |
|---|---|
| bench | **116.2** |
| english | **109.5** |
| code | **137.6** |
| japanese | **111.0** |

Two corrections to the plan:

1. **Prefill overlap fits.** The plan assumed the `2*num_experts` = 1,024-slot floor would force
   `--disable-moe-prefill-overlap` at 16 GiB. At 2,821 slots it is comfortably satisfied and the
   planner kept it on. The A6 degradation fix still matters, but as a guard for smaller budgets
   rather than for this profile.
2. **49 bank layers confirmed** (`CPU MoE executor ready: ... experts=512 layers=49`) — 48 main
   tower + 1 MTP draft layer, i.e. 25,088 experts, not the 24,576 first assumed.

The step is ~20.7 ms here (2.4 accepted/step at 116 tok/s). Tier 3 adds ~5.9 ms of disk at 16
readers, so the 16/64 profile should land near **90-110 tok/s** — against a 5-10 tok/s bar and
a "does not boot today" baseline.

## B2 — PLE row cache, components built and validated

`tier3/ple_rows.py`: `read_ple_layout` (parses the 128 PLE shard headers into a row-addressable
layout), `PleDiskReader` (O_DIRECT aligned-window reads, per-thread fds and bounce buffers) and
`PleRowCache` (pinned rows + explicit slot assignment).

Real layout confirmed: **128 shards x 2,500,012 rows x 160 B = 320 M rows = 47.68 GiB**,
`weight_scale = 1.993e-4`.

**Correctness on the real table: 24/24 sampled rows byte-identical to a `safetensors` reference
read**, before and after the parallel refactor; duplicate ids within a call share one slot and
cost one disk read.

Rows are 160 B and never block-aligned, so each read pulls the containing 1-2 block window into
a per-thread bounce and copies 160 B out. That is the right trade here (the alternative --
repacking so rows are block-aligned -- would inflate 47.7 GiB by 25x); it is the opposite of the
expert tier, where the zero-bounce property is worth designing for.

Latency for a 48-row step (16 heads x 3 verify rows), fully cold:

| readers | p50 | p90 |
|---|---|---|
| 1 | 5.40 ms | 5.59 ms |
| **8** | **1.56 ms** | 1.73 ms |
| 16 | 1.60 ms | 1.78 ms |

Serial issue costs a quarter of a step; concurrent issue is 3.5x better. `DEFAULT_READERS = 16`.

On **real token streams** (the route dumps' own `input_ids`, row ids recomputed through the
same hash constants), 3-row verify with stride 2:

| cache | hit rate | misses/step | p50 | p90 |
|---|---|---|---|---|
| 40 MiB | 53.9 % | 22.1 | 0.78 ms | 0.92 ms |
| 160 MiB | 53.9 % | 22.1 | 0.82 ms | 0.94 ms |
| 2.5 GiB | 53.9 % | 22.1 | 0.83 ms | 0.96 ms |

**~0.8 ms/step, about 4 % of a 20 ms step -- to free 47.7 GiB.** The hit rate being *flat* in
cache size says this 9,952-token trace cannot size the cache: it touches at most ~160 K distinct
rows, which even the 40 MiB cache holds, so 53.9 % is the compulsory-miss floor (each distinct
row missed exactly once) and real Zipfian reuse over a long session can only do better. Sizing
the cache needs a longer capture; the *cost* measured here is the worst case.

### Design note: why not direct-mapped

`slot = row_id % C` would let the GPU translate ids itself with no host round trip, but two live
rows of the *same step* can collide, and the second install silently overwrites the first -- the
gather then returns another n-gram's embedding. Wrong tokens, no crash. Instead the host assigns
slots explicitly and guarantees a step's rows are simultaneously resident
(`test_all_rows_of_one_call_stay_simultaneously_resident`), handing the kernel slot indices.
`ple_gather_rows` is unchanged either way -- it only computes `base + idx * head_dim`.

## B3 + B4 (kernel half) — the indirection, shipped and verified

The whole tier-3 mechanism reduces to one substitution: `src_indices` carries the expert's
**host bank row** instead of its expert id. `fast_index_copy.cuh` is **not touched** — it
computes `base[layer, bank] + row * feat_bytes` either way, which is what per-layer host pools
buy.

Seams into existing files (five, each default-off):

| file | change |
|---|---|
| `moe/offload_cache.py` | `host_slots` / `host_tier` fields; `set_bank_sources` accepts `host_slots`-wide banks; `routing_pack` `[2,N]` -> `[3,N]` behind a shared `ROUTING_PACK_ROWS` |
| `moe/cpu_executor.py` | the pinned mirror follows `ROUTING_PACK_ROWS` (it shape-checks against the device pack and **silently** falls back per-row if they disagree — the two must move together) |
| `moe/offload_kernels.py` | `HOST_TIER: tl.constexpr` in the hybrid ensure kernel + its CPU mirror |
| `kernel/csrc/cpu_moe/cpu_moe_ext.cpp` | `row_of(layer, e)` + `set_host_slot_table`; six routing reads remap the id to the host row |
| `engine/cache_budget.py` | the A6 prefill-overlap degradation fix |

Kernel semantics under `HOST_TIER`:

* a miss whose expert is not in the host pool is **not selectable for a PCIe gather** — there
  is nothing to gather from. It keeps slot -1 and falls through to the CPU branch, which is
  where the NVMe read belongs. `num_missing_full` still counts every miss, so the miss-rate
  statistic keeps its old meaning; the bandwidth-matched split is computed over *fetchable*
  misses.
* `src_indices[i]` gets `host_slot_for_id[layer, e]`.
* routing-pack row 2 publishes the host slots under a live gather, so the installer running
  concurrently on the CPU pool cannot recycle one mid-flight.

Verification (`tests/tier3/`, 61 tests):

* GPU kernel vs its CPU mirror over 4 seeds x 48 steps with a partial host pool — slot
  rewrites, `slot_for_id`, `id_of_slot`, `usage`, `src_indices` and `evict_slots` all identical.
* an NVMe-only expert is never fetched and never given a VRAM slot with no host bytes behind it.
* a deliberately permuted map proves `src_indices` are host slots, not expert ids.
* the C++ executor reproduces the identity-layout result bit-for-bit when weights are shuffled
  into different rows and the map undoes it; an identity map is a no-op; and a **negative
  control** (rolled map) proves the indirection is actually live rather than the suite passing
  vacuously.
* tier off is byte-identical to before.

Regression: `tests/moe/` 223 passed. The two failures in the wider run
(`test_cpu_moe_q4_0::test_cpu_decode_q4_0_matches_ggml_mmvq`, and six `Cfg has no attribute
moe_prefetch` stub failures in `tests/engine/test_cache_budget.py`) reproduce at HEAD with the
changes reverted.

**Note for whoever builds next:** the C++ extension needs the venv's CUDA 13 to compile
(`CUDA_HOME=$FT_VENV/lib/python3.12/site-packages/nvidia/cu13`, as `scripts/serve-qwen38.sh`
sets); the system nvcc is 12.8 and the toolchain guard rejects it. `python setup.py build_ext
--inplace` then takes ~36 s, and the `.so` must be copied into site-packages (the `rsync` ops
rule excludes `*.so`).

## What is left

* **B4, host half** — `ExpertTier3::resolve()` called from `CpuMoeExecutor::submit()`. It has
  to live in C++: `decode_submit` only stages an async D2H of the routing pack and rings a
  doorbell, so the ids are not valid host-side until the GPU reaches it. Doing the resolve in
  Python would force a sync and destroy the overlap the hybrid path exists for. That means a
  C++ port of `HostExpertPool` + `NvmeExpertStore` (both already specified and tested in
  Python, which is the reference to port against).
* **Engine wiring** — allocate `host_slots`-wide pinned banks, seed the pool, install the map
  via `set_host_slot_table`. This can be a new tier3 loader rather than a change to
  `checkpoint/ftw.py`.
* **B2 wiring** — `PleRowCache` into `PLELayer` as a `PLETableBackend`, with the host-side
  resolve running ahead of the graph.
* **B5** prefill staging ring; **B6** read-ahead (demoted to an optimisation); **B7** cancelled.

## B4 complete — expert weights stream from disk, end to end

`kernel/csrc/cpu_moe/expert_nvme.h` (~340 lines) ports `HostExpertPool` + `NvmeExpertStore` to
C++ and runs on the MoE coordinator thread. `CpuMoeExecutor::submit()` calls
`tier3.resolve(...)` before waking the workers, so a routed expert that is not in the host pool
is paged in first. **No CUDA graph change**: the resolve merely lengthens the
`cuStreamWaitValue64` the GPU is already parked on.

`tier3/wiring.py::enable_cpu_tier3` bridges the FTW index and the pool geometry to it.

**Verified end to end** (`tests/tier3/test_end_to_end_nvme.py`): a CPU MoE decode whose expert
weights live only in an FTW file on disk, with a pool holding 12 of 32 experts per layer, is
**bit-identical** (`rtol=0, atol=0`) to a full-residency executor over 12 steps at batch 1 and
3. Plus: the residency map never over-subscribes the pool or double-assigns a slot, and a
negative control shows an undersized pool is *reported* rather than silently wrong.

### Three things integration corrected

1. **Every bank must be paged.** The plan (and the earlier measurement) kept NVFP4's scale and
   global banks resident at full width. That is impossible as written: the fused H2D gather
   applies **one** row index across all banks of a layer, so a full-width bank would be read at
   the fetched expert's *host slot* and return another expert's scales. Keeping them resident
   needs a per-bank row table inside `fast_index_copy_multi` — a real kernel change for a
   second-order gain. So `host_bank_split` now pages everything: +12% bytes, 6 reads per expert
   instead of 2, and the 7.2 GiB the scales would have held goes back into the pool. The two
   unaligned `*_global` banks (2,560 / 5,120 B) get an aligned-window bounce read; they are
   0.3% of the bytes.
2. **An undersized pool is silent corruption, not a slowdown.** If a step routes to more
   distinct experts than the pool has slots, the resolver cannot place them all and the compute
   skips the rest — wrong numbers, no crash. Now counted (`tier3_stats()["pool_exhausted"]`,
   asserted 0 by the tests) and rejected at startup by `enable_cpu_tier3`'s
   `min_step_experts` check. The pool must exceed `max_running_req * top_k`.
3. **A lifetime footgun, found by a segfault.** The C++ holds raw pointers into the handle's
   pinned residency map and base-address tables and writes through them from the coordinator
   thread. My own test dropped the handle, freeing that memory under a live writer
   (`munmap_chunk(): invalid pointer`). `enable_cpu_tier3` now anchors the handle on the
   executor so the caller cannot drop it, with a regression test.

Also fixed while building: a race in the reader pool where a slow reader could re-enter the
job loop after the next batch had begun rebuilding the job vector. Batches are now
double-buffered by generation and readers bail when the generation moves.

## Still outstanding

* **Engine wiring for a live serve** — a tier3 loader that allocates `host_slots`-wide pinned
  banks from FTW (rather than the full-width ones `load_ftw_banks` builds) and calls
  `enable_cpu_tier3` during `_init_offload_moe_cache`. Everything it needs now exists and is
  tested; this is assembly, not design.
* **B2 wiring** — `PleRowCache` into `PLELayer` (the cache and reader are built and validated).
* **B5** prefill staging ring; **B6** read-ahead (an optimisation, ~2.3x on the disk term
  alone); **B7** cancelled by the skew measurement.

## SERVABLE — measured, and slower than the model predicted

`scripts/serve-qwen38-tier3.sh` boots Qwen3.8-Flash-Next with the expert corpus on NVMe and a
355/512 (69 %) host pool, and serves coherent text. Engine seam: `FREETOKEN_TIER3_FTW` +
`FREETOKEN_TIER3_HOST_SLOTS` select `tier3.loader.load_tier3_banks` (empty pinned pools, **no**
disk read at boot) instead of `load_expert_banks`, then `_enable_tier3` arms the resolver.

All runs: `--dense-quant fp8`, KV fp8, `--gpu-reserve-mb 14500` (~14.4 GiB card),
**speculative decoding OFF**, `scripts/bench-decode.py --tokens 256 --runs 2`.

| configuration | expert RAM | bench | english | code | japanese | TTFT |
|---|---|---|---|---|---|---|
| full residency (no tier 3) | 63.4 GiB | **95.8** | **97.4** | **95.3** | **98.9** | 1.14 s |
| tier 3, 505/512 slots (98.6 %) | 62.6 GiB | 61.5 | 64.0 | — | — | ~5 s |
| **tier 3, 355/512 slots (69 %)** | **44 GiB** | **39.1** | **51.1** | **43.2** | **47.1** | 4.8 s |

A repeated identical prompt (radix-cached prefill, warm pool) reaches **91 tok/s**.

**The projection was wrong and it is worth saying why.** The model in this document predicted
90-110 tok/s from NVMe bytes per step. Measured disk traffic during decode is **0 MB/s** -- once
warm the pool absorbs everything, exactly as the simulator said. So the ~2x decode cost is *not*
I/O, and the whole bytes-per-step framing missed it. Two candidates remain, in order of
suspicion, neither yet isolated:

1. **The residency gate pushes work onto the CPU executor.** The ensure kernel may only fetch
   host-resident misses; at 69 % residency roughly a third of misses are ineligible and fall to
   the CPU branch regardless of what the bandwidth-matched split wants. `cpu/layer` rises and
   the CPU becomes the critical path. This predicts the residency dependence (62 tok/s at 98.6 %
   vs 45 at 69 %) and is consistent with disk being idle.
2. **Per-layer resolver latency on the critical path** -- `resolve()` runs on the coordinator
   before the workers are woken, 48 times a step, even when every route hits.

A **device mirror** of the residency map was added (`host_slot_dev`, refreshed with one 2 KB H2D
per layer after the CPU pass) on the theory that the ensure kernel's UVA host gather was the
fixed cost. It measured **no change**, so that theory is dead; the mirror is kept because
reading device memory is the right shape, not because it bought anything.

Prefill costs ~4.8 s regardless of prompt length: a chunk assembles every layer, gathering
pool-resident experts to the buffer over PCIe and reading the other ~31 % from disk. An earlier
version staged that on the CPU and cost ~30 s a chunk -- torch's intra-op pool is pinned to one
thread here because the cores belong to the MoE workers, so the fix was to do both halves as
`fast_index_copy_multi_jit` gathers.

### This is not yet the 16 GB / 64 GB profile

The expert half works, but the 47.7 GiB PLE table is still fully resident, so host RAM is
~44 + 47.7 ~ 92 GiB. Reaching 64 GiB total needs the B2 PLE row cache wired into `PLELayer`
(built and validated, not wired). Until then this demonstrates the expert tier, not the profile.

Also outstanding: MTP experts are a full-width bank layer built from the raw checkpoint, which
the host-slot pool cannot represent -- speculative decoding is off, and turning it back on is
worth roughly a 2x on tokens/step.

## The 16 GB / 64 GB profile, live

`scripts/serve-qwen38-tier3.sh`. Both large host structures now stream from NVMe:

| host structure | fully resident | tier 3 |
|---|---|---|
| routed experts (48 layers x 512) | 63.4 GiB | **52.7 GiB** (425 slots/layer, 83 %) |
| PLE n-gram table (320 M rows) | 47.7 GiB | **4.0 GiB** row cache |
| prefill staging (2 x 512 experts) | — | 2.6 GiB |
| **pinned total** | **111 GiB** | **~59 GiB** |

Measured on the ~14.4 GiB emulated card, `--dense-quant fp8`, KV fp8, spec off,
`bench-decode.py --tokens 200 --runs 2`:

| configuration | pinned RAM | bench | english | code | japanese | TTFT |
|---|---|---|---|---|---|---|
| full residency | 111 GiB | 95.8 | 97.4 | 95.3 | 98.9 | 1.14 s |
| tier 3, experts only, 69 % | 108 GiB | 39.1 | 51.1 | 43.2 | 47.1 | 4.8 s |
| **tier 3, experts 83 % + PLE cache** | **~59 GiB** | **53.1** | **56.4** | **56.7** | **59.7** | 3.1 s |
| same, warm pool | ~59 GiB | 76.3 | 77.8 | — | — | — |
| PLE resident, experts forced to 14 % | 64 GiB | 16.2 | 15.9 | — | — | 13.7 s |

**Tiering the PLE table is what makes the profile work**, and the last row is why: leaving it
resident squeezes the expert pool to ~14 % residency, which puts ~0.67 GB/step on the disk and
makes decode NVMe-**bandwidth**-bound at 16 tok/s. Moving 43.7 GiB from the table to the pool
buys 83 % residency, where a warm pool does **zero** disk I/O.

## MTP

Works with the expert tier. `append_spec_expert_bank` adds the draft experts as a full-width
bank layer built from the raw checkpoint, which a host-slot pool cannot represent -- so that
layer stays **resident** (1.35 GiB) and its residency row is the identity. Only `shape[1:]` was
ever required to match across layers, so the uniform-width assert was over-strict.

The bug this exposed is worth recording: `tier3_configure` was passing the *executor's* layer
count to the C++ `init`, so the resolver ran on the resident MTP layer and indexed `entry_off`
and the per-layer slot arrays **out of range** -- producing token 0 forever. The paged layer
count is now explicit.

MTP + tier 3 measures 45.6 / 55.1 / 51.0 / 45.7 at 69 % residency, ~+10 % over spec-off. Less
than the ~2x it gives at full residency: a 3-row verify triples the distinct experts per layer,
so most of the win is eaten by extra expert traffic. **Not yet combinable with the PLE row
cache** (see below).

## Speculative MoE loading

Implemented as **asynchronous prefill read-ahead**: `PrefillStager.begin()` issues a layer's
NVMe reads and returns, and `wait_prefill_layer` joins them one layer later, so they overlap the
previous layer's GEMMs. Staging is double-buffered, with a CUDA event guarding host reuse of a
slot and per-slot index tensors.

**Measured neutral at short prompts** (TTFT 3.07 s vs 3.09 s): a 50-token prefill has almost no
GPU compute to hide the reads behind. It should matter for long prefill, which is not measured.

More importantly, **decode had nothing to prefetch** at the geometry measured here: at 83 %
host residency a warm pool did 0 MB/s of disk I/O, and at 14 % residency the workload is
bandwidth-bound (0.67 GB/step), where read-ahead cannot help either. Cross-layer expert
prediction was therefore *not* built.

### Re-measured 2026-09-02: the conclusion holds, and here is the steady-state evidence

A first pass appeared to overturn this. With the new `[tier3]` readout on the 256K profile it
reported `host_hit_rate=0.9473 nvme_reads/step=14.15 stall_us/step=14695.7` -- 14.7 ms of a
~23 ms step -- which would have made speculative tier3->tier2 promotion the single biggest
lever available. **That reading was wrong: it was a cold-pool warm-up window**, the only delta
window that run produced, taken ~20-40 decode steps after server start while the pool was still
filling (hit rate 0.947 and climbing).

Run long enough to reach steady state, the same profile and the same 90K-depth needle workload
gives, across 40+ windows:

```
[tier3] host_hit_rate=0.9990 nvme_reads/step=0.45 nvme_MB/step=1.192 stall_us/step=418.8
[tier3] host_hit_rate=1.0000 nvme_reads/step=0.00 nvme_MB/step=0.000 stall_us/step=29.8
[tier3] host_hit_rate=0.9996 nvme_reads/step=0.15 nvme_MB/step=0.397 stall_us/step=145.7
```

**0.0-0.45 reads/step and 0.03-0.42 ms of stall in a ~23 ms step (~1 %).** There is nothing for
a decode-time expert prefetcher to recover, exactly as originally concluded -- now with a
per-step number rather than an inference.

Two traps that produced the false positive, both worth remembering:

* **A single delta window is warm-up, not steady state.** `prewarm_pool` fills each layer with
  the arbitrary leading block `[0, host_slots)`, so the pool needs a few hundred decode steps of
  LRU churn to converge on the hot set. Always read 20+ windows.
* **Block-layer counters attribute prefill to decode.** `/proc/diskstats` showed ~7 GB per
  400-token request, which looks like 17 MB/token of decode traffic; the tier-3 counters over
  the same request report 69 MB. The difference is `_prefill_routed`'s whole-corpus scan. Per-
  partition deltas around a request separate them (`p2` 1 MB vs `p3` 6901 MB, with the corpus
  on p3).

The corollary is that the decode cost at this profile is **tier-1 <-> tier-2**, not tier 2 <->
tier 3: at a 22.9 % GPU miss rate `[moe-stats]` reports `fetch_MB/step=226` over PCIe plus the
CPU GEMV. The levers are the GPU slot count (what NVFP4 KV buys) and `--moe-prefetch layer`,
not NVMe.

### Storage: ntfs3 was not the ceiling at this access pattern either

A3 recorded ntfs3 plateauing at ~2.4 GB/s and flagged ext4 as an escape hatch. Re-measured with
O_DIRECT reads at the actual expert-row size (2,764,800 B) on the same physical drive, ext4 on
`nvme5n1p3` against ntfs3 on `nvme5n1p2`:

| readers | ext4 | ntfs3 |
|---|---|---|
| 1 | 3.15 GB/s | 2.69 GB/s |
| 4 | 5.95 GB/s | 5.69 GB/s |
| 16 | 5.78 GB/s | 5.65 GB/s |

Both reach ~5.7 GB/s at QD>=4; ext4 leads only at QD1 (+17 %). The 2.4 GB/s figure does not
reproduce at 2.64 MiB rows. Moving the corpus to ext4 changed neither TTFT (42.4 s both) nor
decode at depth. Keep it there if convenient -- the QD1 edge is real -- but it is not a lever.

### Long-context decode is not slow; the needle harness measures it wrong

`scripts/needle-check.py` reports a "decode tok/s" computed over a one-word answer -- a handful
of tokens, dominated by first-token latency and MTP warm-up. It gave 16-50 tok/s at 90K on this
profile and sent two separate investigations chasing a depth penalty that does not exist.

Measured instead with a real 300-token generation at increasing depth (same server, KV pool
preallocated at 262K so the MoE geometry is fixed):

| context | decode | step | accepted |
|---|---|---|---|
| 23,824 | 91.4 tok/s | 24.5 ms | 2.24 tok/step |
| 51,024 | 87.4 tok/s | 24.7 ms | 2.16 |
| 90,124 | 90.7 tok/s | 25.6 ms | 2.33 |

**Depth costs ~1 ms/step from 24K to 90K (~4 %)**, which is exactly what the QSA kernels
predict in isolation: the O(context) indexer scan (`qsa_mqa_paged` + block top-k + expand) grows
0.012 -> 0.036 ms per layer between 8K and 128K while the O(budget) sparse attend stays flat at
~0.025 ms; x48 layers that is 1.8 -> 2.9 ms of a ~25 ms step. `[moe-stats]` confirms the rest is
depth-invariant: miss_rate 0.42-0.49 and 357-405 MB/step of PCIe expert traffic at BOTH short
and 90K context.

So the decode levers on this profile are not depth-specific. The dominant term is the ~0.45 GPU
expert miss rate (~400 MB/step over PCIe, roughly 8 ms of the 25 ms step); the levers are GPU
slot count and expert prefetch, and prefill (42 s for 90K, ~2,150 tok/s) is what a long-context
user actually waits on.

### Two flags this launcher left off, and why one of them cannot be turned on

* **`--moe-prefill-hit-d2d`** -- measured **neutral** here (prefill 41.9 s vs 42.0 s at 90K).
  With ~2,000 GPU slots against 25,088 experts almost nothing is resident for prefill to copy
  device-to-device. Kept as an env switch (`FREETOKEN_MOE_PREFILL_HIT_D2D=1`) since it is a win
  on the roomy config.
* **`--moe-prefetch layer`** -- **crashes tier 3** with a CUDA illegal memory access a few
  layers into decode. The lookahead admits host bank rows by EXPERT ID; tier 3 pages them into
  a per-layer pool of `host_slots < num_experts` and the demand path translates through
  `row_of()` / `host_slot_tbl`, which the prefetch admission path does not do, so it reads past
  the pool. `_adjust_config` now refuses the combination at config time with that explanation
  (`tests/tier3/test_prefetch_refused.py`). Making it work means giving the prefetch path the
  same slot indirection -- worth doing, since a 0.45 miss rate is the biggest decode term left.

### Speculative tier3 -> tier2 promotion: recall is fine, bandwidth is not

Re-opened at the 48 GiB geometry (352 host slots), where decode does stall on NVMe: 7.4 ms of a
~32 ms step. Settled offline with `scripts/tier3-promote-recall.py`.

Predicting layer J+2 from layer J's residual through J+2's gate and filtering to non-resident
recalls **62.7 %** of the experts that actually miss (top-32) -- which corrects the older "~4 %
precision, cross-layer prediction is hopeless" reading: a *wide* prediction does find the cold
tail. But it costs **505 MiB/token of promotions against 18.9 MiB/token of demand reads**, i.e.
27x the traffic, on a drive measured at 5.7 GB/s. Only top-4 fits the bandwidth (18.5 MiB/token)
and it recalls 14.8 %, worth ~3 % of decode for double the disk traffic.

Retention policy is dead independently: at 352 slots LRU scores 0.9642 host hit against a 0.9701
oracle, and a frequency-pinned hot set is byte-identical to LRU.

## Remaining

* **MTP + PLE row cache together.** The cache publishes slots in
  `prepare_host_tables_for_replay`, which `GraphRunner.replay` calls; the speculative executor
  replays its own graphs and does not. Refused at startup rather than served stale. Wiring the
  same hook into `spec.py`'s three replay sites is the fix.
* Long-prefill measurement of the read-ahead.
* `--moe-host-cache-gb` / `--ple-cache-gb` as real CLI flags (they are `FREETOKEN_TIER3_*` env
  vars today).

## Optimisation to the 90 tok/s target — reached

Four changes, in order of what they were worth:

**1. Prewarm the pool at boot (+15 %).** The pool started empty, so the residency map was all
-1, the ensure kernel could fetch *nothing*, and every routed expert fell to the CPU behind a
disk read. `prewarm_pool` fills each layer with experts `[0, host_slots)` -- **52.7 GiB in
10.4 s at 5.4 GB/s**, because experts are contiguous inside an FTW entry so it is one
sequential read per (bank, layer) rather than ~122k scattered ones. Which experts hardly
matters: any `host_slots` of them gives the same residency fraction. The C++ is told afterwards
(`tier3_prewarm`) or it would hand out slots that already hold live rows.
Bench 53.1/56.4/56.7/59.7 -> 61.0/64.8/61.9/65.6.

**2. MTP together with the PLE cache (+12 %).** Only spec **graph A** carries the main tower,
so only it needs the PLE hook -- graphs B and C chain the MTP head, and `ple_layer_ids` indexes
the main tower. One hook call after `g.stage(batch, req)` lifted the restriction.
-> 68.8/68.2/77.6/67.7, warm 88.2/87.3/92.2/78.3.

**3. One map refresh per step instead of 48.** A layer's residency only changes in its own
`resolve()`, so a whole-map 96 KiB H2D at the top of a step shows every layer exactly what the
per-layer version did -- for 1 stream op instead of 48.

**4. Staging sized to the non-resident set (+2 GiB of residency).** Prefill staging only ever
holds a layer's *missing* experts, and a prewarmed pool stays exactly `host_slots` full, so it
needs `num_experts - host_slots` rows, not `num_experts`: 2.6 GiB -> 0.28 GiB. Spent on the
pool: 425 -> 458 slots (83 % -> 89 % residency).

### Result, 16 GB VRAM / 64 GB RAM, MTP on

| pass | bench | english | code | japanese |
|---|---|---|---|---|
| first (cold pool + prefill) | 82.9 | 74.6 | 92.9 | 75.3 |
| **warm** | **99.4** | **94.1** | **104.6** | **91.6** |

Warm clears 90 on every task and matches the 111 GiB full-residency configuration (95.8 / 97.4
/ 95.3 / 98.9, spec off) at **~60 GiB pinned**. The first pass through unseen content still
costs 75-93 while LRU reshapes the prewarmed set; a frequency-ordered prewarm (the offline
`decode_freq` profile) is the obvious next step if that matters.

A partial edit cost a run here: the staging allocation shrank to 54 rows while its `uint8` view
still reshaped to 512, so every row address was 172,800 B apart instead of 1,638,400 --
`destination is 172800 B, need 1638400`. Worth stating because the failure was loud only by
luck; the same slip in the *source* offsets would have read plausible wrong weights.

## Bug: file-descriptor exhaustion killed a live server

Symptom, mid-serve on a 6,597-token prompt right after an elastic-KV grow:

```
OSError: [Errno 24] Too many open files
  ... freetoken/kernel/radix.py -> load_aot("radix") -> subprocess -> os.pipe()
```

The traceback pointed at a JIT build of the radix kernel, nowhere near tier 3 -- the build was
just the first thing to ask for a descriptor after the tier had taken them all.

**Cause.** All three readers cached an fd per **(thread, shard)**: `PleDiskReader` (64 readers),
`NvmeExpertStore` (16) and the C++ `ExpertTier3` (16), over 10 shards each -- ~960 descriptors
against a 1024 soft limit. The justification written in the code ("per-thread fds keep the
kernel off a shared file-position lock") was simply wrong: `pread`/`preadv` take an explicit
offset and never touch the file position, so a shared descriptor is safe under concurrency.
That reasoning applies to `read()`, not `pread()`.

**Fix.** One fd per shard, shared, in all three readers (double-checked locking; the C++ side
closes the loser of an open race). Plus a defensive `RLIMIT_NOFILE` nudge at arm time, mirroring
the `RLIMIT_MEMLOCK` nudge in `moe.host_banks`.

**Verified** by rerunning the server under the same tight `ulimit -n 1024`: **111 descriptors**
(was ~960), and a **12,202-token** prompt -- nearly double the one that crashed -- prefilled,
grew the KV pool and answered correctly with speculative decoding live.

Regression tests assert the count stays flat in the reader-pool size
(`test_descriptors_do_not_scale_with_reader_count` in both the expert-store and PLE suites),
because the failure mode surfaces far from its cause and only under a tight limit -- my own
shell has `ulimit -n 1048576`, which is exactly why I never hit it.

## Bug: a cache rebuild left tier 3 holding freed addresses

Symptom, right after an elastic-KV grow (`Cache rebuilt: ... MoE cache 2683/25088`):

```
_invalidate_prefill_buffer -> slot_for_id.view(-1)[...] = -1
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

Reported at an unrelated indexing op, as async CUDA faults are.

**Two distinct faults, both mine, both introduced with the tier:**

1. **Dangling device pointer tables.** `_tier3_buffer_ptrs` and `_materialize_dst_ptrs` cache the
   *addresses* of the prefill overlap buffers and the slot caches. `rebuild()` tears both down
   and reallocates them, so the next tier-3 gather wrote through freed pointers. Deferred
   prefill gathers (`_tier3_pending_fill`) and in-flight staging reads had the same problem:
   their destination no longer existed. Now `rebuild()` calls `_tier3_invalidate()` first, which
   drops the caches (they repopulate lazily from the new buffers) and waits out any in-flight
   read.

2. **`rebuild()` reallocated the routing pack at a hardcoded 2 rows.** Only `__post_init__` had
   been widened to `ROUTING_PACK_ROWS`. So after any rebuild the pack lost its tier-3 host-slot
   row and `gpu_host_slots` aliased the freed tensor -- and because
   `cpu_executor.decode_submit` *shape-checks* the pack against its pinned mirror and silently
   falls back to per-row copies when they disagree, the eviction exclusion list would simply
   have stopped arriving. Worse than a crash: the tier would have kept running with slots
   recyclable under a live gather.

**Verified** by forcing the exact path -- `ft ctl cache rebuild --kv 32768 --moe 2600` ->
`Cache rebuilt` + spec graphs re-captured, no traceback -- then serving both a short and a
12,202-token prompt afterwards, correct output, 111 fds, server alive.

Five regression tests in `tests/tier3/test_rebuild_invalidation.py` cover the pack width, the
re-derived views, the pointer-cache invalidation, inertness when the tier is off, and that the
lazy tables repopulate against the *new* buffers.

**Lesson worth carrying:** the tier adds state that mirrors the cache's own allocations, and
`rebuild()` is the one place that invalidates them. Anything cached off `bank_caches`,
`prefill_bank_buffers` or `_copy_*_ptrs` has to be listed there.

## 128K context on the profile (fp8 KV)

**There is no NVFP4 KV cache.** `--kv-cache-dtype` accepts `auto|bf16|fp8` only; NVFP4 in this
codebase is the *expert weight* format (`--nvfp4-backend`). fp8 is already the narrowest KV
option and what the profile uses, so these are fp8 numbers.

Run: `FREETOKEN_KV_MAX_TOKENS=131072 FREETOKEN_KV_RESERVE_TOKENS=16384`, everything else at the
profile defaults (458 host slots, 2 GiB PLE cache, MTP on, ~14.4 GiB emulated card).

| scenario | result |
|---|---|
| short context, first pass (KV still at the 16,384 reserve) | 78.0 / 77.5 / 85.7 / 75.8 |
| short context, warm, **after** the KV grew to 131,072 | **94.8 / 85.1 / 105.1 / 86.6** |
| 95,723-token prefill (cold) | 43.6 s, 2,196 tok/s |
| 95,723-token prefill (radix-cached) | 1.8 s |
| decode at 96K depth, 171 steps | **36.0 tok/s** (accepted/step 1.34) |
| decode at 96K depth, first 19 steps after the rebuild | 12.1 tok/s |

Two things worth separating:

**The cap is free until used.** Elastic KV allocates only the reserve at boot (16,384 tok,
0.22 GiB), so raising the ceiling to 128K costs nothing until a long request arrives. The grow
then rebuilds: `KV 16384 -> 131072 tokens, expert slots 2771 -> 2173`, KV 1.75 GiB.

**Holding a 128K pool costs ~5 % at short context**, not more: warm mean 92.9 here against 97.4
at the 32K cap, entirely explained by the expert cache dropping 2,771 -> 2,173 slots. The pool
does shrink back after `--kv-elastic-idle-seconds` (default 300).

**Depth itself is the expensive part.** At the *same* 2,173-slot geometry, short context runs
92.9 and 96K depth runs 36.0 -- a 61 % drop that is attention over 96K plus the KV traffic, not
the expert tier. And the 12.1 tok/s figure is a 19-step sample taken immediately after the
rebuild cold-started the slot cache; it is not the steady state. Small samples right after a
rebuild are misleading and should not be quoted.

This run also exercised the elastic grow + MoE rebuild under real load with no traceback --
independent confirmation of the rebuild-invalidation fix above.

## NVFP4 lm_head: exists, measured, worth enabling

`FREETOKEN_LM_HEAD_NVFP4=1` (`models/qwen4_exp/model.py:262`) swaps the target head for
`Nvfp4LMHead`. It is not new work -- it was already implemented and opt-in.

| | expert slots | first pass | warm |
|---|---|---|---|
| fp8 head | 2,771 | 78.0 / 77.5 / 85.7 / 75.8 | — |
| **NVFP4 head** | **3,007** (+236) | **84.1 / 78.3 / 87.3 / 78.0** | 95.5 / 91.2 / 109.8 / 88.6 |

+236 slots is 0.62 GiB, matching the documented saving (head read 636 -> 318 MB plus the draft
copy). Output stayed coherent. **Left opt-in**: the head perturbs the emitted logits and the
GSM8K / MMLU / needle gates have not been run against it -- that is the only thing between it
and being a default.

Note the startup line `lm_head_quant=fp8_pertensor` is the *config* field; the NVFP4 override
happens locally in the model, so the slot count is the thing to check.

## NVFP4 KV cache: does not exist, and the case for it is stronger than I assumed

`--kv-cache-dtype` accepts `auto|bf16|fp8` only. NVFP4 in this codebase is the expert weight
format. So this would be new work.

My first instinct was that it would not help, on the grounds that expert slots are not the
long-context bottleneck -- at a *fixed* 2,173-slot geometry, short context runs 92.9 and 96K
depth runs 36.0, so the drop is attention and KV traffic, not expert residency. That reasoning
was half right and led to the wrong conclusion, because NVFP4 KV attacks **both** terms at once.

Measured sensitivity at 96K (identical config apart from the KV dtype):

| KV dtype | KV at 128K | expert slots at depth | decode @96K |
|---|---|---|---|
| bf16 | 3.35 GiB | 1,792 | 30.1 tok/s |
| **fp8** | **1.75 GiB** | **2,173** | **36.0 tok/s** |
| nvfp4 (projected) | ~0.98 GiB | ~2,460 | ~41 (**not measured**) |

Halving KV bytes bought **+19.6%** decode at depth. NVFP4 is ~0.55x fp8
(`head_dim/2` codes + per-16 e4m3 scales + a row global vs `head_dim` + fp32 scale), so a
similar step is plausible. **Caveat: the two effects are confounded** -- the dtype change moved
KV bytes *and* expert slots together, and this measurement cannot split them.

### What building it involves

1. Storage: a packed-uint8 slab plus per-16 block scales. `QSAKVCache` already carries a third
   tier for fp8's `_kv_scale [2, layers, slots, kv_heads]`, so the shape is precedented.
2. A `qsa_store_kv_nvfp4` quantize path beside `qsa_store_kv_fp8`.
3. **The attend kernel must dequantize NVFP4 in registers** -- the real work, in the QSA sparse
   attention kernel, and the same for the indexer slab.
4. `spec_kv_bytes_per_token` (it currently prices a flat `4 if itemsize == 1` scale), plus the
   `--kv-cache-dtype` choices and `_adjust_config` plumbing.
5. Quality gates. This is the gating risk, not performance: 4-bit KV degrades long-context
   retrieval exactly where this is meant to be used. fp8 KV shipped only after GSM8K / MMLU /
   needle came back identical; NVFP4 KV has to clear the same bar **at 128K**, and it may not.

Recommendation: worth doing, but as a scoped project with the needle test run first on a
prototype, not as an extension of the tier-3 work.

## NVFP4 KV cache: quality gate PASSED -- the kernel project is greenlit

Rather than build the format first, the numerics were probed end to end:
`kvcache/nvfp4_kv.py` rounds K/V through NVFP4 at store time and lets them land in the normal
slab (`FREETOKEN_KV_FAKE_NVFP4=1`, paired with `--kv-cache-dtype bf16` so nothing is quantized
twice). Numerically identical to a real NVFP4 KV cache; no format, no kernel, no memory saving.

Format matches the expert weights exactly, so a real implementation reuses the same math:
`value = E2M1[code] * block_scale_e4m3 * row_global_fp16`, 16-element blocks along `head_dim`,
one global per (token, head) -- the granularity `_kv_scale` already carries for fp8.

**Numerical error** (4096x256, vs the fp8 row-scale round trip the pool does today):

| | fp8 | NVFP4 | ratio |
|---|---|---|---|
| relative error | 0.0261 | 0.0951 | 3.6x |
| attention logit relerr | 0.0261 | 0.0951 | 3.6x |
| softmax L1 | 0.0811 | 0.2673 | 3.3x |

**End-to-end gates**

* **Needle 5/5 at 95,148 tokens** (depths 0.1-0.9), and 5/5 at 30,952. This is the test 4-bit KV
  was most likely to fail, and it did not.
* Reasoning probe, 8 deterministic prompts: **7/8 correct on bf16, fp8 AND NVFP4 -- identical**.
  The 8th is a `max_tokens` truncation affecting all three equally, not a quality difference.
* Text divergence from the bf16 baseline: fp8 5/8 identical, NVFP4 3/8. NVFP4 perturbs the
  sampled path more, exactly as 3.6x the error predicts, but **correctness is unchanged**. The
  fp8 control matters here: the shipping default also diverges, so divergence alone is not a
  verdict.

**Two bugs the probe surfaced, both of which a kernel implementation would inherit:**

1. **Host-tensor lookup under graph capture.** Building the E2M1 grid with `torch.tensor(...)`
   inside `store_kv` is a host->device copy, which capture rejects outright
   ("Cannot copy between CPU and CUDA tensors during CUDA graph capture"). The rounding is now a
   chain of scalar comparisons; a test greps the source to keep it that way.
2. **All-zero rows produced NaN.** The fp16 row global underflows to 0 for a zero row, making
   the block scale `0/0`. Attention would have propagated that silently. The fp8 path documents
   "zero codes x zero scales read as a finite 0" as an invariant; NVFP4 now upholds it via a
   guarded denominator. **A real kernel must carry this guard** -- never-written slots, dummy
   pages and padded rows all hit it.

10 tests in `tests/kvcache/test_nvfp4_kv.py` pin the grid, determinism, the error ratio, outlier
behaviour, per-head independence and the zero-row invariant.

### Remaining work for the real format

1. Packed-uint8 slab + per-16 e4m3 block scales (extend `_kv_scale`'s third tier).
2. `qsa_store_kv_nvfp4` beside `qsa_store_kv_fp8`.
3. **Dequantize in registers inside the QSA attend kernel** -- the bulk of the work -- and the
   indexer slab.
4. `spec_kv_bytes_per_token` (it prices a flat `4 if itemsize == 1`), the `--kv-cache-dtype`
   choices, `_adjust_config`.
5. Re-run these gates against the real kernel; the probe only certifies the numerics.

Projected: KV 1.75 -> ~0.98 GiB at 128K, ~+290 expert slots, and by the measured
bf16->fp8 sensitivity (+19.6% decode at 96K) a further step of similar order.
