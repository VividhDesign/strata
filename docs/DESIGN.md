# Strata design notes

This document explains how Strata works and why it is built this way. Read it alongside
`src/hnsw.cpp`, the core of the project.

## 1. The problem

Given N vectors and a query q, return the k vectors closest to q. Exact search costs
O(N·d) per query: about 128M multiply-adds for SIFT1M. That is too slow at scale.
Approximate nearest-neighbour (ANN) indexes trade a little recall for orders of magnitude
less work.

## 2. HNSW in one page

HNSW (Malkov & Yashunin, 2016/2018) is a proximity graph: each vector is a node, linked to
nearby nodes. A greedy walk ("move to whichever neighbour is closer to q") quickly converges
near q. Two ideas make this work at scale:

1. **Hierarchy.** Every node gets a random level L with P(L ≥ l) = M^(-l). Level 0 contains
   everything; each higher level holds ~1/M of the level below. This is a skip list in graph
   form. Search starts at the top level's entry point, walks greedily, and drops a level each
   time it can't improve. The top levels cover long distances in a few hops; level 0 handles
   the fine-grained search.
2. **Beam search on the bottom level.** Instead of a single greedy walker, level 0 keeps
   the `ef` best candidates seen so far (Algorithm 2 in the paper). Larger `ef` means higher
   recall and slower queries. This is the main knob traded off in the benchmarks.

Complexity: about O(log N) hops, each costing up to 2M distance computations.

### Insertion (paper Alg. 1, `HNSWIndex::insert_one`)

1. Draw the node's level.
2. Greedy-descend from the top down to level+1.
3. On each level from min(level, top) down to 0, run a beam search with `ef_construction`.
   Pick M neighbours (see §3), link both ways, and shrink any neighbour list that overflowed.

### Parameters

| Parameter | Meaning | Effect |
|---|---|---|
| `M` | max links per node per level (level 0 has 2M) | higher → better recall, more memory, slower build |
| `ef_construction` | beam width while building | higher → better graph, slower build |
| `ef` (search) | beam width while querying | the recall ↔ latency knob |

Level 0 gets 2M links because it is where the real search happens. The paper found 2M to
be a good default.

## 3. The neighbour-selection heuristic (and the ablation)

Taking the M *closest* candidates as neighbours seems natural, but in clustered data all M
links end up inside the same cluster and the graph loses its bridges between clusters.
The heuristic (paper Alg. 4, `select_neighbors`) goes through candidates from nearest to
farthest. It keeps a candidate c only if c is closer to the new node than to every
neighbour already kept. The kept edges therefore point in different directions.

`bench/results/glove-ablation.json` measures this on GloVe-100 with everything else
identical. **Recall@10 at ef=64 is 0.762 with the heuristic and 0.662 without it**, which
the chart in the README shows across the whole curve.

## 4. Memory layout

```
level 0, one fixed-size block per node (contiguous array, 64-byte aligned):
  [neighbour count: u32][neighbour ids: u32 × 2M][pad to 16 B][vector: f32 × d][pad to 16 B]
upper levels, separate per-node array (rare: ~1/M of nodes):
  level l: [count: u32][neighbour ids: u32 × M]
```

- A level-0 search step reads a node's neighbour list and then each neighbour's vector.
  Keeping a node's links and vector in the same block means one cache-miss stream instead
  of two separate arrays.
- Internal ids are dense `uint32` indices into this array. User labels (`uint64`) live in a
  side array plus a hash map, so the graph itself stores 4-byte ids.
- While scanning a neighbour list we `__builtin_prefetch` the *next* neighbour's vector.
  This hides DRAM latency behind the current distance computation.

### Scalar quantization (`quantization="sq8"`)

With SQ8 the vector in the level-0 block is replaced by **one byte per dimension**:

```
  [neighbour count: u32][neighbour ids: u32 × 2M][pad to 16 B][codes: u8 × d][pad to 16 B]
```

For d = 128 and M = 16 a block shrinks from 656 to 272 bytes. Search is memory-bound, so the
traversal touches 2.4× fewer cache lines.

- **Codebook.** Each dimension gets its own range: `x ≈ min_d + scale_d · c`, with
  `scale_d = (max_d − min_d) / 255`. The range is trained on the first batch. While the index holds
  fewer than 1,000 vectors, every later batch widens the range and re-encodes the existing codes.
  Without this, one-by-one upserts would freeze a degenerate range taken from the first vector
  (I caught this with a test: recall dropped to 0.50). After that the ranges are frozen and outliers are clamped.
- **Asymmetric distances.** The query stays float. It is transformed once per search, so every
  candidate needs only a widen-and-FMA:
  - L2: `Σ scale_d² · ((q_d − min_d)/scale_d − c_d)²`
  - IP and cosine: `1 − ⟨q, min⟩ − Σ (q_d · scale_d) · c_d`

  The kernels (`kernels::sq8_l2`, `kernels::sq8_dot`) widen u8 → f32 in registers: AVX2 uses
  `cvtepu8_epi32`, NEON uses `vmovl_u8 → vmovl_u16 → vcvtq_f32_u32`. They keep the same multi-accumulator
  structure as the float kernels. The NEON path was checked against the scalar kernel on x86 through
  the NEON_2_SSE emulation header.
- **`rerank=True` (default).** The float vectors are also kept, in a separate cold array outside the
  graph blocks. The graph is **built** with exact float distances, so it is identical to the float index.
  Queries walk the graph on codes and then re-rank the `ef` candidates with exact distances. Recall
  matches the float index and the returned distances are exact.
- **`rerank=False`.** The float vectors are dropped once the ranges freeze. The index is about 3× smaller,
  the graph is built on codes and distances are approximate.

## 5. Distance kernels (`src/distance.cpp`)

Without `-ffast-math` the compiler won't reorder a float sum, so a plain `sum += a[i]*b[i]`
loop stays scalar. The NEON kernels process 16 floats per iteration using **four
independent accumulators**. FMA latency is about 4 cycles, so one accumulator would stall
on its own previous result; four keep the pipeline full. AVX2+FMA does the same with 8-wide
registers. `bench/micro_distance` measures the speedup against the scalar loop.

Metrics:
- L2 returns the **squared** distance. That ranks identically to L2 and skips the sqrt.
- Inner product uses `1 - dot`.
- Cosine normalises vectors once at insert time (and the query once per search), then uses
  `1 - dot`.

## 6. Concurrency model

- **Reads**: `search` takes a `std::shared_mutex` in shared mode, so any number of queries
  run in parallel. They read the graph **without per-node locks**, because no writer can be
  active.
- **Writes**: `add` takes the lock exclusively and then **parallelises internally**: worker
  threads insert different vectors at the same time. They coordinate with:
  - a **1-byte spinlock per node** (`SpinLock`, `include/strata/sync.h`) guarding that
    node's neighbour lists. `std::mutex` is 64 bytes on macOS, so a 1M-node index would
    spend 64 MB on locks. Spinlocks cost 1 MB, and the critical sections are a copy of
    ≤ 2M ints;
  - a mutex for the label → node map;
  - a mutex for the entry point, held for the whole insertion only by a node that becomes
    the new top level (rare).
- **Deadlock freedom**: a node being inserted holds its own lock while it links, and takes
  neighbour locks one at a time. Two inserting nodes cannot wait on each other. For A to
  link to B, B must already have been reachable when A's search ran, which means B had
  finished searching earlier. That ordering cannot hold in both directions at once.
- **Deterministic levels**: a node's level is `hash(seed, label)`, not a shared RNG. That
  makes it thread-safe with no lock, and reproducible.
- **Verification**: the test suite passes under ThreadSanitizer and AddressSanitizer
  (`-DSTRATA_SANITIZE=thread|address`).

Trade-off: writes block reads for the duration of a batch. Production systems such as
Milvus/Lucene avoid that with immutable segments plus a small mutable segment that is
merged in the background. That is the natural next step (see §10).

## 7. Deletes and upserts

Removing a node from an HNSW graph would disconnect its neighbours. So deletes are
**tombstones**: the node stays as a waypoint for navigation but never enters the result
set. An upsert tombstones the old version and inserts a new node.

When tombstones (or filters) exclude nodes, the beam search keeps exploring until it holds
`ef` *valid* results, not just `ef` nodes. `compact()` rebuilds the graph from the live
vectors once tombstones accumulate.

## 8. Filtered search

Filters (metadata, or explicit allow/deny id lists) are applied **during** the graph walk.
Filtered-out nodes are traversed but never returned. Post-filtering (search, then drop) can
return fewer than k results.

When the allowed set is very small, a filtered graph walk wastes time visiting nodes that
cannot qualify. Below `flat_search_cutoff` allowed ids (default 2048), Strata instead
**brute-forces over the allowed set**, which is exact and cheaper at that size. This is the
same planner decision Qdrant and Weaviate make.

## 9. Durability (`src/collection.cpp`, `src/wal.cpp`)

```
collection/
  config.json            immutable settings
  CURRENT                name of the latest complete snapshot  ← commit point
  snap-<seq>/index.bin   HNSW snapshot   (CRC-32 footer)
  snap-<seq>/meta.bin    metadata        (CRC-32 footer)
  wal.log                operations after that snapshot
```

- **Write path**: append a WAL record `[len][crc32][seq][op][payload]` → fsync → apply in
  memory → acknowledge. On macOS `fsync()` only reaches the drive's cache, so we use
  `fcntl(F_FULLFSYNC)` to force a real flush.
- **Checkpoint**:
  1. Write the snapshot into a temp directory and fsync it.
  2. Rename it into place.
  3. Atomically replace `CURRENT` (temp file + rename).
  4. Truncate the WAL.

  A crash at any step leaves either the old snapshot plus the full WAL, or the new snapshot
  plus a WAL whose records are all ≤ the snapshot's seq (skipped on replay).
- **Recovery**: load the snapshot named by `CURRENT`, then replay WAL records with
  seq > snapshot seq. A torn final record (crash mid-write) fails its length or CRC check.
  Replay stops there and truncates the tail. Everything acknowledged before the crash
  survives. The tests simulate this by truncating the files.
- **Corruption**: every snapshot file ends with a CRC-32 (ARMv8 hardware CRC instructions
  where available), so a flipped bit makes `load` fail loudly.

## 10. What I would do next

- **Product quantization / RaBitQ**: int8 scalar quantization is done (section 4). PQ or 1-bit
  RaBitQ codes would cut memory a further 4–8×.
- **Out-of-core rerank vectors**: with SQ8 the float vectors are only read for the final rerank.
  They could live in an mmap'd file while codes and graph stay in RAM.
- **Segments**: an immutable HNSW plus a small mutable buffer, so writes never block
  reads.
- **mmap-able snapshots** for instant startup on large indexes.
- **Replication**: ship the WAL to followers.
