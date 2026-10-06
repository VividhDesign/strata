# Changelog

## 0.2.0

### Added
- **8-bit scalar quantization** (`quantization="sq8"`): one byte per dimension in the graph, with AVX2/NEON/scalar
  asymmetric kernels. With `rerank=True` (the default), the graph is built at full precision and the `ef` candidates
  are re-ranked exactly, so recall matches float32. With `rerank=False`, only codes are stored and the index is
  about 3× smaller.
- **Memory-mapped re-rank vectors**: `Index.load(path, mmap_vectors=True)` keeps the float32 vectors on disk. On
  GloVe-6B 300-d, private memory drops from 537 MiB to 197 MiB at equal recall.
- `quantization` / `rerank` options on `strata.Index`, `strata.Collection.create`, the HTTP client and
  `POST /collections`.
- Benchmarks: `bench_ingest` (queries during inserts), `bench_filter` (filter selectivity sweep), `bench_memory.py`
  (resident memory), and GloVe-6B 100-d / 300-d datasets plus FAISS `IndexHNSWSQ` in `bench_ann.py`.

### Changed
- **Inserts no longer block searches.** `add()` holds the index lock exclusively only to allocate nodes and store
  vectors, then links the new nodes under the shared lock while queries continue. `Collection.upsert` no longer
  holds the collection lock across the WAL write and the index insert. During a 100k-vector insert, a query thread now
  gets p99 0.46 ms, where it was previously blocked for the whole 17 s.
- **Cost-based filtered search**: allow-filters use brute force when |allowed| ≤ sqrt(M · ef · n), not only below a
  fixed 2,048 labels. This is 2.5–25× faster at 2–5% selectivity, with recall 1.0.
- The label → node map is an open-addressing hash table (16 bytes per slot, prefetched lookups).
- The index file format is v2 for quantized indexes. v1 files still load.

### Fixed
- Queries with `ef` ≥ the number of live vectors now scan exactly. Small parallel builds could occasionally leave a
  node without inbound links, which a graph walk can never return.
