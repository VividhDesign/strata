"""Recall vs. throughput benchmark: Strata vs. hnswlib vs. FAISS (IndexHNSWFlat).

All three libraries get identical graph parameters (M, ef_construction). Builds use all
cores; queries run single-threaded (the ann-benchmarks convention) while sweeping ef.

    python bench/bench_ann.py --dataset sift            # 1M x 128, L2
    python bench/bench_ann.py --dataset glove           # 1.18M x 100, angular
    python bench/bench_ann.py --dataset sift --limit 100000 --libs strata
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

import strata

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
DATASETS = {
    "sift": ("sift-128-euclidean.hdf5", "l2"),
    "glove": ("glove-100-angular.hdf5", "cosine"),
}
EF_SWEEP = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def cpu_name() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor()


def load(name: str, limit: int | None, n_queries: int):
    fname, metric = DATASETS[name]
    with h5py.File(DATA / fname, "r") as f:
        train = np.asarray(f["train"][: limit or None], dtype=np.float32)
        test = np.asarray(f["test"][:n_queries], dtype=np.float32)
        truth = np.asarray(f["neighbors"][:n_queries, :10])
    if metric == "cosine":
        train /= np.linalg.norm(train, axis=1, keepdims=True) + 1e-12
        test /= np.linalg.norm(test, axis=1, keepdims=True) + 1e-12
    if limit:  # ground truth must be recomputed for a subset
        import faiss

        exact = faiss.IndexFlatL2(train.shape[1]) if metric == "l2" else faiss.IndexFlatIP(train.shape[1])
        exact.add(train)
        truth = exact.search(test, 10)[1]
    return train, test, truth, metric


def recall_at_10(found: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(set(f[:10]) & set(t[:10])) / 10.0 for f, t in zip(found, truth)]))


class StrataLib:
    name = "strata"

    def __init__(self, dim, metric, M, efc, heuristic=True):
        self.index = strata.Index(dim, metric, M=M, ef_construction=efc, use_heuristic=heuristic)

    def build(self, x, threads):
        self.index.add(x, num_threads=threads)

    def query(self, q, k, ef):
        return self.index.search(q, k=k, ef=ef, num_threads=1)[0]

    def batch_query(self, q, k, ef, threads):
        return self.index.search(q, k=k, ef=ef, num_threads=threads)[0]

    def size_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "i")
            self.index.save(p)
            return os.path.getsize(p)


class HnswlibLib:
    name = "hnswlib"

    def __init__(self, dim, metric, M, efc):
        import hnswlib

        self.index = hnswlib.Index(space="l2" if metric == "l2" else "cosine", dim=dim)
        self.M, self.efc = M, efc

    def build(self, x, threads):
        self.index.init_index(max_elements=len(x), M=self.M, ef_construction=self.efc, random_seed=42)
        self.index.set_num_threads(threads)
        self.index.add_items(x, np.arange(len(x)))

    def query(self, q, k, ef):
        self.index.set_ef(ef)
        self.index.set_num_threads(1)
        return self.index.knn_query(q, k=k)[0]

    def batch_query(self, q, k, ef, threads):
        self.index.set_ef(ef)
        self.index.set_num_threads(threads)
        return self.index.knn_query(q, k=k)[0]

    def size_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "i")
            self.index.save_index(p)
            return os.path.getsize(p)


class FaissLib:
    name = "faiss"

    def __init__(self, dim, metric, M, efc):
        import faiss

        self.faiss = faiss
        m = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        self.index = faiss.IndexHNSWFlat(dim, M, m)
        self.index.hnsw.efConstruction = efc

    def build(self, x, threads):
        self.faiss.omp_set_num_threads(threads)
        self.index.add(x)

    def query(self, q, k, ef):
        self.faiss.omp_set_num_threads(1)
        self.index.hnsw.efSearch = ef
        return self.index.search(q, k)[1]

    def batch_query(self, q, k, ef, threads):
        self.faiss.omp_set_num_threads(threads)
        self.index.hnsw.efSearch = ef
        return self.index.search(q, k)[1]

    def size_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "i")
            self.faiss.write_index(self.index, p)
            return os.path.getsize(p)


def make(lib, dim, metric, M, efc):
    if lib == "strata":
        return StrataLib(dim, metric, M, efc)
    if lib == "strata-noheuristic":
        obj = StrataLib(dim, metric, M, efc, heuristic=False)
        obj.name = lib
        return obj
    if lib == "hnswlib":
        return HnswlibLib(dim, metric, M, efc)
    if lib == "faiss":
        return FaissLib(dim, metric, M, efc)
    raise ValueError(lib)


def timed_queries(lib, q, ef, repeats=2):
    lib.query(q[:200], 10, ef)  # warm-up
    best = float("inf")
    found = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        found = lib.query(q, 10, ef)
        best = min(best, time.perf_counter() - t0)
    return found, len(q) / best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="sift")
    ap.add_argument("--libs", default="strata,hnswlib,faiss")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N base vectors")
    ap.add_argument("--queries", type=int, default=10000)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    train, test, truth, metric = load(args.dataset, args.limit, args.queries)
    print(f"{args.dataset}: base={train.shape} queries={test.shape} metric={metric}", flush=True)
    out = {
        "dataset": args.dataset, "n": len(train), "dim": train.shape[1], "metric": metric,
        "queries": len(test), "M": args.M, "ef_construction": args.efc, "build_threads": args.threads,
        "cpu": cpu_name(), "simd": strata.simd_backend(), "libs": {},
    }
    for name in args.libs.split(","):
        lib = make(name, train.shape[1], metric, args.M, args.efc)
        t0 = time.perf_counter()
        lib.build(train, args.threads)
        build_s = time.perf_counter() - t0
        size = lib.size_bytes()
        print(f"[{name}] build {build_s:.1f}s, index {size / 2**20:.0f} MiB", flush=True)
        curve = []
        for ef in EF_SWEEP:
            found, qps = timed_queries(lib, test, ef)
            r = recall_at_10(found, truth)
            curve.append({"ef": ef, "recall": r, "qps": qps})
            print(f"  ef={ef:4d}  recall@10={r:.4f}  qps={qps:9.0f}", flush=True)
        t0 = time.perf_counter()
        found = lib.batch_query(test, 10, 64, args.threads)
        batch_qps = len(test) / (time.perf_counter() - t0)
        print(f"  all-core batch @ef=64: {batch_qps:.0f} qps (recall {recall_at_10(found, truth):.4f})", flush=True)
        out["libs"][name] = {"build_seconds": build_s, "index_bytes": size, "curve": curve,
                             "batch_qps_ef64": batch_qps}
        del lib

    RESULTS.mkdir(exist_ok=True)
    suffix = f"-{args.tag}" if args.tag else ""
    path = RESULTS / f"{args.dataset}{'-' + str(args.limit) if args.limit else ''}{suffix}.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
