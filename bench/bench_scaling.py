"""Parallel build scaling: build time vs. thread count on a SIFT subset.

    python bench/bench_scaling.py --limit 200000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

import strata

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200_000)
    ap.add_argument("--threads", default="1,2,4,8,12,15")
    args = ap.parse_args()

    with h5py.File(DATA / "sift-128-euclidean.hdf5", "r") as f:
        x = np.asarray(f["train"][: args.limit], dtype=np.float32)
        q = np.asarray(f["test"][:1000], dtype=np.float32)

    import faiss

    exact = faiss.IndexFlatL2(x.shape[1])
    exact.add(x)
    truth = exact.search(q, 10)[1]

    rows = []
    for t in [int(s) for s in args.threads.split(",")]:
        index = strata.Index(128, "l2", M=16, ef_construction=200)
        t0 = time.perf_counter()
        index.add(x, num_threads=t)
        secs = time.perf_counter() - t0
        found = index.search(q, k=10, ef=64)[0]
        recall = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, truth)]))
        rows.append({"threads": t, "seconds": secs, "inserts_per_sec": len(x) / secs, "recall_ef64": recall})
        print(f"threads={t:2d}  build={secs:6.1f}s  {len(x) / secs:8.0f} inserts/s  recall@10(ef=64)={recall:.4f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    out = {"n": len(x), "dim": 128, "M": 16, "ef_construction": 200, "rows": rows}
    (RESULTS / "scaling.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
