"""Resident memory of a loaded index: float32 vs SQ8 (+ mmap'd re-rank vectors) vs SQ8 without rerank.

Each configuration is built once and saved; then a fresh process loads it, runs the queries and
reports peak RSS, recall@10 and single-thread QPS. A fresh process per configuration keeps
allocations from earlier runs out of the RSS number.

    python bench/bench_memory.py --dataset glove-wiki-300
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_ann import RESULTS, load, recall_at_10  # noqa: E402

CONFIGS = {
    "float32": dict(quantization="none", rerank=True, mmap=False),
    "sq8+rerank": dict(quantization="sq8", rerank=True, mmap=False),
    "sq8+rerank (mmap vectors)": dict(quantization="sq8", rerank=True, mmap=True),
    "sq8 (no rerank)": dict(quantization="sq8", rerank=False, mmap=False),
}


def rss_mib(kind: str = "RssAnon") -> float:
    """RssAnon: private memory that cannot be reclaimed. RssFile: mapped file pages, which the OS
    can drop under memory pressure (that is where mmap'd re-rank vectors live). Linux only."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(kind + ":"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("needs /proc/self/status (Linux)")


def child(index_path: str, queries_path: str, truth_path: str, mmap: bool, ef: int) -> None:
    import strata

    q, truth = np.load(queries_path), np.load(truth_path)
    before, file_before = rss_mib(), rss_mib("RssFile")
    index = strata.Index.load(index_path, mmap_vectors=mmap)
    loaded = rss_mib()
    index.search(q[:200], k=10, ef=ef, num_threads=1)
    t0 = time.perf_counter()
    found = index.search(q, k=10, ef=ef, num_threads=1)[0]
    qps = len(q) / (time.perf_counter() - t0)
    print(json.dumps({"rss_index_mib": loaded - before, "rss_after_queries_mib": rss_mib() - before,
                      "rss_file_after_queries_mib": rss_mib("RssFile") - file_before,
                      "recall": recall_at_10(found, truth), "qps": qps,
                      "file_mib": os.path.getsize(index_path) / 2**20}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="glove-wiki-300")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--ef", type=int, default=128)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--workdir", default="/tmp/strata-membench")
    ap.add_argument("--child", nargs=5)
    args = ap.parse_args()
    if args.child:
        idx, qp, tp, mmap, ef = args.child
        return child(idx, qp, tp, mmap == "1", int(ef))

    import strata

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    train, test, truth, metric = load(args.dataset, None, args.queries)
    np.save(work / "q.npy", test)
    np.save(work / "t.npy", truth)
    out = {"dataset": args.dataset, "n": len(train), "dim": train.shape[1], "ef": args.ef,
           "float32_vectors_mib": train.nbytes / 2**20, "rows": {}}
    built = {}
    for name, cfg in CONFIGS.items():
        key = (cfg["quantization"], cfg["rerank"])
        if key not in built:
            index = strata.Index(train.shape[1], metric, M=16, ef_construction=200, capacity=len(train),
                                 quantization=cfg["quantization"], rerank=cfg["rerank"])
            index.add(train, num_threads=args.threads)
            path = work / f"{cfg['quantization']}-{int(cfg['rerank'])}.bin"
            index.save(str(path))
            built[key] = path
            del index
        res = subprocess.run([sys.executable, __file__, "--child", str(built[key]), str(work / "q.npy"),
                              str(work / "t.npy"), "1" if cfg["mmap"] else "0", str(args.ef)],
                             capture_output=True, text=True, check=True)
        row = json.loads(res.stdout.strip().splitlines()[-1])
        out["rows"][name] = row
        print(f"{name:28s} anon RSS {row['rss_after_queries_mib']:7.0f} MiB  mapped {row['rss_file_after_queries_mib']:6.0f} MiB  "
              f"file {row['file_mib']:6.0f} MiB  "
              f"recall@10 {row['recall']:.4f}  {row['qps']:6.0f} QPS", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"memory-{args.dataset}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
