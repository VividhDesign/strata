"""End-to-end HTTP benchmark of strata-server.

Loads SIFT vectors through the REST API (measuring ingest throughput), then fires query
requests from N concurrent keep-alive connections and reports throughput and latency
percentiles. Start the server first:

    ./build/strata-server --data-dir /tmp/strata-load --port 8080
    python bench/load_test.py --n 200000
"""

from __future__ import annotations

import argparse
import http.client
import json
import threading
import time
from pathlib import Path

import h5py
import numpy as np

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"


def request(conn: http.client.HTTPConnection, method: str, path: str, body: bytes | None = None):
    conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    if resp.status >= 300:
        raise RuntimeError(f"{resp.status}: {data[:200]!r}")
    return json.loads(data)


def run_queries(host, port, bodies, concurrency, total):
    latencies = []
    lock = threading.Lock()
    counter = iter(range(total))

    def worker():
        conn = http.client.HTTPConnection(host, port)
        local = []
        while True:
            with lock:
                i = next(counter, None)
            if i is None:
                break
            t0 = time.perf_counter()
            request(conn, "POST", "/collections/sift/query", bodies[i % len(bodies)])
            local.append(time.perf_counter() - t0)
        with lock:
            latencies.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    lat = np.array(latencies) * 1000
    return {"concurrency": concurrency, "requests": total, "qps": total / wall,
            "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--requests", type=int, default=20_000)
    ap.add_argument("--ef", type=int, default=64)
    args = ap.parse_args()

    with h5py.File(DATA / "sift-128-euclidean.hdf5", "r") as f:
        x = np.asarray(f["train"][: args.n], dtype=np.float32)
        q = np.asarray(f["test"][:2000], dtype=np.float32)

    conn = http.client.HTTPConnection(args.host, args.port)
    try:
        request(conn, "DELETE", "/collections/sift")
    except RuntimeError:
        pass
    request(conn, "POST", "/collections", json.dumps({"name": "sift", "dim": 128, "metric": "l2"}).encode())

    t0 = time.perf_counter()
    for start in range(0, len(x), args.batch):
        chunk = x[start:start + args.batch]
        body = {"ids": list(range(start, start + len(chunk))), "vectors": chunk.tolist(),
                "metadata": [{"shard": int(i % 10)} for i in range(start, start + len(chunk))]}
        request(conn, "POST", "/collections/sift/upsert", json.dumps(body).encode())
    ingest_s = time.perf_counter() - t0
    print(f"ingested {len(x)} vectors over HTTP in {ingest_s:.1f}s ({len(x) / ingest_s:,.0f} vectors/s)", flush=True)

    bodies = [json.dumps({"vector": v.tolist(), "k": 10, "ef": args.ef, "include_metadata": False}).encode() for v in q]
    filtered = [json.dumps({"vector": v.tolist(), "k": 10, "ef": args.ef, "filter": {"shard": 3},
                            "include_metadata": False}).encode() for v in q]
    run_queries(args.host, args.port, bodies, 8, 2000)  # warm-up

    rows = []
    for c in (1, 8, 32):
        r = run_queries(args.host, args.port, bodies, c, args.requests)
        rows.append(r)
        print(f"concurrency={c:3d}  {r['qps']:8,.0f} req/s  p50={r['p50_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms  p99={r['p99_ms']:.2f}ms", flush=True)
    rf = run_queries(args.host, args.port, filtered, 8, args.requests // 2)
    print(f"filtered (shard=3, 10% selectivity), concurrency=8: {rf['qps']:,.0f} req/s  p99={rf['p99_ms']:.2f}ms")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "server.json").write_text(json.dumps({
        "n": len(x), "ef": args.ef, "ingest_vectors_per_sec": len(x) / ingest_s, "query": rows,
        "filtered_query": rf}, indent=2))


if __name__ == "__main__":
    main()
