"""Prints Markdown summary tables from bench/results/*.json (used to write the README).

QPS at a recall target is linearly interpolated between the two sweep points that bracket
the target, so libraries are compared at equal recall rather than equal ef.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
NAMES = {"strata": "**Strata**", "hnswlib": "hnswlib", "faiss": "FAISS HNSWFlat"}


def qps_at(curve: list[dict], target: float) -> float | None:
    pts = sorted(curve, key=lambda p: p["recall"])
    for a, b in zip(pts, pts[1:]):
        if a["recall"] <= target <= b["recall"]:
            if b["recall"] == a["recall"]:
                return min(a["qps"], b["qps"])
            w = (target - a["recall"]) / (b["recall"] - a["recall"])
            return a["qps"] + w * (b["qps"] - a["qps"])
    return None


def ann_table(name: str, targets: tuple[float, ...]) -> str:
    res = json.loads((RESULTS / f"{name}.json").read_text())
    head = "| Library | Build (s) | Index (MiB) | " + " | ".join(f"QPS @ R≥{t}" for t in targets) + " | All-core QPS (ef=64) |"
    lines = [head, "|---|---:|---:|" + "---:|" * len(targets) + "---:|"]
    for lib, r in res["libs"].items():
        cells = [f"{q:,.0f}" if (q := qps_at(r["curve"], t)) else "—" for t in targets]
        lines.append(f"| {NAMES.get(lib, lib)} | {r['build_seconds']:.1f} | {r['index_bytes'] / 2**20:.0f} | "
                     + " | ".join(cells) + f" | {r['batch_qps_ef64']:,.0f} |")
    return (f"{res['dataset']}: n={res['n']:,}, dim={res['dim']}, {res['metric']}, M={res['M']}, "
            f"efC={res['ef_construction']}, {res['queries']:,} queries, {res['cpu']}\n\n" + "\n".join(lines))


def main():
    print(ann_table("sift", (0.9, 0.95, 0.99)))
    print()
    print(ann_table("glove", (0.7, 0.8, 0.9)))
    ab = json.loads((RESULTS / "glove-ablation.json").read_text())["libs"]["strata-noheuristic"]["curve"]
    base = json.loads((RESULTS / "glove.json").read_text())["libs"]["strata"]["curve"]
    print("\n| ef | recall@10 with heuristic | without | ")
    print("|---:|---:|---:|")
    for a, b in zip(base, ab):
        if a["ef"] in (32, 64, 128, 256, 512):
            print(f"| {a['ef']} | {a['recall']:.3f} | {b['recall']:.3f} |")
    sc = json.loads((RESULTS / "scaling.json").read_text())
    base_s = sc["rows"][0]["seconds"]
    print("\n| Threads | Build (s) | Inserts/s | Speedup | Recall@10 (ef=64) |")
    print("|---:|---:|---:|---:|---:|")
    for r in sc["rows"]:
        print(f"| {r['threads']} | {r['seconds']:.1f} | {r['inserts_per_sec']:,.0f} | {base_s / r['seconds']:.1f}× | {r['recall_ef64']:.4f} |")
    sv = json.loads((RESULTS / "server.json").read_text())
    print(f"\nserver: ingest {sv['ingest_vectors_per_sec']:,.0f} vectors/s")
    for q in sv["query"]:
        print(f"| {q['concurrency']} | {q['qps']:,.0f} | {q['p50_ms']:.2f} | {q['p95_ms']:.2f} | {q['p99_ms']:.2f} |")
    f = sv["filtered_query"]
    print(f"filtered: {f['qps']:,.0f} req/s p50 {f['p50_ms']:.2f} p99 {f['p99_ms']:.2f}")


if __name__ == "__main__":
    main()
