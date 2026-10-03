"""Renders benchmark charts (light + dark PNG variants) from bench/results/*.json.

    python bench/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).parent / "results"
OUT = Path(__file__).parent.parent / "docs" / "img"

# Validated categorical slots 1-3 (blue, orange, aqua) for each mode.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e1",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#33332f",
             "series": ["#3987e5", "#d95926", "#199e70"]},
}
LIB_LABEL = {"strata": "Strata (this repo)", "hnswlib": "hnswlib", "faiss": "FAISS HNSWFlat",
             "strata-noheuristic": "Strata, no heuristic"}


def style(ax, fig, t):
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(which="both", colors=t["muted"], labelsize=9, length=0)
    ax.grid(True, color=t["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["muted"])
    ax.yaxis.label.set_color(t["muted"])


def end_label(ax, x, y, text, t):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=t["text"])


def spread_labels(ys: list[float], min_ratio: float = 1.18) -> list[float]:
    """Nudges label y positions (log scale) apart so overlapping curves get readable labels."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] < out[a] * min_ratio:
            out[b] = out[a] * min_ratio
    return out


def log_ticks(ax, lo: float, hi: float):
    ticks = [m * 10 ** e for e in range(2, 7) for m in (1, 2, 5) if lo <= m * 10 ** e <= hi]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def recall_qps(dataset: str, libs: dict, mode: str, title: str, fname: str, min_recall: float):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    style(ax, fig, t)
    starts = []
    for i, (name, res) in enumerate(libs.items()):
        pts = [p for p in res["curve"] if p["recall"] >= min_recall]
        xs, ys = [p["recall"] for p in pts], [p["qps"] for p in pts]
        ax.plot(xs, ys, color=t["series"][i], linewidth=2, marker="o", markersize=5,
                markeredgecolor=t["surface"], markeredgewidth=1.5, label=LIB_LABEL.get(name, name), zorder=3)
        starts.append((xs[0], ys[0], LIB_LABEL.get(name, name)))
    for (x, _, text), y in zip(starts, spread_labels([s[1] for s in starts])):
        end_label(ax, x, y, text, t)
    ax.set_yscale("log")
    all_qps = [p["qps"] for r in libs.values() for p in r["curve"] if p["recall"] >= min_recall]
    log_ticks(ax, min(all_qps) * 0.8, max(all_qps) * 1.6)
    ax.set_xlabel("Recall@10  (higher is better →)")
    ax.set_ylabel("Queries / second, 1 thread  (log)")
    ax.set_title(title, loc="left", color=t["text"], fontsize=11, pad=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    recalls = [p["recall"] for r in libs.values() for p in r["curve"] if p["recall"] >= min_recall]
    ax.set_xlim(min(recalls) - 0.005, min(max(recalls) + 0.06, 1.005))
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}-{mode}.png", facecolor=t["surface"])
    plt.close(fig)


def scaling(data: dict, mode: str):
    t = THEMES[mode]
    rows = data["rows"]
    base = rows[0]["seconds"]
    threads = [r["threads"] for r in rows]
    speedup = [base / r["seconds"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    style(ax, fig, t)
    ax.plot(threads, threads, color=t["muted"], linewidth=1, zorder=2)
    end_label(ax, threads[-1], threads[-1], "linear", t)
    ax.plot(threads, speedup, color=t["series"][0], linewidth=2, marker="o", markersize=5,
            markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3)
    end_label(ax, threads[-1], speedup[-1], f"{speedup[-1]:.1f}×", t)
    ax.set_xlabel("Build threads")
    ax.set_ylabel("Speedup vs. 1 thread")
    ax.set_title(f"Parallel index build, {data['n'] // 1000}k SIFT vectors", loc="left", color=t["text"], fontsize=11, pad=12)
    ax.set_xticks(threads)
    fig.tight_layout()
    fig.savefig(OUT / f"build-scaling-{mode}.png", facecolor=t["surface"])
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        for ds, title, lo in [("sift", "SIFT1M (1M × 128, L2)", 0.8), ("glove", "GloVe-100 (1.18M × 100, angular)", 0.6)]:
            p = RESULTS / f"{ds}.json"
            if p.exists():
                res = json.loads(p.read_text())
                recall_qps(ds, res["libs"], mode, f"{title}: recall vs. throughput", f"{ds}-recall-qps", lo)
        ab = RESULTS / "glove-ablation.json"
        if ab.exists() and (RESULTS / "glove.json").exists():
            libs = {"strata": json.loads((RESULTS / "glove.json").read_text())["libs"]["strata"],
                    "strata-noheuristic": json.loads(ab.read_text())["libs"]["strata-noheuristic"]}
            recall_qps("glove", libs, mode, "Neighbour-selection heuristic ablation (GloVe-100)", "ablation-heuristic", 0.5)
        sc = RESULTS / "scaling.json"
        if sc.exists():
            scaling(json.loads(sc.read_text()), mode)
    print("charts written to", OUT)


if __name__ == "__main__":
    main()
