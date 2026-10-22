#!/usr/bin/env python3
"""Summarise a benchmark v2 run and draw its figure.

Usage:  python3 scripts/v2/analyze_v2.py results/v2/<run-id>

Reads <run>/<mode>/rep*/<case>.json written by `vllm bench serve --save-detailed`
and writes summary.csv, summary.md and v2_summary.png / .svg into <run>.

Per-request metrics (TTFT, TPOT, ITL) are pooled across repetitions for the
table; bars in the figure show the median of the per-repetition values with
min-max whiskers, so the spread between runs stays visible.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

MODES = [("timeslice", "Time-slicing"), ("mps", "MPS"), ("mig", "MIG")]
CASES = {
    "B_idle": ("idle_B", "Quiet tenant B, neighbour idle"),
    "B_busy": ("contended_B", "Quiet tenant B, neighbour spiking"),
    "A_alone": ("spikealone_A", "Noisy tenant A, spiking alone"),
    "A_busy": ("contended_A", "Noisy tenant A, spiking next to B"),
}
METRICS = ["ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99", "tpot_std", "itl_p99", "throughput", "failed"]


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def per_request(result):
    """Per-request TTFT and TPOT (ms) and all inter-token gaps (ms) of successful requests."""
    errors = result.get("errors") or [""] * len(result["ttfts"])
    ttft, tpot, itl = [], [], []
    for i, err in enumerate(errors):
        n_out = result["output_lens"][i]
        if err or n_out == 0:
            continue
        ttft.append(result["ttfts"][i] * 1e3)
        gaps = result["itls"][i]
        if n_out > 1 and gaps:
            tpot.append(sum(gaps) / (n_out - 1) * 1e3)   # same definition as vLLM's TPOT
            itl.extend(g * 1e3 for g in gaps)
    return np.array(ttft), np.array(tpot), np.array(itl)


def metrics(ttft, tpot, itl, throughput, failed):
    return {
        "ttft_p50": pct(ttft, 50), "ttft_p99": pct(ttft, 99),
        "tpot_p50": pct(tpot, 50), "tpot_p99": pct(tpot, 99),
        "tpot_std": float(np.std(tpot)) if len(tpot) else float("nan"),
        "itl_p99": pct(itl, 99),
        "throughput": throughput, "failed": failed,
    }


def collect(run):
    out = {}
    for mode, _ in MODES:
        for case, (stem, _) in CASES.items():
            files = sorted((run / mode).glob(f"rep*/{stem}.json"))
            if not files:
                continue
            reps, pool = [], ([], [], [])
            for f in files:
                r = json.loads(f.read_text())
                arrays = per_request(r)
                reps.append(metrics(*arrays, r.get("request_throughput", float("nan")), r.get("failed", 0)))
                for acc, arr in zip(pool, arrays):
                    acc.append(arr)
            pooled = metrics(*(np.concatenate(a) for a in pool),
                             float(np.median([x["throughput"] for x in reps])),
                             int(sum(x["failed"] for x in reps)))
            out[(mode, case)] = {"reps": reps, "pooled": pooled, "n": len(files)}
    return out


def read_env(run):
    env = {}
    path = run / "env.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.partition(":")
            env[key.strip()] = value.strip()
    return env


def mode_label(mode, label, env):
    return f"MIG {env.get('mig_profile', '')}".strip() if mode == "mig" else label


def write_tables(run, res, env):
    rows = []
    for mode, label in MODES:
        for case, (_, desc) in CASES.items():
            if (mode, case) not in res:
                continue
            item = res[(mode, case)]
            p99s = [x["ttft_p99"] for x in item["reps"]]
            thr = [x["throughput"] for x in item["reps"]]
            rows.append({
                "mode": mode_label(mode, label, env), "case": desc, "runs": item["n"],
                **{k: round(v, 2) for k, v in item["pooled"].items()},
                "ttft_p99_range": f"{min(p99s):.1f}-{max(p99s):.1f}",
                "throughput_range": f"{min(thr):.2f}-{max(thr):.2f}",
            })
    with open(run / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    header = ("| Mode | Case | Runs | p50 TTFT (ms) | p99 TTFT (ms) | p50 TPOT (ms) | p99 TPOT (ms) "
              "| TPOT std (ms) | p99 ITL (ms) | Throughput (req/s) | Failed |")
    lines = [f"# Benchmark v2 results ({env.get('date', '')})", "",
             f"GPU: {env.get('gpu', '?')} | vLLM {env.get('vllm_client', '?')} | {env.get('model', '?')} | "
             f"{env.get('tokens_in_out', '?')} tokens in/out | {env.get('reps', '?')} runs per case", "",
             "Latencies pool every request across runs; throughput is the median of the runs "
             "(min-max range in summary.csv).", "",
             header, "|" + " --- |" * 11]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['case']} | {r['runs']} | {r['ttft_p50']:.1f} | {r['ttft_p99']:.1f} | "
                     f"{r['tpot_p50']:.1f} | {r['tpot_p99']:.1f} | {r['tpot_std']:.2f} | {r['itl_p99']:.1f} | "
                     f"{r['throughput']:.2f} | {r['failed']} |")
    lines += ["", "## Headline numbers", ""] + [f"- {s}" for s in headlines(res, env)]
    (run / "summary.md").write_text("\n".join(lines) + "\n")


def headlines(res, env):
    out = []
    for mode, label in MODES:
        name = mode_label(mode, label, env)
        if (mode, "B_idle") in res and (mode, "B_busy") in res:
            a, b = res[(mode, "B_idle")]["pooled"], res[(mode, "B_busy")]["pooled"]
            out.append(f"{name}: when A spikes, B's p99 TTFT goes {a['ttft_p99']:.1f} -> {b['ttft_p99']:.1f} ms "
                       f"({b['ttft_p99'] / a['ttft_p99']:.2f}x) and its median TPOT {a['tpot_p50']:.1f} -> "
                       f"{b['tpot_p50']:.1f} ms.")
        if (mode, "A_alone") in res and (mode, "A_busy") in res:
            a, b = res[(mode, "A_alone")]["pooled"], res[(mode, "A_busy")]["pooled"]
            out.append(f"{name}: A's throughput is {a['throughput']:.1f} req/s alone and {b['throughput']:.1f} "
                       f"req/s next to B ({b['throughput'] / a['throughput'] - 1:+.0%}).")
    if ("timeslice", "B_idle") in res and ("mig", "B_idle") in res:
        ts, mg = res[("timeslice", "B_idle")]["pooled"], res[("mig", "B_idle")]["pooled"]
        out.append(f"Cost of isolation with the neighbour idle: B's median TPOT is {ts['tpot_p50']:.1f} ms on the "
                   f"shared GPU and {mg['tpot_p50']:.1f} ms on its MIG slice ({mg['tpot_p50'] / ts['tpot_p50']:.2f}x).")
    if ("timeslice", "A_alone") in res and ("mig", "A_alone") in res:
        ts, mg = res[("timeslice", "A_alone")]["pooled"], res[("mig", "A_alone")]["pooled"]
        out.append(f"A's capacity alone: {ts['throughput']:.1f} req/s on the whole GPU vs {mg['throughput']:.1f} "
                   f"req/s on its MIG slice.")
    return out


def draw(run, res, env):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quiet, busy = "#0072B2", "#E69F00"            # Okabe-Ito blue / orange, colour-blind safe
    panels = [
        ("Quiet tenant B: p99 TTFT (ms)", ("B_idle", "B_busy"), "ttft_p99", ("A idle", "A spiking")),
        ("Quiet tenant B: median TPOT (ms)", ("B_idle", "B_busy"), "tpot_p50", ("A idle", "A spiking")),
        ("Noisy tenant A: throughput (req/s)", ("A_alone", "A_busy"), "throughput", ("B idle", "B steady")),
    ]
    modes = [(m, mode_label(m, lab, env)) for m, lab in MODES if any((m, c) in res for c in CASES)]
    x = np.arange(len(modes))
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, (title, cases, key, legend) in zip(axes, panels):
        for j, (case, colour, name) in enumerate(zip(cases, (quiet, busy), legend)):
            med, lo, hi = [], [], []
            for m, _ in modes:
                vals = [r[key] for r in res[(m, case)]["reps"]] if (m, case) in res else [np.nan]
                mid = float(np.median(vals))
                med.append(mid)
                lo.append(mid - float(np.min(vals)))
                hi.append(float(np.max(vals)) - mid)
            ax.bar(x + (j - 0.5) * width, med, width, color=colour, label=name,
                   yerr=[lo, hi], capsize=3, ecolor="#333333", error_kw={"linewidth": 1})
        ax.set_xticks(x, [label for _, label in modes])
        ax.set_title(title, fontsize=11, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
    gpu = env.get("gpu", "").split(",")[0]
    fig.suptitle(f"Two {env.get('model', 'vLLM')} instances on one {gpu}: "
                 f"median of {env.get('reps', '?')} runs, whiskers = min-max", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(run / f"v2_summary.{ext}", dpi=200)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run = Path(sys.argv[1])
    res = collect(run)
    if not res:
        sys.exit(f"No results found under {run}")
    env = read_env(run)
    write_tables(run, res, env)
    draw(run, res, env)
    print((run / "summary.md").read_text())


if __name__ == "__main__":
    main()
