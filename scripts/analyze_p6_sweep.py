#!/usr/bin/env python3
"""Consume the SALAD-v2 training-size sweep and answer P6, P7, P24 and P8 at once.

The sweep (scripts/p6_sweep.py, running on the Mac Mini) writes one JSON per cell
with strict_macro_f1, off_vocab_rate, train_seconds, eval_seconds, n_train, arm
and seed. That is enough to remediate four held papers from a single run:

  P6  training-size scaling of label compliance, with per-seed stability
  P7  cost per F1 point -- from the same runs, no extra compute
  P24 entropy -> difficulty datapoints for the two SALAD-v2 arms
  P8  the same two arms as new points in the N=8 entropy regression

Cost reporting follows the decision of 2026-07-30: wall-clock hours on named
hardware is the primary, factual unit. Dollars are reported twice -- measured
electricity, and an imputed cloud-rental scenario -- with both assumptions
stated, rather than quoting a rental price that was never paid.
"""
import json, glob, os, argparse, math
from collections import defaultdict, Counter

# --- cost assumptions, stated rather than buried -------------------------------
# Apple M4 Mac Mini under sustained MLX load. Apple rates the M4 Mac mini at 65 W
# maximum continuous power; we use that as an upper bound rather than a typical
# figure, so the electricity cost is an over-estimate rather than a flattering one.
WATTS = 65.0
# Thailand residential average, 2026. Stated so a reader can substitute their own.
THB_PER_KWH = 4.20
THB_PER_USD = 34.0
# Imputed cloud scenario. NOT what was paid -- an equivalence assumption for
# readers who price work in rental terms. Rate is for a small cloud GPU instance.
CLOUD_USD_PER_HOUR = 0.30


def load(sweep_dir):
    cells = []
    for f in sorted(glob.glob(os.path.join(sweep_dir, "*.json"))):
        if "preds" in os.path.basename(f):
            continue
        d = json.load(open(f))
        if "strict_macro_f1" not in d or "n_train" not in d:
            continue
        d["_file"] = os.path.basename(f)
        cells.append(d)
    return cells


def p6_scaling(cells):
    """Per-arm, per-size: mean/std across seeds, and whether the ordering is
    seed-stable. P6's original finding died on the seed mean; report both."""
    by = defaultdict(list)
    for c in cells:
        by[(c["arm"], c["n_train"])].append(c)

    print("\n" + "=" * 78)
    print("P6 — training-size scaling of label compliance (SALAD-v2, leak-screened)")
    print("=" * 78)
    out = {}
    for arm in sorted({a for a, _ in by}):
        sizes = sorted({n for a, n in by if a == arm})
        print(f"\n  {arm}")
        print(f"    {'n_train':>8} {'seeds':>6} {'strict macro-F1':>22} {'off-vocab':>11}")
        rows = []
        for n in sizes:
            cs = by[(arm, n)]
            f1 = [c["strict_macro_f1"] for c in cs]
            ov = [c.get("off_vocab_rate", 0.0) for c in cs]
            mean = sum(f1) / len(f1)
            std = (sum((x - mean) ** 2 for x in f1) / len(f1)) ** 0.5 if len(f1) > 1 else 0.0
            print(f"    {n:>8} {len(f1):>6}   {mean:.4f} +/- {std:.4f}   {sum(ov)/len(ov):>10.4f}")
            rows.append({"n_train": n, "seeds": len(f1), "mean": mean, "std": std,
                         "per_seed": {c["seed"]: c["strict_macro_f1"] for c in cs},
                         "off_vocab_mean": sum(ov) / len(ov)})
        # peak location, and whether every seed agrees on it
        if rows and all(r["seeds"] > 1 for r in rows):
            peak_mean = max(rows, key=lambda r: r["mean"])["n_train"]
            seeds = sorted(set().union(*[set(r["per_seed"]) for r in rows]))
            per_seed_peak = {}
            for s in seeds:
                cand = [(r["per_seed"][s], r["n_train"]) for r in rows if s in r["per_seed"]]
                if cand:
                    per_seed_peak[s] = max(cand)[1]
            # A raw argmax calls a 0.0008 lead a different peak. Compare each seed's
            # best against its own value at the mean-peak, and treat anything inside
            # that size's cross-seed sd as a tie -- otherwise the verdict flips on noise.
            tol = next((r["std"] for r in rows if r["n_train"] == peak_mean), 0.0)
            ties = {}
            for s, pk in per_seed_peak.items():
                at_peak = next((r["per_seed"].get(s) for r in rows if r["n_train"] == peak_mean), None)
                best = max(r["per_seed"][s] for r in rows if s in r["per_seed"])
                ties[s] = (pk if best - (at_peak or 0) > tol else f"{peak_mean} (tied, +{best-(at_peak or 0):.4f})")
            unanimous = len({str(v).split()[0] for v in ties.values()}) == 1
            print(f"    peak on the seed mean: n={peak_mean}   (tolerance = that size's cross-seed sd, {tol:.4f})")
            print(f"    per-seed peaks after tie-breaking: {ties}")
            print(f"    {'UNANIMOUS (ties counted as agreement)' if unanimous else 'NOT unanimous even allowing ties'}")
            out[arm] = {"rows": rows, "peak_mean": peak_mean,
                        "per_seed_peak": per_seed_peak, "unanimous": unanimous}
        else:
            out[arm] = {"rows": rows, "note": "single seed — no stability claim available"}
    return out


def p7_cost(cells):
    """Cost per F1 point, in hours first and dollars twice."""
    print("\n" + "=" * 78)
    print("P7 — cost of label-compliant fine-tuning (same runs, no extra compute)")
    print("=" * 78)
    print(f"  assumptions: {WATTS:.0f} W sustained (Apple's max continuous rating, an "
          f"over-estimate),\n               {THB_PER_KWH} THB/kWh, {THB_PER_USD} THB/USD; "
          f"imputed cloud rate ${CLOUD_USD_PER_HOUR}/h")

    by = defaultdict(list)
    for c in cells:
        by[(c["arm"], c["n_train"])].append(c)

    out = {}
    for arm in sorted({a for a, _ in by}):
        print(f"\n  {arm}")
        print(f"    {'n_train':>8} {'train h':>8} {'eval h':>7} {'F1':>7} "
              f"{'USD elec':>9} {'USD cloud':>10} {'USD/F1pt (elec)':>16}")
        rows = []
        for n in sorted({n for a, n in by if a == arm}):
            cs = by[(arm, n)]
            th = sum(c["train_seconds"] for c in cs) / len(cs) / 3600
            eh = sum(c["eval_seconds"] for c in cs) / len(cs) / 3600
            f1 = sum(c["strict_macro_f1"] for c in cs) / len(cs)
            kwh = (th + eh) * WATTS / 1000
            usd_e = kwh * THB_PER_KWH / THB_PER_USD
            usd_c = (th + eh) * CLOUD_USD_PER_HOUR
            per_pt = usd_e / (f1 * 100) if f1 > 0 else float("nan")
            print(f"    {n:>8} {th:>8.2f} {eh:>7.2f} {f1:>7.4f} "
                  f"{usd_e:>9.3f} {usd_c:>10.3f} {per_pt:>16.5f}")
            rows.append({"n_train": n, "train_hours": th, "eval_hours": eh,
                         "strict_macro_f1": f1, "usd_electricity": usd_e,
                         "usd_cloud_imputed": usd_c, "usd_per_f1_point_elec": per_pt})
        out[arm] = rows

    tot_h = sum(c["train_seconds"] + c["eval_seconds"] for c in cells) / 3600
    print(f"\n  sweep total so far: {tot_h:.1f} GPU-hours  "
          f"= ${tot_h * WATTS / 1000 * THB_PER_KWH / THB_PER_USD:.2f} electricity, "
          f"${tot_h * CLOUD_USD_PER_HOUR:.2f} imputed cloud")
    print("  NOTE: eval dominates (10,000-row generation per cell). P7 should say")
    print("        whether it prices training only or training+evaluation; the")
    print("        original paper's figure was training-only.")
    out["_totals"] = {"gpu_hours": tot_h,
                      "usd_electricity": tot_h * WATTS / 1000 * THB_PER_KWH / THB_PER_USD,
                      "usd_cloud_imputed": tot_h * CLOUD_USD_PER_HOUR}
    return out


def entropy(counts):
    n = sum(counts.values())
    return -sum((v / n) * math.log2(v / n) for v in counts.values() if v)


def p24_p8_points(cells):
    """Label entropy and best-achieved F1 per arm — the datapoints P24's grading
    framework and P8's entropy regression both need."""
    print("\n" + "=" * 78)
    print("P24 / P8 — entropy and difficulty datapoints for the two SALAD-v2 arms")
    print("=" * 78)
    out = {}
    for arm in sorted({c["arm"] for c in cells}):
        cs = [c for c in cells if c["arm"] == arm]
        gold = Counter()
        for c in cs:
            gold.update(c.get("gold_vocab", {}))
        h = entropy(gold) if gold else float("nan")
        best = max(c["strict_macro_f1"] for c in cs)
        print(f"  {arm:<16} H(Y)={h:.4f} bits  K={len(gold)}  "
              f"best LLM strict macro-F1={best:.4f}")
        out[arm] = {"H_Y_bits": h, "K": len(gold), "best_llm_strict_f1": best,
                    "n_cells": len(cs)}
    print("\n  These are LLM-side numbers. P24/P8 also need each arm's classical")
    print("  baselines, which P13 already reports (UNSW DT/SVM 0.5874/0.6505;")
    print("  CICIDS 0.9310/0.9636) -- do not recompute, reuse and cite.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="results/salad2/p6_sweep")
    ap.add_argument("--out", default="results/salad2/p6_sweep_analysis.json")
    a = ap.parse_args()

    cells = load(a.sweep_dir)
    if not cells:
        print(f"no completed cells in {a.sweep_dir}")
        return
    arms = Counter(c["arm"] for c in cells)
    print(f"loaded {len(cells)} completed cells: {dict(arms)}")
    done = {(c["arm"], c["n_train"], c["seed"]) for c in cells}
    print(f"distinct (arm, n_train, seed) combinations: {len(done)}")

    report = {"n_cells": len(cells),
              "cost_assumptions": {"watts": WATTS, "thb_per_kwh": THB_PER_KWH,
                                   "thb_per_usd": THB_PER_USD,
                                   "cloud_usd_per_hour_imputed": CLOUD_USD_PER_HOUR},
              "p6": p6_scaling(cells),
              "p7": p7_cost(cells),
              "p24_p8": p24_p8_points(cells)}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2, default=str)
    print(f"\nwrote {a.out}")
    print("\nRun again when the sweep finishes; it is incremental and safe to "
          "re-run at any time.")


if __name__ == "__main__":
    main()
