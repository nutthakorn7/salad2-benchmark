#!/usr/bin/env python3
"""Paired bootstrap comparison of every non-LLM baseline on SALAD-v2.

Consumes the per-row prediction dumps written by salad2_baseline_stats.py (Decision Tree,
LinearSVC, majority floor) and bert_dump_preds.py (BERT-base), which are scored on the same
test rows in the same order. Every bootstrap iteration resamples ONE set of row indices and
scores all four models on it, so each pairwise difference is genuinely paired rather than a
comparison of two independently-resampled estimates.

Reports, per pair: the observed macro-F1 difference, its 95% percentile CI, and the fraction
of resamples in which the sign flips (a bootstrap p-value analogue). A pair whose CI spans 0
is NOT a demonstrated separation, however large the point difference looks.

The LLM ladder is deliberately absent: per-row predictions for those 24 cells on the full
10,000-row test set were not retained, so they cannot be paired against anything here. That
gap is disclosed in the paper's Limitations rather than papered over with an unpaired test.

Usage:
    python3 scripts/salad2_paired_tests.py --dataset unsw
    python3 scripts/salad2_paired_tests.py --dataset cicids
"""
import argparse
import itertools
import json

import numpy as np
from sklearn.metrics import f1_score

N_BOOT = 2000
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["unsw", "cicids"], default="unsw")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    base = json.load(open(f"results/salad2/{args.dataset}_baseline_preds.json"))
    bert = json.load(open(f"results/salad2/{args.dataset}_bert_preds.json"))
    if base["gold"] != bert["gold"]:
        raise SystemExit("ABORT: gold label sequences differ; rows are not aligned.")

    gold = np.array(base["gold"])
    labels = sorted(set(base["gold"]))
    models = {
        "decision_tree": np.array(base["decision_tree"]),
        "linear_svc": np.array(base["linear_svc"]),
        "bert": np.array(bert["bert"]),
        "majority_floor": np.array(base["majority_class_floor"]),
    }

    def mf1(g, p):
        return f1_score(g, p, labels=labels, average="macro", zero_division=0)

    point = {k: round(float(mf1(gold, v)), 4) for k, v in models.items()}
    print(f"SALAD-v2 {args.dataset}: n={len(gold)} classes={len(labels)}")
    print("\npoint estimates:")
    for k, v in sorted(point.items(), key=lambda x: -x[1]):
        print(f"  {k:16s} {v}")

    rng = np.random.default_rng(SEED)
    n = len(gold)
    draws = {k: [] for k in models}
    pairs = list(itertools.combinations(models, 2))
    pair_diffs = {p: [] for p in pairs}

    print(f"\nbootstrapping ({args.n_boot} resamples, shared row indices)...")
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        g = gold[idx]
        s = {k: mf1(g, v[idx]) for k, v in models.items()}
        for k in models:
            draws[k].append(s[k])
        for a, b in pairs:
            pair_diffs[(a, b)].append(s[a] - s[b])

    print("\nmarginal 95% CIs:")
    ci = {}
    for k, vals in draws.items():
        vals = np.array(vals)
        ci[k] = {"ci95_low": round(float(np.percentile(vals, 2.5)), 4),
                 "ci95_high": round(float(np.percentile(vals, 97.5)), 4)}
        print(f"  {k:16s} {point[k]}  [{ci[k]['ci95_low']}, {ci[k]['ci95_high']}]")

    print("\npaired differences (A - B):")
    out_pairs = {}
    for (a, b), d in pair_diffs.items():
        d = np.array(d)
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        frac_flip = float((d <= 0).mean()) if d.mean() > 0 else float((d >= 0).mean())
        separated = (lo > 0) or (hi < 0)
        out_pairs[f"{a}_minus_{b}"] = {
            "observed": round(float(point[a] - point[b]), 4),
            "boot_mean": round(float(d.mean()), 4),
            "ci95_low": round(float(lo), 4),
            "ci95_high": round(float(hi), 4),
            "frac_sign_flip": round(frac_flip, 4),
            "separated_at_95": bool(separated),
        }
        flag = "SEPARATED" if separated else "NOT separated (CI spans 0)"
        print(f"  {a:16s} - {b:16s} {point[a]-point[b]:+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  flip={frac_flip:.1%}  {flag}")

    out = {
        "dataset": args.dataset,
        "n_test": int(n),
        "n_bootstrap": args.n_boot,
        "point_estimates": point,
        "marginal_ci": ci,
        "paired": out_pairs,
        "note": "LLM ladder cells excluded: per-row predictions on the full test set were "
                "not retained, so they cannot be paired against these baselines.",
    }
    path = f"results/salad2/{args.dataset}_paired_tests.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
