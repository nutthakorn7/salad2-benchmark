#!/usr/bin/env python3
"""Uncertainty and per-class breakdown for SALAD-v2's classical baselines.

WHY THIS EXISTS: scripts/salad2_baselines.py reports a single point estimate per model
(DT / LinearSVC / majority floor). An external review of papers/p13-salad-v2 flagged that
the paper's headline model ordering rests on point estimates with no uncertainty, while the
LLM ladder it is compared against carries a 3-seed spread -- so adjacent pairs separated by
0.0072 and 0.0104 macro-F1 were being described with words ("beats") the evidence could not
support. This script supplies what was missing:

  * percentile bootstrap CIs over test rows for every classical baseline
  * a PAIRED bootstrap of the DT-vs-LinearSVC difference (same resampled rows for both)
  * per-class F1, so a reader can see which classes carry the macro average
  * per-row predictions dumped to disk, so any later paired test (e.g. against BERT or an
    LLM cell) can be run without re-fitting anything

Pipeline is copied from salad2_baselines.py deliberately -- same TfidfVectorizer params,
same DecisionTreeClassifier(max_depth=20, random_state=42), same LinearSVC(random_state=42).
If those two files ever disagree, the numbers in the paper are wrong somewhere.

Usage:
    python3 scripts/salad2_baseline_stats.py --dataset unsw
    python3 scripts/salad2_baseline_stats.py --dataset cicids
"""
import argparse
import json
import re
import warnings
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

N_BOOT = 2000
SEED = 42


def load(path):
    X, y = [], []
    for row in json.load(open(path)):
        conv = row["conversations"]
        human = [c["value"] for c in conv if c["from"] == "human"][0]
        gpt = [c["value"] for c in conv if c["from"] == "gpt"][0]
        m = re.search(r"Attack Category:\s*(.+)", gpt)
        if m:
            X.append(human)
            y.append(m.group(1).strip())
    return X, y


def macro_f1(gold, pred, labels):
    return f1_score(gold, pred, labels=labels, average="macro", zero_division=0)


def bootstrap_ci(gold, preds_by_model, labels, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap over TEST ROWS. All models are scored on the SAME resampled
    rows in each iteration, which is what makes the pairwise differences below paired."""
    rng = np.random.default_rng(seed)
    n = len(gold)
    gold = np.asarray(gold)
    preds_by_model = {k: np.asarray(v) for k, v in preds_by_model.items()}

    draws = {k: [] for k in preds_by_model}
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        g = gold[idx]
        scored = {}
        for name, p in preds_by_model.items():
            scored[name] = macro_f1(g, p[idx], labels)
            draws[name].append(scored[name])
        if "decision_tree" in scored and "linear_svc" in scored:
            diffs.append(scored["linear_svc"] - scored["decision_tree"])

    out = {}
    for name, vals in draws.items():
        vals = np.array(vals)
        out[name] = {
            "boot_mean": round(float(vals.mean()), 4),
            "ci95_low": round(float(np.percentile(vals, 2.5)), 4),
            "ci95_high": round(float(np.percentile(vals, 97.5)), 4),
        }
    if diffs:
        d = np.array(diffs)
        out["_paired_svc_minus_dt"] = {
            "mean_diff": round(float(d.mean()), 4),
            "ci95_low": round(float(np.percentile(d, 2.5)), 4),
            "ci95_high": round(float(np.percentile(d, 97.5)), 4),
            # fraction of resamples in which the sign flips -- a bootstrap p-value analogue
            "frac_favoring_dt": round(float((d <= 0).mean()), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["unsw", "cicids"], default="unsw")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    data_dir = f"data/salad2_{args.dataset}"
    Xtr, ytr = load(f"{data_dir}/salad2_train.json")
    Xte, yte = load(f"{data_dir}/salad2_test.json")
    labels = sorted(set(yte))
    print(f"SALAD-v2 {args.dataset}: train={len(Xtr)} test={len(Xte)} "
          f"classes_in_test={len(labels)}")

    vec = TfidfVectorizer(max_features=5000, stop_words="english",
                          token_pattern=r"[^\s:]+")
    Atr = vec.fit_transform(Xtr)
    Ate = vec.transform(Xte)

    preds = {}
    dt = DecisionTreeClassifier(random_state=42, max_depth=20).fit(Atr, ytr)
    preds["decision_tree"] = list(dt.predict(Ate))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svm = LinearSVC(random_state=42).fit(Atr, ytr)
    preds["linear_svc"] = list(svm.predict(Ate))
    maj = Counter(ytr).most_common(1)[0][0]
    preds["majority_class_floor"] = [maj] * len(yte)

    point = {k: round(float(macro_f1(yte, v, labels)), 4) for k, v in preds.items()}
    print("\npoint estimates (must match salad2_baselines.json):")
    for k, v in point.items():
        print(f"  {k:24s} {v}")

    print(f"\nbootstrapping ({args.n_boot} resamples over {len(yte)} test rows)...")
    ci = bootstrap_ci(yte, preds, labels, n_boot=args.n_boot)
    for k in point:
        c = ci[k]
        print(f"  {k:24s} {point[k]}  95% CI [{c['ci95_low']}, {c['ci95_high']}]")
    p = ci["_paired_svc_minus_dt"]
    print(f"\n  paired LinearSVC - DecisionTree: {p['mean_diff']:+.4f} "
          f"95% CI [{p['ci95_low']}, {p['ci95_high']}]  "
          f"resamples favoring DT: {p['frac_favoring_dt']:.1%}")

    per_class = {}
    for name, pv in preds.items():
        f1s = f1_score(yte, pv, labels=labels, average=None, zero_division=0)
        per_class[name] = {lab: round(float(f), 4) for lab, f in zip(labels, f1s)}
    support = Counter(yte)
    print("\nper-class F1:")
    header = f"  {'class':<18}{'n':>6}" + "".join(f"{k[:12]:>14}" for k in preds)
    print(header)
    for lab in labels:
        row = f"  {lab:<18}{support[lab]:>6}"
        for name in preds:
            row += f"{per_class[name][lab]:>14.4f}"
        print(row)

    out = {
        "dataset": args.dataset,
        "n_test": len(yte),
        "n_bootstrap": args.n_boot,
        "labels": labels,
        "support": {k: int(v) for k, v in support.items()},
        "point_estimates": point,
        "bootstrap": ci,
        "per_class_f1": per_class,
        "method": "TF-IDF(max_features=5000, stop_words=english, token_pattern=[^\\s:]+) "
                  "over rendered flow-text prompt; identical pipeline to salad2_baselines.py",
    }
    out_path = f"results/salad2/{args.dataset}_baseline_stats.json"
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"\nSaved -> {out_path}")

    pred_path = f"results/salad2/{args.dataset}_baseline_preds.json"
    json.dump({"gold": yte, **preds}, open(pred_path, "w"))
    print(f"Saved per-row predictions -> {pred_path}")


if __name__ == "__main__":
    main()
