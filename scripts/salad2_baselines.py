#!/usr/bin/env python3
"""Classical ML baselines for SALAD-v2 (UNSW-NB15 and CICIDS-2017 arms), computed on the
SHIPPED salad2_train.json / salad2_test.json files -- these are the numbers papers/p13-salad-v2's
main.tex Table IV / Sec. V-A compares the LLM fine-tuning ladder against (UNSW-NB15 arm only,
as of the paper's current draft; the CICIDS-2017 numbers are computed here in advance of that
arm's own LLM ladder, which is still running as of 2026-07-19).

METHOD: TF-IDF over the rendered flow-text prompt (the exact text the LLM sees), then
Decision Tree (depth 20, matching check_leakage()'s own tree depth) / LinearSVC / a
majority-class floor. This mirrors data/salad2_unsw/README.md's own baseline table
header ("Baselines (TF-IDF, macro-F1 over 10 classes)"), which is TF-IDF-over-text, NOT
the structured-feature encoding check_leakage() uses internally for its gate reference
score (that internal 0.5817 number is a DIFFERENT measurement over a different
train/test population -- see check_leakage()'s own 70/30 dedup split in build_salad2.py
-- and is not meant to reproduce this file's numbers; do not conflate the two).

Independently re-run against the shipped splits, this reproduces the paper's reported
UNSW-NB15 values (DT 0.5874, LinearSVC 0.6505, majority floor 0.0234).

Usage:
    python3 scripts/salad2_baselines.py --dataset unsw     (default)
    python3 scripts/salad2_baselines.py --dataset cicids
"""
import argparse
import json
import re
import warnings
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["unsw", "cicids"], default="unsw")
    args = ap.parse_args()

    data_dir = f"data/salad2_{args.dataset}"
    out_path = f"results/salad2/{args.dataset}_baselines.json"

    Xtr, ytr = load(f"{data_dir}/salad2_train.json")
    Xte, yte = load(f"{data_dir}/salad2_test.json")
    print(f"SALAD-v2 {args.dataset}: train={len(Xtr)} test={len(Xte)} classes={len(set(ytr))}")

    vec = TfidfVectorizer(max_features=5000, stop_words="english", token_pattern=r"[^\s:]+")
    Atr = vec.fit_transform(Xtr)
    Ate = vec.transform(Xte)

    results = {}

    dt = DecisionTreeClassifier(random_state=42, max_depth=20)
    dt.fit(Atr, ytr)
    f1_dt = f1_score(yte, dt.predict(Ate), average="macro", zero_division=0)
    results["decision_tree"] = round(float(f1_dt), 4)
    print(f"  Decision Tree (depth 20)  macro-F1 = {f1_dt:.4f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # LinearSVC convergence warning, default max_iter
        svm = LinearSVC(random_state=42)
        svm.fit(Atr, ytr)
    f1_svm = f1_score(yte, svm.predict(Ate), average="macro", zero_division=0)
    results["linear_svc"] = round(float(f1_svm), 4)
    print(f"  LinearSVC                 macro-F1 = {f1_svm:.4f}")

    maj = Counter(ytr).most_common(1)[0][0]
    f1_maj = f1_score(yte, [maj] * len(yte), average="macro", zero_division=0)
    results["majority_class_floor"] = round(float(f1_maj), 4)
    print(f"  Majority-class floor       macro-F1 = {f1_maj:.4f} (predicts '{maj}')")

    results["method"] = "TF-IDF(max_features=5000, stop_words=english, token_pattern=[^\\s:]+) over rendered flow-text prompt"
    results["train_file"] = f"{data_dir}/salad2_train.json"
    results["test_file"] = f"{data_dir}/salad2_test.json"
    json.dump(results, open(out_path, "w"), indent=1)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
