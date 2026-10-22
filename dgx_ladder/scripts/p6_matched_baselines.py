#!/usr/bin/env python3
"""
Training-matched classical baselines for the P6 training-size sweep (SALAD-v2).

Why: P6/P7 compare every LLM sweep cell against the CANONICAL baselines (salad2_baselines.py: TF-IDF + DT depth 20 /
LinearSVC trained on the canonical 5,000-row split with its natural class mix, 16x UNSW / 4x CICIDS imbalance). But every
P6 sweep cell trains on a fixed-proportion subset (about 89x UNSW / 20x CICIDS imbalance, data/p6_fixed/<arm>_n<N>/) and,
as run by scripts/mlx_salad2.py, on the FIRST 90% of that file (the last 10% is the MLX validation split). This script
trains the SAME classical models (identical TF-IDF/DT/LinearSVC settings as salad2_baselines.py) on exactly those rows so
the size sweep can be read against baselines that saw the same data. Test = the arm's shipped 10,000-row salad2_test.json.

Run from the repo root that contains data/salad2_<arm>/ and data/p6_fixed/ (a checkout with the built corpus):
    python3 scripts/p6_matched_baselines.py   -> results/salad2/p6_matched_baselines.json
"""
import json, re, warnings
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter("ignore")
SIZES = (1000, 2500, 5000, 10000, 20000)


def load(rows):
    X, y = [], []
    for row in rows:
        conv = row["conversations"]
        human = [c["value"] for c in conv if c["from"] == "human"][0]
        gpt = [c["value"] for c in conv if c["from"] == "gpt"][0]
        m = re.search(r"Attack Category:\s*(.+)", gpt)
        if m:
            X.append(human); y.append(m.group(1).strip())
    return X, y


out = {"method": "TF-IDF(max_features=5000, stop_words=english, token_pattern=[^\\s:]+); DecisionTree(max_depth=20, "
                 "random_state=42); LinearSVC(random_state=42); macro-F1 over gold-present classes on the arm's full "
                 "10,000-row test set", "cells": []}
for arm in ("unsw", "cicids"):
    Xte, yte = load(json.load(open(f"data/salad2_{arm}/salad2_test.json")))
    for n in SIZES:
        rows = json.load(open(f"data/p6_fixed/{arm}_n{n}/salad2_train.json"))
        for tag, rs in (("first90pct_as_trained_by_mlx", rows[: int(len(rows) * 0.9)]), ("all_rows", rows)):
            Xtr, ytr = load(rs)
            cnt = Counter(ytr)
            vec = TfidfVectorizer(max_features=5000, stop_words="english", token_pattern=r"[^\s:]+")
            A = vec.fit_transform(Xtr); B = vec.transform(Xte)
            dt = DecisionTreeClassifier(random_state=42, max_depth=20).fit(A, ytr)
            sv = LinearSVC(random_state=42).fit(A, ytr)
            f = lambda m: round(float(f1_score(yte, m.predict(B), average="macro", zero_division=0)), 4)
            out["cells"].append({"arm": arm, "n": n, "train_rows_used": tag, "n_train": len(Xtr),
                                 "class_counts": dict(cnt), "max_to_min_ratio": round(max(cnt.values()) / min(cnt.values()), 1),
                                 "decision_tree": f(dt), "linear_svc": f(sv)})
            print(arm, n, tag, len(Xtr), "DT", out["cells"][-1]["decision_tree"], "SVC", out["cells"][-1]["linear_svc"], flush=True)
json.dump(out, open("results/salad2/p6_matched_baselines.json", "w"), indent=1)
