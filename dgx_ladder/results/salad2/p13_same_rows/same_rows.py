# Same-rows classical baselines for the P13 ladder (verifier run 2026-09-26).
# Trains the paper's exact TF-IDF + DecisionTree(depth 20) / LinearSVC (scripts/salad2_baselines.py settings)
# on exactly the rows each LLM ladder cell saw, then scores on the full 10,000-row shipped test split.
# Row indices come from rows_seen.py (replays mlx_lm 0.29.1 iterate_batches with np.random.seed(--seed);
# run in the MLX virtualenv: python - unsw < rows_seen.py).
# tok_check.py: fingerprint check -- summed chat-template token counts of the first 10/50/100 replayed batches
# equal the logged "Trained Tokens" (cicids s42 39752/199230/400350; cicids s123 40267/198913/396896;
# unsw s77 21069/105333/209899; unsw s42 21184 at step 10 / 211187 at step 100, from ladder.log).
# Steps per cell: 600 for UNSW 0.5B/1.5B/3B s42 (uninterrupted), 300 for UNSW 7B s42, 100 for the other 20 cells.
import json, re, warnings, sys, statistics as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
warnings.simplefilter("ignore")
import os
H = os.path.dirname(os.path.abspath(__file__)) + "/"
D = os.path.abspath(H + "../../../") + "/"   # repo root
def load(rows):
    X, y = [], []
    for row in rows:
        conv = row["conversations"]
        h = [c["value"] for c in conv if c["from"] == "human"][0]
        g = [c["value"] for c in conv if c["from"] == "gpt"][0]
        m = re.search(r"Attack Category:\s*(.+)", g)
        if m: X.append(h); y.append(m.group(1).strip())
    return X, y
def fit_eval(Xtr, ytr, Xte, yte):
    vec = TfidfVectorizer(max_features=5000, stop_words="english", token_pattern=r"[^\s:]+")
    A = vec.fit_transform(Xtr); B = vec.transform(Xte)
    dt = DecisionTreeClassifier(random_state=42, max_depth=20).fit(A, ytr)
    sv = LinearSVC(random_state=42).fit(A, ytr)
    f = lambda m: float(f1_score(yte, m.predict(B), average="macro", zero_division=0))
    return f(dt), f(sv)
llm = {}
for arm in ("unsw", "cicids"):
    for size in ("0.5b", "1.5b", "3b", "7b"):
        for s in (42, 77, 123):
            llm[(arm, size, s)] = json.load(open(f"{D}results/salad2/full_test/{arm}_{size}_s{s}_full.json"))["strict_macro_f1"]
out = {}
for arm in ("unsw", "cicids"):
    tr = json.load(open(f"{D}data/salad2_{arm}/salad2_train.json"))
    Xte, yte = load(json.load(open(f"{D}data/salad2_{arm}/salad2_test.json")))
    pool = tr[:4500]
    rows = json.load(open(H + f"rows_{arm}.json"))
    Xc, yc = load(tr)
    print(arm, "canonical (all 5000):", [round(v, 4) for v in fit_eval(Xc, yc, Xte, yte)], flush=True)
    Xp, yp = load(pool)
    print(arm, "first-4500 pool:", [round(v, 4) for v in fit_eval(Xp, yp, Xte, yte)], flush=True)
    for seed in ("42", "77", "123"):
        seq = rows[seed]
        for steps in (100, 300, 600):
            idx = seq[: steps * 4]
            X = [Xp[i] for i in idx]; y = [yp[i] for i in idx]
            dt, sv = fit_eval(X, y, Xte, yte)
            from collections import Counter
            c = Counter(y)
            out[f"{arm}_s{seed}_steps{steps}"] = dict(n=len(set(idx)), n_classes=len(c), min_class=min(c.values()), dt=round(dt, 4), svc=round(sv, 4))
            print(arm, "seed", seed, "steps", steps, "distinct", len(set(idx)), "classes", len(c), "DT %.4f SVC %.4f" % (dt, sv), flush=True)
json.dump(out, open(H + "same_rows_out.json", "w"), indent=1)
