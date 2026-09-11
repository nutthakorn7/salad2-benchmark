#!/usr/bin/env python3
"""
P6 training-size sweep on leak-free SALAD-v2. 1.5B x 5 training sizes x 3 seeds x 2 arms.

Design notes that must survive into the paper:
  * Training sets come from data/p6_fixed/, built with --fixed-proportions so class
    imbalance does NOT co-vary with n (it is ~90x on UNSW / ~21x on CICIDS at EVERY
    size). The default round-robin would have taken UNSW from 2.6x to 91.3x across the
    sweep, confounding size with imbalance under a macro-F1 metric.
  * Every size scores on the SHIPPED test split, byte-identical to P13's 24-cell ladder,
    so these numbers are comparable to it.
  * --iters is FIXED at 600. Training compute is therefore constant across sizes; what
    varies is how many distinct examples the model sees within that budget. This is a
    fixed-compute/varying-diversity design, NOT a "spend more to get more" design, and
    P7's cost curve must be written accordingly.

Resumable: skips any (arm,n,seed) whose result JSON exists. Checkpoints per run.
"""
import os, subprocess, sys, time, json

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
SIZES = [1000, 2500, 5000, 10000, 20000]
SEEDS = [42, 77, 123]
ARMS  = ["unsw", "cicids"]
ITERS = 600
OUT   = "results/salad2/p6_sweep"
os.makedirs(OUT, exist_ok=True); os.makedirs("logs", exist_ok=True)

def log(m): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

runs = [(a, n, s) for a in ARMS for n in SIZES for s in SEEDS]
log(f"P6 sweep: {len(runs)} runs (1.5B, {len(SIZES)} sizes, {len(SEEDS)} seeds, {len(ARMS)} arms)")

for i, (arm, n, seed) in enumerate(runs, 1):
    tag = f"{arm}_n{n}_s{seed}"
    res = f"{OUT}/{tag}.json"
    if os.path.exists(res):
        log(f"[{i}/{len(runs)}] {tag}: done, skip"); continue
    data = f"data/p6_fixed/{arm}_n{n}"
    mlx  = f"data/p6_fixed/{arm}_n{n}/mlx"
    adap = f"adapters/p6_{tag}"
    t0 = time.time()
    try:
        if not os.path.exists(f"{mlx}/train.jsonl"):
            subprocess.run([sys.executable, "scripts/mlx_salad2.py", "prep",
                            "--data", data, "--out", mlx], check=True)
        log(f"[{i}/{len(runs)}] {tag}: training ({ITERS} iters)")
        subprocess.run([sys.executable, "scripts/mlx_salad2.py", "train",
                        "--mlx-data", mlx, "--model", MODEL, "--iters", str(ITERS),
                        "--seed", str(seed), "--adapter", adap], check=True)
        ttrain = time.time() - t0
        log(f"[{i}/{len(runs)}] {tag}: eval on full 10k test set")
        t1 = time.time()
        subprocess.run([sys.executable, "scripts/mlx_salad2.py", "eval",
                        "--data", data, "--model", MODEL, "--adapter", adap,
                        "--limit", "10000", "--out", res,
                        "--preds-out", f"{OUT}/{tag}_preds.jsonl"], check=True)
        teval = time.time() - t1
        # stamp timings for P7's cost analysis
        try:
            d = json.load(open(res)); d["train_seconds"] = round(ttrain, 1)
            d["eval_seconds"] = round(teval, 1); d["iters"] = ITERS
            d["n_train"] = n; d["arm"] = arm; d["seed"] = seed
            json.dump(d, open(res, "w"), indent=1)
        except Exception as e:
            log(f"    (could not stamp timings: {e})")
        log(f"[{i}/{len(runs)}] {tag}: DONE train={ttrain/60:.1f}m eval={teval/60:.1f}m")
    except subprocess.CalledProcessError as e:
        log(f"[{i}/{len(runs)}] {tag}: FAILED ({e}) — continuing, will retry on relaunch")
        continue
log("P6 SWEEP COMPLETE")
