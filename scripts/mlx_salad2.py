#!/usr/bin/env python3
"""
SALAD v2 on Apple Silicon (MLX) — fine-tune + strict-F1 eval, no cloud GPU.

WHY MLX / Mac Mini
------------------
P7 measured cloud fine-tuning at ~$0.60/model, so a 4-model ladder is ~$2.50 -- cheap, but
the Mac Mini M4 (32 GB, measured via `sysctl hw.memsize`) runs the whole thing for free and
24/7. Prior MLX work in this repo
(scripts/icl_baseline_mlx.py, base_control_mlx.py) was inference-only; this adds training.

WHAT IT MEASURES
----------------
SALAD v1's benchmark was a lookup: its prompt carried `Alert Type`, a deterministic 1:1
rename of the label, so *one column alone* scored macro-F1 0.96 -- the entire task. v2 is
built from real UNSW-NB15 / CICIDS-2017 flow features with a leakage gate (see
scripts/build_salad2.py). Classical baselines on v2:

    v1 (leaked)                DT 0.96   <- Alert Type alone also 0.96
    v1 leak-free (2 fields)    DT 0.37
    v2 / unsw                  DT 0.59, SVM 0.65    <- the honest target
    v2 / cicids                DT 0.96 (honest but near-solved; weak venue for LLM value)
    majority floor             0.02

So the question this script answers is finally a real one: **can a fine-tuned LLM beat
DT 0.59 on flow features that do not contain the answer?** On v1 that question was
meaningless.

Strict scoring: exact-match macro-F1 over gold-present classes. No alias normalization --
aliasing is what masked v1's 50-point gap in the first place.

USAGE (on the Mac Mini, 32 GB M4)
---------------------------------
    # one-time
    python3 -m venv .venv && .venv/bin/pip install mlx-lm datasets scikit-learn

    # convert v2 -> MLX chat jsonl
    python mlx_salad2.py prep --data data/salad2_unsw --out data/salad2_unsw/mlx

    # fine-tune (LoRA). 0.5B/1.5B are quick; 7B wants the 4-bit community weights.
    python mlx_salad2.py train --mlx-data data/salad2_unsw/mlx \
        --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --iters 600 --seed 42

    # evaluate strict macro-F1 (adapter optional -> omit for the BASE control)
    python mlx_salad2.py eval --data data/salad2_unsw --adapter adapters/qwen0.5b-s42 \
        --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --limit 2000

Suggested ladder for 32 GB (all -4bit community weights):
    Qwen2.5-0.5B-Instruct-4bit   Qwen2.5-1.5B-Instruct-4bit
    Qwen2.5-3B-Instruct-4bit     Qwen2.5-7B-Instruct-4bit
Run 3 seeds (42, 77, 123) per model: v1's headline instability was seed-driven, so a
single seed cannot support a claim here either.
"""
import argparse, json, os, random, re, subprocess, sys
from collections import Counter


# ---------- prep ----------
def cmd_prep(a):
    """SALAD v2 (ShareGPT-ish) -> MLX chat jsonl. mlx_lm expects {"messages":[...]}."""
    os.makedirs(a.out, exist_ok=True)
    role = {"system": "system", "human": "user", "gpt": "assistant"}
    counts = {}
    for split, fname in [("train", "salad2_train.json"), ("valid", "salad2_train.json"),
                         ("test", "salad2_test.json")]:
        src = os.path.join(a.data, fname)
        recs = json.load(open(src))
        # mlx_lm requires a valid split; carve it out of train rather than reusing test
        if split == "train":
            recs = recs[: int(len(recs) * 0.9)]
        elif split == "valid":
            recs = recs[int(len(recs) * 0.9):]
        out = []
        for r in recs:
            msgs = [{"role": role[c["from"]], "content": c["value"]} for c in r["conversations"]]
            out.append({"messages": msgs})
        p = os.path.join(a.out, f"{split}.jsonl")
        with open(p, "w") as fh:
            for o in out:
                fh.write(json.dumps(o) + "\n")
        counts[split] = len(out)
    print(f"wrote {a.out}/  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print("note: valid is carved from train (10%); test is untouched and never trained on.")


# ---------- train ----------
def cmd_train(a):
    ad = a.adapter or f"adapters/{os.path.basename(a.model).lower()}-s{a.seed}"
    os.makedirs(os.path.dirname(ad) or ".", exist_ok=True)
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", a.model, "--train",
           "--data", a.mlx_data, "--iters", str(a.iters), "--batch-size", str(a.batch),
           "--num-layers", str(a.layers), "--adapter-path", ad, "--seed", str(a.seed)]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"\nadapter -> {ad}")


# ---------- eval ----------
def parse_pred(txt):
    m = re.search(r"Attack Category:\s*([A-Za-z][A-Za-z0-9_\- ]*)", txt)
    return m.group(1).strip() if m else txt.strip().split("\n")[0].strip()


def cmd_eval(a):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    from sklearn.metrics import f1_score

    recs = json.load(open(os.path.join(a.data, "salad2_test.json")))
    if a.limit:
        random.Random(a.seed).shuffle(recs)
        recs = recs[: a.limit]
    model, tok = load(a.model, adapter_path=a.adapter) if a.adapter else load(a.model)
    sampler = make_sampler(temp=0.0)

    gold, pred, raw, text = [], [], [], []
    for i, r in enumerate(recs):
        msgs = [{"role": {"system": "system", "human": "user", "gpt": "assistant"}[c["from"]],
                 "content": c["value"]} for c in r["conversations"] if c["from"] != "gpt"]
        g = parse_pred([c["value"] for c in r["conversations"] if c["from"] == "gpt"][0])
        human = [c["value"] for c in r["conversations"] if c["from"] == "human"][0]
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        o = generate(model, tok, prompt=p, max_tokens=a.max_tokens, sampler=sampler, verbose=False)
        gold.append(g); pred.append(parse_pred(o)); raw.append(o.strip()[:80]); text.append(human)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)

    labels = sorted(set(gold))                      # gold-present classes only
    strict = f1_score(gold, pred, labels=labels, average="macro", zero_division=0)
    offvocab = sum(1 for p in pred if p not in labels) / len(pred)
    acc = sum(int(g == p) for g, p in zip(gold, pred)) / len(pred)
    res = {"model": a.model, "adapter": a.adapter, "n": len(gold),
           "strict_macro_f1": round(strict, 4), "exact_accuracy": round(acc, 4),
           "off_vocab_rate": round(offvocab, 4),
           "pred_vocab": dict(Counter(pred).most_common(12)),
           "gold_vocab": dict(Counter(gold).most_common(12))}
    print(json.dumps(res, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({**res, "samples": raw[:20]}, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    if a.preds_out:
        os.makedirs(os.path.dirname(a.preds_out) or ".", exist_ok=True)
        with open(a.preds_out, "w") as f:
            for t, g, p in zip(text, gold, pred):
                f.write(json.dumps({"text": t, "label": g, "pred": p}) + "\n")
        print(f"wrote {a.preds_out} ({len(gold)} rows, full per-example gold/pred)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep"); p.add_argument("--data", required=True)
    p.add_argument("--out", required=True); p.set_defaults(f=cmd_prep)

    t = sub.add_parser("train"); t.add_argument("--mlx-data", required=True)
    t.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    t.add_argument("--iters", type=int, default=600); t.add_argument("--batch", type=int, default=4)
    t.add_argument("--layers", type=int, default=8); t.add_argument("--seed", type=int, default=42)
    t.add_argument("--adapter"); t.set_defaults(f=cmd_train)

    e = sub.add_parser("eval"); e.add_argument("--data", required=True)
    e.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    e.add_argument("--adapter"); e.add_argument("--limit", type=int, default=2000)
    e.add_argument("--max-tokens", type=int, default=16); e.add_argument("--seed", type=int, default=42)
    e.add_argument("--out"); e.add_argument("--preds-out", default=None,
                   help="optional: write full per-example (text,label,pred) JSONL for auditing")
    e.set_defaults(f=cmd_eval)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
