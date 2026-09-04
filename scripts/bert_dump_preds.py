#!/usr/bin/env python3
"""Dump per-row test predictions from an already-trained SALAD-v2 BERT checkpoint.

WHY: scripts/bert_baseline.py saves only summary metrics. papers/p13-salad-v2 needs BERT's
own bootstrap CI and a PAIRED comparison against the classical baselines (whose per-row
predictions scripts/salad2_baseline_stats.py already dumps). Retraining to get predictions
would risk reporting a different model than the one whose score is in the paper, so this
loads the exact selected checkpoint and only runs inference.

Label encoding is reconstructed the same way bert_baseline.py builds it (LabelEncoder fit on
the union of train and test label strings, which sklearn sorts alphabetically), so class
indices match what the checkpoint was trained against. The script re-scores macro-F1 and
refuses to write output if it does not match the expected value, which is the guard against
a silently mismatched label mapping.

Usage:
    python3 scripts/bert_dump_preds.py --dataset unsw   --ckpt outputs/bert-unsw-refit/checkpoint-1410   --expect 0.5946
    python3 scripts/bert_dump_preds.py --dataset cicids --ckpt outputs/bert-cicids-refit/checkpoint-1128 --expect 0.9532
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification, BertTokenizer


def load_sharegpt(path):
    texts, labels = [], []
    for item in json.load(open(path)):
        inp = out = ""
        for c in item.get("conversations", []):
            if c["from"] == "human":
                inp = c["value"]
            elif c["from"] == "gpt":
                out = c["value"]
        label = out.strip()
        if "Attack Category:" in out:
            for line in out.split("\n"):
                if "Attack Category:" in line:
                    label = line.split(":", 1)[1].strip()
                    break
        texts.append(inp[:512])
        labels.append(label[:80])
    return texts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["unsw", "cicids"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--expect", type=float, required=True,
                    help="macro-F1 the checkpoint is reported as scoring; guards label mapping")
    ap.add_argument("--tol", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    d = f"data/salad2_{args.dataset}"
    tr_texts, tr_labels = load_sharegpt(f"{d}/salad2_train.json")
    te_texts, te_labels = load_sharegpt(f"{d}/salad2_test.json")

    # identical to bert_baseline.py: fit on the union so a test-only class stays encodable
    le = LabelEncoder().fit(list(tr_labels) + list(te_labels))
    y_true = le.transform(te_labels)

    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(args.ckpt)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(te_texts), args.batch_size):
            batch = te_texts[i:i + args.batch_size]
            enc = tok(batch, truncation=True, padding=True, max_length=512,
                      return_tensors="pt")
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).tolist())
            if (i // args.batch_size) % 25 == 0:
                print(f"  {i}/{len(te_texts)}", flush=True)

    preds = np.array(preds)
    got = f1_score(y_true, preds, average="macro", zero_division=0)
    print(f"\nmacro-F1 from checkpoint: {got:.4f}   expected: {args.expect}")
    if abs(got - args.expect) > args.tol:
        sys.exit(f"ABORT: macro-F1 {got:.4f} != expected {args.expect} "
                 f"(tol {args.tol}). Label mapping or checkpoint is wrong; nothing written.")

    pred_labels = le.inverse_transform(preds).tolist()
    out_path = f"results/salad2/{args.dataset}_bert_preds.json"
    os.makedirs("results/salad2", exist_ok=True)
    json.dump({"gold": te_labels, "bert": pred_labels,
               "checkpoint": args.ckpt, "macro_f1": round(float(got), 4)},
              open(out_path, "w"))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
