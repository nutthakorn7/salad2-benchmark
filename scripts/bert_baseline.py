#!/usr/bin/env python3
"""
BERT-base classification baseline for Q1 Rule of Law compliance.
Uses HuggingFace Trainer (not LlamaFactory) since BERT is not generative.

2026-07-21: adapted for SALAD-v2 (Mac Mini, CPU-only to avoid contending with the
concurrent MLX/Metal training). Two changes from an earlier cluster-era version:
(1) BERT_PATH was a cluster-local path; it now pulls "bert-base-uncased" from the HF
Hub directly. (2) train_bert() used to re-split its single input file 80/20 internally,
which is NOT the same split DT/SVM/the LLM ladder are evaluated against -- now accepts
explicit train/test files and uses SALAD-v2's real, pre-split, zero-overlap files as-is,
so the BERT number is a fair apples-to-apples comparison against the other baselines.
"""
import json, re, sys, os
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

def load_sharegpt(path, task="attack_category"):
    """Load ShareGPT format and extract labels."""
    with open(path) as f:
        data = json.load(f)
    
    texts, labels = [], []
    for item in data:
        convs = item.get("conversations", [])
        inp = ""
        out = ""
        for c in convs:
            if c["from"] == "human": inp = c["value"]
            elif c["from"] == "gpt": out = c["value"]
        
        label = out.strip()
        if "Attack Category:" in out:
            for line in out.split("\n"):
                if "Attack Category:" in line:
                    label = line.split(":", 1)[1].strip()
                    break
        elif label.startswith("Category: "):
            label = label[10:]
        
        texts.append(inp[:512])
        labels.append(label[:80])
    
    return texts, labels


def train_bert(train_texts, train_labels, test_texts, test_labels, output_dir, epochs=5, batch_size=16):
    """Train BERT-base for classification on SALAD-v2's real, pre-split train/test files."""
    from transformers import (
        BertTokenizer, BertForSequenceClassification,
        TrainingArguments, Trainer
    )
    from torch.utils.data import Dataset
    import torch

    # Label encoder fit on the UNION of train+test labels -- a label that appears only in
    # the test split (real possibility with a small train set and rare classes) must still
    # be encodable, or evaluation crashes on an unseen label.
    le = LabelEncoder()
    le.fit(list(train_labels) + list(test_labels))
    y_train = le.transform(train_labels)
    y_test = le.transform(test_labels)
    num_labels = len(le.classes_)
    X_train, X_test = train_texts, test_texts

    BERT_PATH = "bert-base-uncased"  # HF Hub ID

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    
    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
            self.labels = labels
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)
    
    # 2026-07-30 FIX: model selection used to run against test_ds, so the reported
    # number was the best of N epochs chosen on the very set it was reported on.
    # Selection now happens on a validation split carved from TRAIN; the test set
    # is touched only for the final report.
    from sklearn.model_selection import train_test_split
    strat = y_train if min(Counter(y_train).values()) >= 2 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=strat)

    train_ds = TextDataset(X_tr, list(y_tr))
    val_ds = TextDataset(X_val, list(y_val))
    test_ds = TextDataset(X_test, y_test.tolist())
    
    model = BertForSequenceClassification.from_pretrained(
        BERT_PATH, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=False,  # CPU-only here; avoid GPU-precision assumptions
        use_cpu=True,  # force CPU: avoids MPS/Metal contention with the concurrent MLX training
        logging_steps=50,
        seed=42,
    )
    
    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=-1)
        f1 = f1_score(pred.label_ids, preds, average="macro", zero_division=0)
        acc = (preds == pred.label_ids).mean()
        return {"f1": f1, "accuracy": acc}
    
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,   # selection on val, never test
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    # per-epoch trace on the VALIDATION split, persisted so nobody has to dig
    # through a .out log to find out whether selection was a no-op (see the
    # bert-baseline-selects-on-test note in the project memory).
    val_trace = [round(h["eval_f1"], 4) for h in trainer.state.log_history if "eval_f1" in h]
    best_ckpt = trainer.state.best_model_checkpoint
    results = trainer.evaluate(test_ds)   # the reported number: TEST, after selection on val
    
    # Per-class report
    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"  BERT-base Results: {output_dir}")
    print(f"{'='*60}")
    print(f"  Macro-F1: {results['eval_f1']:.4f}")
    print(f"  Accuracy: {results['eval_accuracy']:.4f}")
    print(f"\n{report}")
    
    # Save results
    with open(os.path.join(output_dir, "bert_results.json"), "w") as f:
        json.dump({
            "macro_f1": results["eval_f1"],
            "accuracy": results["eval_accuracy"],
            "num_classes": num_labels,
            "classes": le.classes_.tolist(),
            "selection": "val split (10% of train), stratified; test used only for the reported score",
            "val_f1_per_epoch": val_trace,
            "best_checkpoint": best_ckpt,
            "epochs": epochs,
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    # Usage: bert_baseline.py <train.json> <test.json> [output_dir]
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "outputs/bert-baseline"

    print(f"Loading train: {train_path}")
    train_texts, train_labels = load_sharegpt(train_path)
    print(f"  {len(train_texts)} samples, {len(set(train_labels))} classes")
    print(f"Loading test: {test_path}")
    test_texts, test_labels = load_sharegpt(test_path)
    print(f"  {len(test_texts)} samples, {len(set(test_labels))} classes")

    os.makedirs(output_dir, exist_ok=True)
    train_bert(train_texts, train_labels, test_texts, test_labels, output_dir)
