# SALAD-v2 — Reproducibility Artifact Manifest

**Repository:** <https://github.com/nutthakorn7/salad2-benchmark>

Companion to *"SALAD-v2: A Leak-Screened Benchmark and Validated Methodology for
Real-Network-Flow SOC Alert Classification."*

The built SALAD-v2 corpus is **not** redistributed here. Both source captures come with
their own access terms, and rather than assert a redistribution right we have not had
independently confirmed, we ship the construction pipeline and the exact parameters needed
to rebuild the corpus byte-for-byte. This manifest is what makes "rebuild it yourself and
you will get our numbers" a checkable claim rather than a promise.

## 1. Source datasets and pinned revisions

| Arm | Source | Host | Revision (pinned) |
|---|---|---|---|
| UNSW-NB15 | Moustafa & Slay, MilCIS 2015 | Hugging Face `Mireu-Lab/UNSW-NB15` | `3eb02c4a` |
| CICIDS-2017 | Sharafaldin et al., ICISSP 2018 | Hugging Face `bvk/CICIDS-2017` | `b9515d5c` |

Revisions are pinned so the Hub cannot silently drift. A rebuild against a different
revision is not expected to reproduce the checksums below.

## 2. Built corpus checksums (SHA-256)

A correct rebuild reproduces these exactly. If your checksum differs, stop — the numbers
in the paper will not reproduce either, and the difference is upstream of every result.

```
f0ddaf88bd06994c32f8151f7ca71c24703ac0d624276d5d4a40770dd58fc11d  salad2_unsw/salad2_train.json     (5,000 rows)
b98c9cb420f0b48c50bd8818148f5d043316b469988ec65f73b7ea768e20ddb2  salad2_unsw/salad2_test.json      (10,000 rows)
86095b15594c60005597a5088e9ddee4ff3bcb5ddf5bd0a23db50c451f5159d0  salad2_cicids/salad2_train.json   (5,000 rows)
d866861a65f7d7bb6f728da9a69b107b94c40fd04a4c574ee76ea7900f3ab2db  salad2_cicids/salad2_test.json    (10,000 rows)
```

## 3. Scripts, and which claim each one backs

| Script | Produces | Paper location |
|---|---|---|
| `build_salad2.py` | Both arms, including the leakage gate (`check_leakage()`) and the contamination gate; refuses to write on failure | §III |
| `validate_leak_gate.py` | Re-certifies a built arm against both gate criteria | §III, §V |
| `salad2_baselines.py` | Decision Tree / LinearSVC / majority-floor point estimates | §VI-A |
| `salad2_baseline_stats.py` | Bootstrap CIs, paired LinearSVC-vs-DT test, per-class F1, per-row predictions | §VI-A |
| `bert_baseline.py` | BERT-base baseline (10% stratified val split from train; test scored once) | §VI-B |
| `bert_dump_preds.py` | Per-row BERT test predictions from a selected checkpoint | §VI-B |
| `mlx_salad2.py` | LoRA fine-tuning + evaluation for the 24-run ladder | §VI-B |

## 4. Fixed parameters

- **Seeds:** 42, 77, 123 (all three reported; no seed selection).
- **LLM ladder:** Qwen2.5-Instruct 0.5B / 1.5B / 3B / 7B, 4-bit community weights, LoRA on
  8 layers, 600 iterations. The 7B cells alone use learning-rate warmup with cosine decay
  after two constant-rate attempts diverged; this is disclosed in §VI-B and is why the
  ladder is described as four size-dependent configurations rather than a controlled
  scaling experiment.
- **Classical baselines:** `TfidfVectorizer(max_features=5000, stop_words="english",
  token_pattern=r"[^\s:]+")`; `DecisionTreeClassifier(max_depth=20, random_state=42)`;
  `LinearSVC(random_state=42)`.
- **BERT:** `bert-base-uncased`, 5 epochs, batch 16, lr 5e-5, warmup ratio 0.1, CPU,
  `seed=42`; checkpoint selected on a stratified 10% validation split carved from train.
- **Evaluation:** strict exact-match macro-F1, no alias normalisation, full 10,000-row test
  set per arm. Base-model (zero-shot) controls are the one exception and remain a 600-row
  subsample; this is marked in both result tables.

## 5. Hardware

All experiments ran on one Apple Silicon workstation (Mac Mini, M4, 32 GB unified memory).
LoRA fine-tuning and inference used MLX; BERT ran CPU-only. No HPC cluster was used.

## 6. What is *not* in this artifact

Stated explicitly so a reader does not go looking:

- The built corpus itself (see the note at the top).
- Per-row predictions for the 24 LLM ladder cells on the full 10,000-row test set. The
  ladder's reported macro-F1 values are complete, and per-row predictions exist for a
  600-row subsample only, so a paired significance test between an LLM cell and a classical
  baseline cannot be run from what is released here. The classical and BERT baselines *do*
  ship per-row predictions, so their pairwise comparisons are fully reproducible.
- Any grouped, temporal, or campaign-aware split. The contamination gate certifies exact
  rendered-prompt uniqueness only; see the Limitations section of the paper.
