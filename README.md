# SALAD-v2 — construction pipeline and reproducibility artifact

Code artifact for *"SALAD-v2: A Leak-Screened Benchmark and Validated Methodology for
Real-Network-Flow SOC Alert Classification."*

## What this is

A prior SOC alert-classification benchmark turned out not to measure classification. One of
its prompt fields, `Alert Type`, was a deterministic one-to-one rename of the gold label, so
a Decision Tree given **that field alone** reached macro-F1 1.00 — the same score as the full
feature set. The defect came from the dataset's label-synthesis step, not from its underlying
network-flow source.

SALAD-v2 is the rebuild: two arms (UNSW-NB15 and CICIDS-2017) constructed directly from real
flow captures, behind two gates that **fail the build rather than warn**.

- **Leakage gate** — for every candidate field, asks whether that one field alone can
  reconstruct the label. Criterion A is a held-out single-feature macro-F1 test; Criterion B
  catches low-cardinality deterministic renames that a small-sample tree estimate would miss.
- **Contamination gate** — refuses the build if any rendered prompt appears on both sides of
  the train/test split.

The paper's headline result is negative: across a 24-run LoRA ladder (Qwen2.5 0.5B–7B, both
arms, three seeds), no fine-tuned LLM cell reaches any classical baseline. BERT-base beats the
LLM ladder but still loses to a linear SVM on both arms.

## What is NOT here

**The built corpus.** SALAD-v2 derives from two third-party captures whose own terms govern
redistribution, and rather than assert a redistribution right we have not had independently
confirmed, we ship the pipeline instead of the data. `ARTIFACT.md` pins the exact source
revisions and publishes SHA-256 checksums for all four built splits, so a rebuild is
verifiable: if your checksums match, you have our corpus.

Also absent: per-row predictions for the 24 LLM ladder cells on the full test set. Their
macro-F1 values are complete and reported, but without per-row output no paired significance
test between an LLM cell and a classical baseline is possible from this artifact. The
classical and BERT baselines **do** ship per-row predictions, so their comparisons are fully
reproducible here.

## Layout

```
scripts/
  build_salad2.py           construction pipeline, including both gates
  validate_leak_gate.py     re-certify a built arm against both criteria
  salad2_baselines.py       Decision Tree / LinearSVC / majority-floor point estimates
  salad2_baseline_stats.py  bootstrap CIs, per-class F1, per-row prediction dumps
  salad2_paired_tests.py    paired bootstrap across the non-LLM baselines
  bert_baseline.py          BERT-base baseline (val split from train; test scored once)
  bert_dump_preds.py        per-row BERT predictions from a selected checkpoint
  mlx_salad2.py             LoRA fine-tuning and evaluation for the ladder
results/                    the derived metrics reported in the paper
ARTIFACT.md                 pinned revisions, checksums, hyperparameters, scope
```

## Reproducing

1. Obtain UNSW-NB15 and CICIDS-2017 under each dataset's own access terms.
2. `python3 scripts/build_salad2.py` — both gates run during construction and fail the build
   on violation.
3. Check your split checksums against `ARTIFACT.md`. They must match before anything else.
4. `python3 scripts/salad2_baselines.py --dataset {unsw,cicids}` for the classical point
   estimates, then `salad2_baseline_stats.py` and `salad2_paired_tests.py` for intervals and
   paired comparisons.
5. `python3 scripts/mlx_salad2.py` for the LoRA ladder; `bert_baseline.py` for the encoder
   baseline.

Seeds are 42, 77, and 123 throughout, and all three are reported — there is no seed selection.

## Citing

Please cite the paper. This repository is the code artifact accompanying it.
