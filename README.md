# SALAD-v2 — construction pipeline and reproducibility artifact

Code artifact for *"SALAD-v2: A Leak-Screened Benchmark and Validated Methodology for
Real-Network-Flow SOC Alert Classification."*

## What this is

Our earlier SOC alert-classification benchmark (SALAD v1, unpublished) turned out not to
measure classification. One of its prompt fields, `Alert Type`, was a deterministic one-to-one
rename of the gold label, so a Decision Tree given **that field alone** reached macro-F1 1.00 —
the same score as the full feature set. The defect came from the dataset's label-synthesis step,
not from its underlying network-flow source.

SALAD-v2 is the rebuild: two arms (UNSW-NB15 and CICIDS-2017) constructed directly from real
flow captures, behind two gates that **fail the build rather than warn**.

- **Leakage gate** — for every candidate field, asks whether that one field alone can
  reconstruct the label. Criterion A is a held-out single-feature macro-F1 test; Criterion B
  catches low-cardinality deterministic renames that a small-sample tree estimate would miss.
- **Contamination gate** — refuses the build if any rendered prompt appears on both sides of
  the train/test split.

On the rebuilt benchmark the paper compares LoRA-fine-tuned Qwen2.5 models with classical
baselines under two recipes, and the answer depends on the recipe:

- an untuned, low-budget 4-bit MLX ladder (24 runs, 0.5B–7B) scores below the Decision Tree and
  LinearSVC fitted on all 5,000 training rows on both arms (although on CICIDS-2017 11 of 12 cells
  exceed a Decision Tree fitted on only the rows they saw);
- a validation-tuned fp32 PyTorch ladder (12 runs, 0.5B and 1.5B) leaves LinearSVC ahead of all
  six UNSW-NB15 cells, while on CICIDS-2017 all six exceed the Decision Tree and none differs
  significantly from LinearSVC after multiplicity adjustment (this is not an equivalence claim).

The two recipes differ in many ways at once, so the runs do not identify which difference matters.

## What is NOT here

**The built corpus.** SALAD-v2 derives from two third-party captures whose own terms govern
redistribution, and we have not confirmed a right to redistribute a derived corpus, so we ship
the pipeline instead of the data. `ARTIFACT.md` pins the exact source revisions and publishes
SHA-256 checksums of the four files the paper scored. A fresh rebuild reproduces the two
**training** checksums but **not** the two test checksums (the shipped test files predate a fix to
the builder's random-number handling), so a rebuild scores a different, equally valid test draw.

Also absent: per-row predictions for the 24 untuned MLX cells (their macro-F1 values ship), the
MLX crash-resume supervisor, and model weights and logs. The per-row predictions of the 12 tuned
cells and of the classical and BERT baselines **do** ship, so every tuned-ladder score and paired
comparison recomputes without the corpus. `ARTIFACT.md` lists what reproduces and what does not.

## Layout

```
scripts/
  build_salad2.py           construction pipeline, including both gates
  validate_leak_gate.py     synthetic dose-response sweep and CICIDS-2017 feature ranking for the gate
  salad2_baselines.py       Decision Tree / LinearSVC / majority-floor point estimates
  salad2_baseline_stats.py  bootstrap CIs, per-class F1, per-row prediction dumps
  salad2_paired_tests.py    paired bootstrap across the non-LLM baselines
  bert_baseline.py          BERT-base baseline (val split from train; test scored once)
  bert_dump_preds.py        per-row BERT predictions from a selected checkpoint
  mlx_salad2.py             LoRA fine-tuning and evaluation for the untuned MLX ladder
  p6_sweep.py, analyze_p6_sweep.py   companion training-size study
dgx_ladder/                 the validation-tuned ladder: runner, queue files, per-row predictions,
                            analysis package, same-rows replay, fact-sheet script (see ARTIFACT.md §7)
results/                    derived metrics, gate-validation outputs, paired tests
ARTIFACT.md                 pinned revisions, checksums, hyperparameters, scope, what reproduces
```

## Reproducing

1. Obtain UNSW-NB15 and CICIDS-2017 under each dataset's own access terms.
2. `python3 scripts/build_salad2.py --dataset unsw` and `--dataset cicids` — both gates run during
   construction and fail the build on violation (output: `data/salad2_<arm>/`).
3. Check your **training**-file checksums against `ARTIFACT.md`; they should match. Test-file
   checksums will not (see above).
4. `mkdir -p results/salad2`, then `python3 scripts/salad2_baselines.py --dataset {unsw,cicids}` for the
   classical point estimates, and `salad2_baseline_stats.py` / `salad2_paired_tests.py --dataset ...` for
   intervals and paired comparisons. On your own rebuild these score a different test draw and will land
   close to, not exactly on, the paper's numbers.
5. `scripts/mlx_salad2.py` has three subcommands (`prep`, `train`, `eval`; `--help` for each) for the MLX
   ladder; `python3 scripts/bert_baseline.py <train.json> <test.json> <output_dir>` for the encoder baseline.
6. Without any corpus: `cd dgx_ladder/scripts && python3 -m dgx_analysis ladder --B 5000` recomputes the
   tuned-ladder scores and paired comparisons from the shipped per-row predictions (`ladder` defaults to
   B = 2,000; the paper's Table VI uses B = 5,000, and a count that depends on a borderline interval can
   differ between the two).
7. `python3 scripts/validate_leak_gate.py --synthetic` re-runs the gate's synthetic sweep (seconds, no
   downloads); `--cicids-engelen` needs the cached CICIDS-2017 source.

Seeds are 42, 77, and 123 throughout, and all three are reported — there is no seed selection.

## Citing

Please cite the paper. This repository is the code artifact accompanying it.
