# SALAD-v2 — Reproducibility Artifact Manifest

**Repository:** <https://github.com/nutthakorn7/salad2-benchmark>

Companion to *"SALAD-v2: A Leak-Screened Benchmark and Validated Methodology for
Real-Network-Flow SOC Alert Classification."*

The built SALAD-v2 corpus is **not** redistributed here. Both source captures come with
their own access terms, and we have not confirmed a right to redistribute a derived corpus.
We ship the construction pipeline, both gates, the exact source revisions and parameters, and
the per-row predictions behind the tuned LoRA ladder.

## What reproduces, and what does not

| You can | from |
|---|---|
| rebuild the two **training** files byte-for-byte (checksums in §2) | the pipeline and pinned revisions, with your own licensed copies of the sources |
| re-run both gates and compare the gate statistics with Table I of the paper | the same |
| recompute every tuned-ladder F1 and every paired comparison (Tables IV–VI) | the shipped per-row predictions in `dgx_ladder/`, **without the corpus** |
| re-check the classical and BERT baselines' comparisons | the shipped per-row predictions |

| You cannot | because |
|---|---|
| regenerate the exact scored **test** files | In our re-check a fresh build from the pinned revisions reproduced both training checksums and **neither** test checksum: the shipped test files predate a later fix to the builder's random-number handling, and the builder's `--reuse-test` option needs the shipped file, which we do not redistribute. A rebuild scores a different, equally valid test draw, so baseline numbers land close to, not exactly on, the reported ones. |
| retrain a tuned or MLX cell and match the paper's score exactly | it needs the corpus, and the test-split limitation above applies |

## 1. Source datasets and pinned revisions

| Arm | Source | Host | Revision (pinned) |
|---|---|---|---|
| UNSW-NB15 | Moustafa & Slay, MilCIS 2015 | Hugging Face `Mireu-Lab/UNSW-NB15` | `3eb02c4a` |
| CICIDS-2017 | Sharafaldin et al., ICISSP 2018 | Hugging Face `bvk/CICIDS-2017` | `b9515d5c` |

Revisions are pinned so the Hub cannot silently drift. A rebuild against a different
revision is not expected to reproduce the training checksums below.

## 2. Reference checksums (SHA-256) of the four files the paper scored

```
f0ddaf88bd06994c32f8151f7ca71c24703ac0d624276d5d4a40770dd58fc11d  salad2_unsw/salad2_train.json     (5,000 rows)
b98c9cb420f0b48c50bd8818148f5d043316b469988ec65f73b7ea768e20ddb2  salad2_unsw/salad2_test.json      (10,000 rows)
86095b15594c60005597a5088e9ddee4ff3bcb5ddf5bd0a23db50c451f5159d0  salad2_cicids/salad2_train.json   (5,000 rows)
d866861a65f7d7bb6f728da9a69b107b94c40fd04a4c574ee76ea7900f3ab2db  salad2_cicids/salad2_test.json    (10,000 rows)
```

The two training checksums are expected to match a correct rebuild; the two test checksums
are reference identifiers of the files that were scored and are **not** expected to match
(see the table above). Per-row predictions in this repository are indexed by row position
in those scored test files.

## 3. Scripts, and which claim each one backs

| Script | Produces | Paper location |
|---|---|---|
| `scripts/build_salad2.py` | Both arms, including the leakage gate (`check_leakage()`) and the contamination gate; refuses to write on failure | §III |
| `scripts/validate_leak_gate.py` | `--synthetic`: the dose-response sweep (output: `results/leak_gate_synthetic_validation.json`); `--cicids-engelen`: the single-feature ranking of CICIDS-2017 (output: `results/leak_gate_cicids_engelen_validation.json`, needs the cached source) | §IV, Tables II–III |
| `scripts/salad2_baselines.py` | Decision Tree / LinearSVC / majority-floor point estimates | §VI-A |
| `scripts/salad2_baseline_stats.py` | Bootstrap CIs, paired LinearSVC-vs-DT test, per-class F1, per-row predictions | §VI-A |
| `scripts/salad2_paired_tests.py` | Paired bootstrap among Decision Tree, LinearSVC and BERT-base | §VI-A, §VI-B |
| `scripts/bert_baseline.py` | BERT-base baseline (10% stratified validation split from train; checkpoint selected on it; test scored once) | §VI-B |
| `scripts/bert_dump_preds.py` | Per-row BERT test predictions from the selected checkpoint | §VI-B |
| `scripts/mlx_salad2.py` | LoRA fine-tuning and evaluation for the 24-run untuned MLX ladder | §VI-B |
| `dgx_ladder/scripts/dgx/attrib.py`, `run_lane.sh`, `queue_missing.py`, `queues_snapshot/` | PyTorch fp32 runner, crash-resume driver and the queue files of the validation-tuned ladder and its tuning grid | §VI-B |
| `dgx_ladder/scripts/dgx_analysis/` (`python3 -m dgx_analysis ladder`) | Recomputes each tuned cell's F1 from its per-row predictions and runs the paired bootstrap against Decision Tree, LinearSVC and BERT-base | §VI-B, Tables IV–VI |
| `dgx_ladder/scripts/p13_fact_sheet.py` | Writes the fact sheet of every ladder number, count, tuning-grid and resume fact quoted in the paper. Its `--check` flag is a numeric-token check of a manuscript (it does not verify seeds, signs or counted claims) | §VI-B |
| `dgx_ladder/results/salad2/p13_same_rows/` | Same-rows replay of the MLX ladder: the row indices each MLX cell saw, and the Decision Tree / LinearSVC fitted on exactly those rows (re-running it needs the corpus) | §VI-B |
| `scripts/p6_sweep.py`, `scripts/analyze_p6_sweep.py`, `dgx_ladder/scripts/p6_matched_baselines.py` | Companion training-size study (not used by this paper's claims): the sweep, its analysis, and training-matched classical baselines | — |

## 4. Fixed parameters

- **Seeds:** 42, 77, 123 (all three reported; no seed selection).
- **Tuned ladder (`dgx_ladder/`):** Qwen2.5-Instruct 0.5B / 1.5B, fp32, PyTorch 2.9.1; LoRA rank
  16, scale 2, on `q_proj`/`v_proj` of every layer; AdamW (PyTorch defaults), cosine decay, no
  warmup, batch 4, loss over the full padded sequence, plain-text prompts (no system prompt,
  1,536-token limit), greedy 16-token decoding. lr and epochs come from the minimum validation
  answer-token NLL of a 10–11-cell grid on the 0.5B model (seed 42, 500 rows held out), adopted
  unchanged for both sizes, then every final cell retrains on all 5,000 rows: UNSW-NB15 lr 5e-4
  × 12 epochs, CICIDS-2017 lr 1e-3 × 6 epochs. Four of the 12 cells resumed from a checkpoint
  after an interruption; each cell's own JSON records it.
- **Untuned MLX ladder:** Qwen2.5-Instruct 0.5B / 1.5B / 3B / 7B, 4-bit community weights, LoRA
  rank 8 (scale 20) on the last eight blocks, constant Adam 1e-5, batch 4, 600 steps over the first
  4,500 training rows, a system prompt listing the valid labels. Only 3 of the 24 cells ran as one
  uninterrupted 600-step run (UNSW-NB15 seed 42 at 0.5B, 1.5B and 3B); the UNSW-NB15 7B seed-42 cell is
  300 + 300 steps; the other 20 ran as six consecutive 100-step invocations that each replayed the
  same first 100 batches (≈400 distinct rows seen six times). The 7B cells use warmup with cosine
  decay after constant-rate attempts diverged. See §VI-B of the paper.
- **Classical baselines:** `TfidfVectorizer(max_features=5000, stop_words="english",
  token_pattern=r"[^\s:]+")`; `DecisionTreeClassifier(max_depth=20, random_state=42)`;
  `LinearSVC(random_state=42)`.
- **BERT:** `bert-base-uncased`, 5 epochs, batch 16, lr 5e-5, warmup ratio 0.1, CPU, `seed=42`;
  trained on 4,500 rows, checkpoint selected on a stratified 10% validation split carved from train.
- **Evaluation:** strict exact-match macro-F1, no alias normalisation, full 10,000-row test set
  per arm. The zero-shot base-model controls are the one exception: a 600-row subsample, reported
  in the text of §VI-B (not in the result tables); their predictions are not shipped.

## 5. Hardware

The MLX ladder and the dataset construction ran on one Apple Silicon workstation (Mac Mini, M4,
32 GB unified memory); BERT ran CPU-only. The tuned ladder ran in fp32 with PyTorch 2.9.1 on a
local NVIDIA GB10 workstation. No HPC cluster was used.

## 6. What is *not* in this artifact

Stated explicitly so a reader does not go looking:

- The built corpus itself, and therefore the exact scored test files (see the top of this page).
- Per-row predictions for the 24 MLX ladder cells on the full 10,000-row test set; their macro-F1
  values ship in `dgx_ladder/results/salad2/full_test/`. No paired test between an MLX cell and a
  classical baseline can be run from this artifact.
- The crash-resume supervisor that ran 20 of the 24 MLX cells (the paper specifies its schedule;
  reproducing those cells requires reimplementing it).
- Model weights, adapters, checkpoints and training logs; batch-size-1 generation outputs (only
  the recorded agreement counts in each cell's JSON).
- The `verify` checks of the analysis package (they compare predictions with the corpus's gold
  labels, so they need a reader's own rebuild).
- SALAD v1 (our earlier, unpublished benchmark) and the forensic and relabeling analyses the
  Introduction mentions; the Hugging Face model revisions of the Qwen weights are not pinned (the
  runner fetches by name and defaults to `HF_HUB_OFFLINE=1`).
- Any grouped, temporal, or campaign-aware split. The contamination gate certifies exact
  rendered-prompt uniqueness only; see the Limitations section of the paper.

## 7. Running the tuned-ladder analysis (`dgx_ladder/`)

```
cd dgx_ladder/scripts
python3 -m dgx_analysis ladder --B 5000      # numpy only (the paper's Table VI uses B = 5,000; default 2,000); writes ../results/salad2/dgx/analysis/ladder.{json,md}
python3 p13_fact_sheet.py                   # the fact sheet quoted in the paper (set DGX_ANALYSIS_OUT to redirect output)
```

The runner and driver are shipped as they ran on our workstation: `run_lane.sh` changes to
`~/p22`, uses a virtual environment there, and writes to `results/attrib/` (shipped results are
under `dgx_ladder/results/salad2/dgx/attrib/`). Adapt the paths before re-running them. No
licence is granted over the third-party datasets.
