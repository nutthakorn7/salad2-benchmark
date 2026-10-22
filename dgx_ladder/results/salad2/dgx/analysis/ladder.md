# Tuned model-size ladder (LT_ cells)

Test set: each arm's 10,000 shared test rows (never validation). Cells: `LT_<arm>_<size>_lora_r16_s<seed>` scored on the full test set; strict macro-F1 = mean per-class F1 over the gold-present labels, zero_division = 0. n = **12 comparable cells** of 12 expected (expected set from the queue snapshot (scripts/dgx/queues_snapshot/*/*.json)). Paired bootstrap: B = 5,000, seed 0, 36 cell x baseline tests.

## Summary

- 12 of 12 expected LT cells are present and comparable on the shared 10,000 test rows (0 missing, 0 present but not comparable).
- unsw: 6 comparable cells over sizes 0.5b, 1.5b; best single LLM cell LT_unsw_1.5b_lora_r16_s77 = 0.6164 against DT 0.5874, LinearSVC 0.6505, BERT-REFIT 0.5946 (the best cell is picked on the test scores: descriptive, post-hoc).
- unsw: recipe = lora r16 alpha 32.0, adamw + cosine, lr 0.0005, 12 epochs = 15000 steps, adopted from `T_unsw_lora_r16_lr5e-4_e12_v500` in 6 of 6 cells (read from each cell's config.tuned_resolved; see the recipe table).
- unsw: of 18 cell x baseline paired-bootstrap tests, the 95% CI excludes 0 in favour of the LLM in 5, in favour of the baseline in 11, neither in 2 (unadjusted); under Holm-adjusted p < 0.05 (Wald-z p from the bootstrap SD, Holm over all 36 tests): LLM 3, baseline 9, neither 6.
- unsw: highest seed-mean size = 1.5b (0.6097, n = 3 seed(s); post-hoc); peak claim: NO, size(s) 3b, 7b absent: a peak over the ladder cannot be claimed.
- unsw: shape: every step is up in every matched seed (increasing in size in each seed; direction counts only, not a tested trend).
- cicids: 6 comparable cells over sizes 0.5b, 1.5b; best single LLM cell LT_cicids_0.5b_lora_r16_s77 = 0.9656 against DT 0.9310, LinearSVC 0.9636, BERT-REFIT 0.9532 (the best cell is picked on the test scores: descriptive, post-hoc).
- cicids: recipe = lora r16 alpha 32.0, adamw + cosine, lr 0.001, 6 epochs = 7500 steps, adopted from `T_cicids_lora_r16_lr1e-3_e6_v500` in 6 of 6 cells (read from each cell's config.tuned_resolved; see the recipe table).
- cicids: of 18 cell x baseline paired-bootstrap tests, the 95% CI excludes 0 in favour of the LLM in 11, in favour of the baseline in 1, neither in 6 (unadjusted); under Holm-adjusted p < 0.05 (Wald-z p from the bootstrap SD, Holm over all 36 tests): LLM 9, baseline 0, neither 9.
- cicids: highest seed-mean size = 0.5b (0.9625, n = 3 seed(s); post-hoc); peak claim: NO, size(s) 3b, 7b absent: a peak over the ladder cannot be claimed.
- cicids: shape: NOT monotone in every seed (the step directions are not all the same across steps and seeds; a tie counts as no direction): no monotone claim is made.

## Read this first (caveats)

1. **Baseline exposure.** DT and LinearSVC are one deterministic fit each on the arm's whole training file (`train_file` in `<arm>_baselines.json` = `data/salad2_<arm>/salad2_train.json`; rows counted here: unsw 5,000, cicids 5,000), TF-IDF over the rendered flow-text prompt. BERT-REFIT is NOT trained on all of those rows: its result file (`bert_<arm>_results.REFIT.json`) records the selection protocol ('val split (10% of train), stratified; test used only for the reported score') and the epochs (unsw 5, cicids 5) but not the number of training rows; the P13 manuscript (Section VI-B) states it was trained from 4,500 rows per arm (the other 500 held out for selection). Treat BERT's exposure as 4,500 rows, not 5,000.
2. **LLM exposure (matched on distinct rows, not on passes).** 12 of 12 comparable LT cells have `val = 0` and `steps = 0`, so each trains on all 5,000 training rows for E epochs: (5000 // 4) x E = 1,250 x E optimizer steps (`scripts/dgx/attrib.py:85-90, 116`), reshuffled every epoch with `Random(seed + epoch)` (`attrib.py:127`), in one training loop per cell: 4 of 12 comparable cells resumed from a checkpoint after an interruption (`resumed_from_step` in the cell's own JSON: LT_cicids_0.5b_lora_r16_s42 at step 4000, LT_cicids_0.5b_lora_r16_s77 at step 3000, LT_cicids_1.5b_lora_r16_s42 at step 5000, LT_unsw_1.5b_lora_r16_s77 at step 9500); a resume restores adapter, optimizer and scheduler state and replays the epoch shuffles, but is not bit-identical to an uninterrupted run; the field is absent from 2 cell(s) that predate it (LT_unsw_0.5b_lora_r16_s42, LT_unsw_0.5b_lora_r16_s77), which are counted as not resumed - that rests on the lane's driver log, which is not part of these result files. So the batch-replay defect of P13's MLX ladder (20 of 24 cells replayed the same first 100 batches, about 400 distinct rows seen six times: the P13 manuscript, Section VI-B) does not apply here, and the DT/SVC baselines and the LLM see the same set of distinct rows. The LLM sees each row E times, DT/SVC once, BERT a few epochs over 4,500 rows: 'matched exposure' means the same rows, not the same number of passes.
3. **Tuned on 0.5b only.** The adopted lr / epochs were selected on validation answer-NLL of 0.5b tuning cells (seed(s) 42, one run per grid point, 500 held-out TRAIN rows, the model trained on the rest) and adopted UNCHANGED for every size on the ladder: the optimum can move with model size, so a size trend here mixes size with a recipe selected on the smallest model, which need not be right for the larger ones. The adopted grid point sits at an edge of the tried grid (`T_cicids_lora_r16_lr1e-3_e6_v500` (max-epochs-edge, lr-slice-upper-edge); `T_unsw_lora_r16_lr5e-4_e12_v500` (max-epochs-edge, lr-slice-upper-edge)), so it was not bracketed and a better setting may lie beyond it (see `tuning.md`). Also, each LT cell takes 1,250 x E steps against 1,125 x E for the grid point it adopted (about 11% more, because it trains on 5,000 rows, not 4,500).
4. **Execution differs by size (not part of the recipe).** `micro` and `eval_bs` vary across sizes (see the execution table); with `mask_pad = 0` the loss is taken over padding too and padding and the loss mean are computed per micro-batch (`attrib.py:118-141`), so `micro = 2` is not numerically identical to `micro = 0`. Size is therefore partly confounded with micro-batching; `eval_bs_check` (batched vs batch-1 generation) is shown per size.
5. **Uncertainty is partial.** Paired-bootstrap CIs (B = 5,000, seed 0, rows resampled jointly, label set fixed from the full gold, the SAME resamples for every test) measure test-row sampling of ONE trained model against ONE deterministic baseline fit; they do not include training-seed variance. Seed spread comes from up to 3 training seeds per size (the design; n per size is in every table) on a SINGLE fixed 5,000-row training file (it excludes variance from re-drawing the training rows), and seeds are matched across sizes by seed number only (the different models share nothing but the data-order seed), so 'per seed' orderings are matched-index comparisons, not true pairings.
6. **Small n.** With 3 seeds per size (the design) an exact seed-level test cannot go below p = 0.10 (Mann-Whitney) or 0.25 (paired sign test), so no adjacent-size difference can be significant at 0.05 on seed direction alone; the step counts are descriptive. Exact tests are run only when both sizes have >= 3 seeds.
7. **Multiplicity.** 36 cell x baseline bootstrap tests in this report. The seed counts by CI use unadjusted 95% intervals (the primary, requested summary). The Holm-adjusted counts beside them use Holm over all 36 tests on a normal-approximation p (z = difference / bootstrap SD): the add-one bootstrap p has resolution 2/(B+1) = 4.0e-04, so after Holm over 36 tests its smallest attainable adjusted p is 0.014 (< 0.05: usable, but shown only in the JSON). A Holm-adjusted direction is never allowed to be stronger than the unadjusted CI beside it: 0 test(s) whose z-based Holm p < 0.05 while the percentile CI spans 0 were reset to 'neither'. 2 adjacent-size Mann-Whitney tests are Holm-adjusted separately.
8. **Macro-F1 is noisy on rare classes.** Macro-F1 averages over the gold-present classes, so a class with few test rows moves it a lot: unsw: K = 10, smallest class `Worms` has 120 of 10,000 test rows; cicids: K = 8, smallest class `WebAttack` has 184 of 10,000 test rows.

## Coverage

Directories scanned: `dgx_ladder/results/salad2/dgx/attrib`; `dgx_ladder/results/salad2/dgx/p22_attrib` (DOES NOT EXIST).

> queue snapshot has 12 LT_ names vs 24 in the 2 x 4 x 3 design grid; only in snapshot: []; only in design: ['LT_cicids_3b_lora_r16_s123', 'LT_cicids_3b_lora_r16_s42', 'LT_cicids_3b_lora_r16_s77', 'LT_cicids_7b_lora_r16_s123', 'LT_cicids_7b_lora_r16_s42', 'LT_cicids_7b_lora_r16_s77', 'LT_unsw_3b_lora_r16_s123', 'LT_unsw_3b_lora_r16_s42', 'LT_unsw_3b_lora_r16_s77', 'LT_unsw_7b_lora_r16_s123', 'LT_unsw_7b_lora_r16_s42', 'LT_unsw_7b_lora_r16_s77']

| arm | size | s42 | s77 | s123 |
| :--- | :--- | :--- | :--- | :--- |
| cicids | 0.5b | ok | ok | ok |
| cicids | 1.5b | ok | ok | ok |
| unsw | 0.5b | ok | ok | ok |
| unsw | 1.5b | ok | ok | ok |

## Reference baselines (canonical, same 10,000 rows)

n = 10,000 test rows per arm; one deterministic fit per baseline (no seed spread). DT / LinearSVC: `<arm>_baseline_preds.json`; BERT-REFIT: `<arm>_bert_preds.json` (validation-selected epochs, trained on 4,500 rows: see caveat 1).

| arm | baseline | macro-F1 (recomputed) | reported | note |
| :--- | :--- | ---: | ---: | :--- |
| unsw | DT | 0.5874 | 0.5874 |  |
| unsw | LinearSVC | 0.6505 | 0.6505 |  |
| unsw | BERT-REFIT | 0.5946 | 0.5946 |  |
| cicids | DT | 0.9310 | 0.9310 |  |
| cicids | LinearSVC | 0.9636 | 0.9636 |  |
| cicids | BERT-REFIT | 0.9532 | 0.9532 |  |

## Arm: unsw

### unsw: recipe each cell adopted (`config.tuned_resolved`, read per cell)

Constant across all cells: `method=lora`, `rank=16`, `alpha=32.0`, `sched=cosine`, `opt=adamw`, `layers=all`, `rslora=0`, `mask_pad=0`, `dtype=fp32`, `val=0`, `steps=0`, `tuned=unsw:lora`, `tuned_use=lr+epochs`.

| adopted from (T_ cell) | lr | epochs | total steps | val NLL | grid size at launch | fallback | cells | which |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| T_unsw_lora_r16_lr5e-4_e12_v500 | 0.0005 | 12 | 15000 | 0.1331 | 11 | no | 6 | 0.5b: 42,77,123; 1.5b: 42,77,123 |

Adopted tuning cell(s) ran on model(s) **0.5b**, seed(s) 42, validation hold-out 500 rows.

Execution per size (not recipe; sizes differ here):

| size | cells | micro | eval_bs | dtype | eval_bs_check identical / rows | cells w/o check |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: |
| 0.5b | 3 | 0 | 32 | fp32 | 300 / 300 | 0 |
| 1.5b | 3 | 0 | 32 | fp32 | 300 / 300 | 0 |

### unsw: strict macro-F1 by size (n seeds, mean, sample SD)

| size | n seeds | mean F1 | SD (ddof=1) | per seed | off-vocab mean | off-vocab max |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| 0.5b | 3 | 0.5698 | 0.0121 | s42 0.5575, s77 0.5702, s123 0.5817 | 0.0000 | 0.0000 |
| 1.5b | 3 | 0.6097 | 0.0062 | s42 0.6085, s77 0.6164, s123 0.6043 | 0.0002 | 0.0004 |

### unsw: point difference to the canonical baselines (mean LLM F1 - baseline F1) and paired bootstrap per cell

| size | n | mean F1 | vs DT (0.5874) | vs LinearSVC (0.6505) | vs BERT-REFIT (0.5946) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 0.5b | 3 | 0.5698 | -0.0177 | -0.0807 | -0.0249 |
| 1.5b | 3 | 0.6097 | +0.0223 | -0.0407 | +0.0151 |

Per cell (difference LLM - baseline [95% percentile CI, test rows resampled jointly]; `LLM` / `base` / `ns` = CI excludes 0 for the LLM / for the baseline / spans 0; `f>0` = fraction of the bootstrap resamples in which the LLM cell scores higher):

| size | seed | F1 | vs DT | vs LinearSVC | vs BERT-REFIT |
| :--- | ---: | ---: | :--- | :--- | :--- |
| 0.5b | 42 | 0.5575 | -0.0299 [-0.0449, -0.0151] base, f>0 0.000 | -0.0930 [-0.1079, -0.0783] base, f>0 0.000 | -0.0371 [-0.0517, -0.0222] base, f>0 0.000 |
| 0.5b | 77 | 0.5702 | -0.0173 [-0.0318, -0.0036] base, f>0 0.007 | -0.0803 [-0.0943, -0.0667] base, f>0 0.000 | -0.0245 [-0.0391, -0.0105] base, f>0 0.000 |
| 0.5b | 123 | 0.5817 | -0.0058 [-0.0186, +0.0072] ns, f>0 0.186 | -0.0688 [-0.0818, -0.0557] base, f>0 0.000 | -0.0130 [-0.0253, -0.0009] base, f>0 0.018 |
| 1.5b | 42 | 0.6085 | +0.0211 [+0.0081, +0.0334] LLM, f>0 1.000 | -0.0420 [-0.0546, -0.0291] base, f>0 0.000 | +0.0139 [+0.0019, +0.0256] LLM, f>0 0.990 |
| 1.5b | 77 | 0.6164 | +0.0290 [+0.0171, +0.0413] LLM, f>0 1.000 | -0.0341 [-0.0463, -0.0220] base, f>0 0.000 | +0.0218 [+0.0096, +0.0336] LLM, f>0 1.000 |
| 1.5b | 123 | 0.6043 | +0.0168 [+0.0034, +0.0299] LLM, f>0 0.994 | -0.0462 [-0.0583, -0.0343] base, f>0 0.000 | +0.0097 [-0.0023, +0.0213] ns, f>0 0.941 |

### unsw: wins matrix (# seeds whose 95% CI excludes 0 in favour of the LLM / of the baseline / neither; unadjusted CIs)

| size | n seeds | DT | LinearSVC | BERT-REFIT |
| :--- | ---: | :--- | :--- | :--- |
| 0.5b | 3 | 0 / 2 / 1 -- mixed / unresolved (unadjusted) | 0 / 3 / 0 -- baseline ahead in all seeds (unadjusted) | 0 / 3 / 0 -- baseline ahead in all seeds (unadjusted) |
| 1.5b | 3 | 3 / 0 / 0 -- LLM ahead in all seeds (unadjusted) | 0 / 3 / 0 -- baseline ahead in all seeds (unadjusted) | 2 / 0 / 1 -- mixed / unresolved (unadjusted) |

Same counts under Holm-adjusted p < 0.05 (Wald-z p from the bootstrap SD, Holm across all 36 tests in the report; where the two tables give different verdict words, the Holm one is the multiplicity-corrected reading):

| size | n seeds | DT | LinearSVC | BERT-REFIT |
| :--- | ---: | :--- | :--- | :--- |
| 0.5b | 3 | 0 / 1 / 2 -- mixed / unresolved (Holm) | 0 / 3 / 0 -- baseline ahead in all seeds (Holm) | 0 / 2 / 1 -- mixed / unresolved (Holm) |
| 1.5b | 3 | 2 / 0 / 1 -- mixed / unresolved (Holm) | 0 / 3 / 0 -- baseline ahead in all seeds (Holm) | 1 / 0 / 2 -- mixed / unresolved (Holm) |

### unsw: ladder ordering per seed

Sizes absent from this arm: 3b, 7b (orderings are over the sizes present: 0.5b, 1.5b).

| seed | complete | argmax size | ranking (F1) |
| :--- | :--- | :--- | :--- |
| s42 | yes | 1.5b | 1.5b (0.6085) > 0.5b (0.5575) |
| s77 | yes | 1.5b | 1.5b (0.6164) > 0.5b (0.5702) |
| s123 | yes | 1.5b | 1.5b (0.6043) > 0.5b (0.5817) |

Per-seed argmax agreement: 3 of 3 complete seeds agree on 1.5b (1.5b: 3).

**Peak claim: NO -- size(s) 3b, 7b absent: a peak over the ladder cannot be claimed.**

Adjacent-size steps, direction per matched seed (up = the larger model scores higher). **every step is up in every matched seed (increasing in size in each seed; direction counts only, not a tested trend).**

| step | matched seeds | up / down / tie | mean difference | exact Mann-Whitney p (two-sided) | min sign-test p |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 0.5b -> 1.5b | 3 | 3 / 0 / 0 | +0.0399 | 0.1000 (Holm 0.2000; min attainable 0.100; 20 permutations) | 0.250 |

## Arm: cicids

### cicids: recipe each cell adopted (`config.tuned_resolved`, read per cell)

Constant across all cells: `method=lora`, `rank=16`, `alpha=32.0`, `sched=cosine`, `opt=adamw`, `layers=all`, `rslora=0`, `mask_pad=0`, `dtype=fp32`, `val=0`, `steps=0`, `tuned=cicids:lora`, `tuned_use=lr+epochs`.

| adopted from (T_ cell) | lr | epochs | total steps | val NLL | grid size at launch | fallback | cells | which |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| T_cicids_lora_r16_lr1e-3_e6_v500 | 0.001 | 6 | 7500 | 0.0101 | 10, 11 | no | 6 | 0.5b: 42,77,123; 1.5b: 42,77,123 |

- INFO cicids:lora: `T_cicids_lora_r16_lr1e-3_e6_v500` is still the winner, but the grid has grown from 10, 11 to 11 candidates since launch

Adopted tuning cell(s) ran on model(s) **0.5b**, seed(s) 42, validation hold-out 500 rows.

Execution per size (not recipe; sizes differ here):

| size | cells | micro | eval_bs | dtype | eval_bs_check identical / rows | cells w/o check |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: |
| 0.5b | 3 | 0 | 16 | fp32 | 300 / 300 | 0 |
| 1.5b | 3 | 2 | 16 | fp32 | 300 / 300 | 0 |

### cicids: strict macro-F1 by size (n seeds, mean, sample SD)

| size | n seeds | mean F1 | SD (ddof=1) | per seed | off-vocab mean | off-vocab max |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| 0.5b | 3 | 0.9625 | 0.0029 | s42 0.9620, s77 0.9656, s123 0.9599 | 0.0000 | 0.0000 |
| 1.5b | 3 | 0.9612 | 0.0024 | s42 0.9630, s77 0.9622, s123 0.9585 | 0.0000 | 0.0001 |

### cicids: point difference to the canonical baselines (mean LLM F1 - baseline F1) and paired bootstrap per cell

| size | n | mean F1 | vs DT (0.9310) | vs LinearSVC (0.9636) | vs BERT-REFIT (0.9532) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 0.5b | 3 | 0.9625 | +0.0314 | -0.0012 | +0.0093 |
| 1.5b | 3 | 0.9612 | +0.0302 | -0.0024 | +0.0080 |

Per cell (difference LLM - baseline [95% percentile CI, test rows resampled jointly]; `LLM` / `base` / `ns` = CI excludes 0 for the LLM / for the baseline / spans 0; `f>0` = fraction of the bootstrap resamples in which the LLM cell scores higher):

| size | seed | F1 | vs DT | vs LinearSVC | vs BERT-REFIT |
| :--- | ---: | ---: | :--- | :--- | :--- |
| 0.5b | 42 | 0.9620 | +0.0309 [+0.0235, +0.0391] LLM, f>0 1.000 | -0.0017 [-0.0066, +0.0034] ns, f>0 0.261 | +0.0088 [+0.0029, +0.0149] LLM, f>0 0.999 |
| 0.5b | 77 | 0.9656 | +0.0345 [+0.0273, +0.0424] LLM, f>0 1.000 | +0.0019 [-0.0028, +0.0068] ns, f>0 0.788 | +0.0123 [+0.0068, +0.0185] LLM, f>0 1.000 |
| 0.5b | 123 | 0.9599 | +0.0288 [+0.0215, +0.0366] LLM, f>0 1.000 | -0.0038 [-0.0086, +0.0013] ns, f>0 0.066 | +0.0067 [+0.0013, +0.0124] LLM, f>0 0.992 |
| 1.5b | 42 | 0.9630 | +0.0320 [+0.0246, +0.0398] LLM, f>0 1.000 | -0.0006 [-0.0056, +0.0046] ns, f>0 0.414 | +0.0098 [+0.0040, +0.0161] LLM, f>0 1.000 |
| 1.5b | 77 | 0.9622 | +0.0311 [+0.0238, +0.0392] LLM, f>0 1.000 | -0.0014 [-0.0064, +0.0037] ns, f>0 0.289 | +0.0090 [+0.0033, +0.0151] LLM, f>0 1.000 |
| 1.5b | 123 | 0.9585 | +0.0274 [+0.0202, +0.0351] LLM, f>0 1.000 | -0.0052 [-0.0101, -0.0001] base, f>0 0.024 | +0.0053 [-0.0004, +0.0112] ns, f>0 0.967 |

### cicids: wins matrix (# seeds whose 95% CI excludes 0 in favour of the LLM / of the baseline / neither; unadjusted CIs)

| size | n seeds | DT | LinearSVC | BERT-REFIT |
| :--- | ---: | :--- | :--- | :--- |
| 0.5b | 3 | 3 / 0 / 0 -- LLM ahead in all seeds (unadjusted) | 0 / 0 / 3 -- mixed / unresolved (unadjusted) | 3 / 0 / 0 -- LLM ahead in all seeds (unadjusted) |
| 1.5b | 3 | 3 / 0 / 0 -- LLM ahead in all seeds (unadjusted) | 0 / 1 / 2 -- mixed / unresolved (unadjusted) | 2 / 0 / 1 -- mixed / unresolved (unadjusted) |

Same counts under Holm-adjusted p < 0.05 (Wald-z p from the bootstrap SD, Holm across all 36 tests in the report; where the two tables give different verdict words, the Holm one is the multiplicity-corrected reading):

| size | n seeds | DT | LinearSVC | BERT-REFIT |
| :--- | ---: | :--- | :--- | :--- |
| 0.5b | 3 | 3 / 0 / 0 -- LLM ahead in all seeds (Holm) | 0 / 0 / 3 -- mixed / unresolved (Holm) | 1 / 0 / 2 -- mixed / unresolved (Holm) |
| 1.5b | 3 | 3 / 0 / 0 -- LLM ahead in all seeds (Holm) | 0 / 0 / 3 -- mixed / unresolved (Holm) | 2 / 0 / 1 -- mixed / unresolved (Holm) |

### cicids: ladder ordering per seed

Sizes absent from this arm: 3b, 7b (orderings are over the sizes present: 0.5b, 1.5b).

| seed | complete | argmax size | ranking (F1) |
| :--- | :--- | :--- | :--- |
| s42 | yes | 1.5b | 1.5b (0.9630) > 0.5b (0.9620) |
| s77 | yes | 0.5b | 0.5b (0.9656) > 1.5b (0.9622) |
| s123 | yes | 0.5b | 0.5b (0.9599) > 1.5b (0.9585) |

Per-seed argmax agreement: 2 of 3 complete seeds agree on 0.5b (1.5b: 1, 0.5b: 2).

**Peak claim: NO -- size(s) 3b, 7b absent: a peak over the ladder cannot be claimed.**

Adjacent-size steps, direction per matched seed (up = the larger model scores higher). **NOT monotone in every seed (the step directions are not all the same across steps and seeds; a tie counts as no direction): no monotone claim is made.**

| step | matched seeds | up / down / tie | mean difference | exact Mann-Whitney p (two-sided) | min sign-test p |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 0.5b -> 1.5b | 3 | 1 / 2 / 0 | -0.0012 | 1.0000 (Holm 1.0000; min attainable 0.100; 20 permutations) | 0.250 |

## Method

- Strict macro-F1 recomputed from `NAME.preds.jsonl` (`common.macro_f1`), not copied from the JSON; a >6e-4 discrepancy is flagged.
- A cell enters the analysis only if it is a full-test cell (10,000 rows, ids 0..9999 in order, gold identical to the arm's test gold).
- Point difference = LLM F1 - baseline F1 (canonical `<arm>_baseline_preds.json` DT/LinearSVC; BERT = validation-selected REFIT). Paired bootstrap = `paired.paired_bootstrap` (rows resampled jointly, percentile 95% CI, label set fixed from the full gold).
- Classification: `LLM` = whole CI above 0; `base` = whole CI below 0; `ns` = CI spans 0 (a bound exactly at 0 counts as spanning). Wins-matrix cell = `#seeds LLM / #seeds baseline / #seeds neither`, then the verdict word.
- Ordering: sizes ranked per seed by F1; argmax over the sizes that seed has; a peak is claimed only when all four sizes are present, every seed is complete, there are >= 2 seeds, all agree on one size and that size is interior (an argmax at 0.5b or 7b is an edge of the ladder, never a peak). The claim is descriptive: the reason prints the chance level (1/k)^(n-1) of n seeds agreeing on one of k exchangeable sizes and the margin over the runner-up size against the across-seed SD; no between-size significance test is run. Step directions are counted per matched seed (up = the larger model scores higher by more than 1e-12).
- Verdict words in the wins matrix are shorthand for the seed counts beside them; the unadjusted table and the Holm table can give different words, and the Holm one is the multiplicity-corrected reading.
- Pre-specified: every comparable cell against each of DT / LinearSVC / BERT-REFIT, and every adjacent-size step. Post-hoc (selected by looking at the test scores, descriptive only): 'best single cell' and 'highest seed-mean size'.
- Holm family for the cell x baseline tests = all of them in the report; Holm p uses z = difference / bootstrap SD (two-sided normal), because the add-one bootstrap p (smallest attainable 2 / (B + 1)) is resolution-limited after Holm; both are stored in the JSON.
- Exact Mann-Whitney: permutation on midranks over all C(n1+n2, n1) assignments, two-sided; run only when both sizes have >= 3 seeds.
