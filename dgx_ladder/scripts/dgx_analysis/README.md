# dgx_analysis (the subset this artifact ships)

Recomputes the tuned-ladder results of the SALAD-v2 paper (Section VI-B) from per-row predictions.

```
cd scripts && python3 -m dgx_analysis ladder              # -> ../results/salad2/dgx/analysis/ladder.{json,md}
cd scripts && python3 -m dgx_analysis ladder --B 200 --quiet   # fast smoke run
cd scripts && python3 -m dgx_analysis tuning              # the validation tuning grid (needs scipy)
```

`ladder` reads `results/salad2/dgx/attrib/LT_*.json` and `LT_*.preds.jsonl` (the 12 tuned cells), recomputes strict macro-F1 and the
off-vocabulary rate from the per-row predictions, checks them against each cell's own JSON, and runs the paired bootstrap (B = 5,000 by
default, one shared seed so all comparisons use the same resamples) of every cell against the Decision Tree, LinearSVC and BERT-base
baselines read from `results/salad2/*_baseline_preds.json` / `*_bert_preds.json`. Set `DGX_ANALYSIS_OUT` to write the reports elsewhere.
Needs numpy. It does not need the built corpus; the checks that compare predictions with the gold labels of the corpus
(`verify`) are not shipped because they need a reader's own rebuild.
