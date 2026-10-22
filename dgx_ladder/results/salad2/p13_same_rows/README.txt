Same-rows classical baselines for the MLX ladder (Section VI-B of the paper).
  rows_{unsw,cicids}.json : row indices (into the first 4,500 rows of salad2_train.json) visited by 600 steps x batch 4,
                            per seed (42/77/123); a 100-step chunk is the first 400 entries, a 300-step run the first 1,200.
  same_rows_out.json      : Decision Tree / LinearSVC macro-F1 (full 10,000-row test) trained on those rows.
  rows_seen.py, tok_check.py, same_rows.py : how the indices were produced, fingerprint-checked and scored (re-running needs the built corpus).
The canonical baselines reproduce exactly with scikit-learn 1.8.0: UNSW 0.5874 / 0.6505, CICIDS 0.9310 / 0.9636.
