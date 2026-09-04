#!/usr/bin/env python3
"""
Independent validation of check_leakage() (build_salad2.py) — NOT calibration.

WHY THIS EXISTS
---------------
check_leakage()'s thresholds (dominance=0.90, abs_floor=0.85, purity>=0.999,
cardinality<=max(2*n_classes,16)) were tuned reactively against exactly two cases:
SALAD v1 (must FAIL) and SALAD v2 (must PASS). A gate whose only calibration set is
"the two cases it was built to separate" is circular — it proves the gate agrees with
itself, not that it would catch a leak it wasn't designed around. A skeptical reviewer
will ask: does this generalize, or did you just build a detector for the one bug you
already knew about?

This script answers that with two INDEPENDENT tests the gate was never tuned against:

  1. SYNTHETIC DOSE-RESPONSE — inject columns of precisely known leak strength (from
     0% to 100% deterministic) into fully synthetic data with a known ground-truth
     label distribution. Sweep both leak MECHANISMS the gate's two criteria are meant
     to catch (a noisy-but-informative "statistical answer key" for criterion A; a
     low-cardinality deterministic rename for criterion B), plus two adversarial
     NEGATIVE controls the gate must NOT flag (an honestly-weak feature at real-data
     strength; a high-cardinality near-unique column with zero predictive power, the
     exact "Flow Duration" false-positive shape called out in check_leakage()'s own
     docstring). This traces the detection boundary as a curve, not a spot check.

  2. LITERATURE-DOCUMENTED LEAK — Engelen, Rimmer & Joosen, "Troubleshooting an
     Intrusion Detection Dataset: the CICIDS2017 Case Study," IEEE S&P Workshops
     (DLS) 2021 (verified by direct PDF read, not a search-engine summary — an
     earlier draft of this test cited the SAME paper for a "Destination Port"
     leak claim the paper does not actually make; see NOTE below). What the paper
     ACTUALLY documents (Section IV-A, "TCP appendix impact"): their Random Forest
     classifier does "shortcut learning" using four features that "do not have any
     semantic connection with the intended malicious activity" but instead encode
     an artifact of CICFlowMeter's flawed TCP-flow-termination logic (early-FIN
     "appendix" flows) — quote: "it uses four prominent but largely irrelevant
     features to crisply differentiate between appendices of different classes
     (Fwd Header Len, Total Fwd Pkt, Min Seg Size Fwd, and Init Win bytes Fwd)".
     This is a real, independently-published, non-circular finding: it was derived
     from manual PCAP-level forensics of the CICFlowMeter tool by a team with no
     connection to this repo, years before SALAD2 existed. build_salad2.py's own
     CICIDS drop-list does not exclude any of these four. This test checks whether
     check_leakage() -- designed around a completely different bug (SALAD v1's
     label-rename) -- independently flags the SAME feature(s) Engelen et al. named,
     on the actual feature set build_salad2.py currently ships.

     NOTE ON CITATION HYGIENE: the first draft of this script asserted "Destination
     Port alone reaches near-ceiling accuracy" and attributed it to this paper, based
     on WebSearch result summaries. Directly reading the PDF (see git history / this
     session's transcript) showed the paper does NOT make that quantitative claim --
     it only says "using the destination port as a NIDS feature is up for debate,"
     grouped with other fields it recommends excluding, with no solo-accuracy number
     given. That claim was corrected before being written into any paper or result
     file: verify against the primary source rather than a search-result synthesis.

Usage:
    python3 scripts/validate_leak_gate.py --synthetic       # no downloads, seconds
    python3 scripts/validate_leak_gate.py --cicids-engelen  # needs cached bvk/CICIDS-2017
    python3 scripts/validate_leak_gate.py --all
"""
import argparse, json, random, sys, os

sys.path.insert(0, os.path.dirname(__file__))
from build_salad2 import check_leakage, LABEL


# ---------------------------------------------------------------------------------
# 1. SYNTHETIC DOSE-RESPONSE
# ---------------------------------------------------------------------------------

def make_base_rows(n=6000, n_classes=8, seed=0):
    """Ground-truth labels + two honest, real-strength-ish features (never leaks)."""
    rs = random.Random(seed)
    classes = [f"Class{i}" for i in range(n_classes)]
    rows = []
    for _ in range(n):
        y = rs.choice(classes)
        # "honest_weak": correlated with label but nowhere near sufficient alone —
        # modeled on v2's real Source Bytes (solo 0.56 vs full 0.59): each class has
        # a preferred numeric band, but bands overlap heavily across classes.
        band = classes.index(y)
        honest_weak = round(rs.gauss(band * 10, 15), 2)
        rows.append({LABEL: y, "honest_weak": honest_weak})
    return rows, classes


def inject_statistical_leak(rows, classes, purity, seed=1):
    """Criterion-A test: a HIGH-CARDINALITY column that encodes the label with
    exactly `purity` fraction certainty and is otherwise random noise. High
    cardinality (~n_classes * 50 distinct values) keeps criterion B's cardinality
    bound from ever firing, isolating criterion A's statistical test."""
    rs = random.Random(seed)
    out = []
    for r in rows:
        r = dict(r)
        if rs.random() < purity:
            # encodes the true label, but through a wide value range (high card)
            r["stat_leak"] = f"v{classes.index(r[LABEL]) * 137 + rs.randint(0, 40)}"
        else:
            r["stat_leak"] = f"v{rs.randint(0, 9999)}"
        out.append(r)
    return out


def inject_rename_leak(rows, classes, purity, seed=2):
    """Criterion-B test: a LOW-CARDINALITY column (one value per class, like SALAD
    v1's 'Alert Type') that is deterministic at exactly `purity` fraction of rows."""
    rs = random.Random(seed)
    rename = {c: f"code_{i}" for i, c in enumerate(classes)}
    out = []
    for r in rows:
        r = dict(r)
        if rs.random() < purity:
            r["rename_leak"] = rename[r[LABEL]]
        else:
            r["rename_leak"] = rs.choice(list(rename.values()))
        out.append(r)
    return out


def make_imbalanced_rows(n=130, n_classes=8, seed=0, dominant_share=0.80):
    """Like make_base_rows but with SALAD v1's actual skew (one dominant class ~80%,
    the rest sharing the remainder) — tests whether criterion A's small-sample
    unreliability (claimed in check_leakage()'s docstring) is actually an
    imbalance effect, not a raw-N effect (the balanced n=130 test above did NOT
    reproduce it)."""
    rs = random.Random(seed)
    classes = [f"Class{i}" for i in range(n_classes)]
    weights = [dominant_share] + [(1 - dominant_share) / (n_classes - 1)] * (n_classes - 1)
    rows = []
    for _ in range(n):
        y = rs.choices(classes, weights=weights, k=1)[0]
        band = classes.index(y)
        honest_weak = round(rs.gauss(band * 10, 15), 2)
        rows.append({LABEL: y, "honest_weak": honest_weak})
    return rows, classes


def inject_false_positive_trap(rows, seed=3):
    """Negative control: near-unique continuous column with ZERO relationship to
    the label (a row index + noise) — mimics CICIDS 'Flow Duration' (card 38079,
    purity 0.965, solo_f1 0.24, correctly NOT flagged). Must stay unflagged."""
    rs = random.Random(seed)
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        r["fp_trap_unique_noise"] = round(i * 1.0001 + rs.random(), 4)
        out.append(r)
    return out


def run_synthetic():
    print("=" * 88)
    print("SYNTHETIC DOSE-RESPONSE — sweeping known leak strength through both gate criteria")
    print("=" * 88)
    base, classes = make_base_rows()

    results = []

    # --- criterion A sweep: statistical leak at increasing purity ---
    print("\n--- Criterion A (statistical answer key), high-cardinality noisy leak ---")
    print(f"{'purity_injected':>16} {'solo_f1':>8} {'full_f1':>8} {'flagged':>8} {'criterion':>20}")
    for purity in [0.0, 0.5, 0.7, 0.8, 0.85, 0.90, 0.95, 0.99, 1.0]:
        rows = inject_statistical_leak(base, classes, purity)
        feats = ["honest_weak", "stat_leak"]
        findings, full_f1 = check_leakage(rows, feats)
        f = next(x for x in findings if x["feature"] == "stat_leak")
        flagged = f["leak_like"]
        print(f"{purity:16.2f} {f['solo_f1']:8.3f} {full_f1:8.3f} {str(flagged):>8} {str(f['criterion']):>20}")
        results.append({"test": "criterion_A_sweep", "purity_injected": purity,
                         "solo_f1": f["solo_f1"], "full_f1": full_f1,
                         "flagged": flagged, "criterion": f["criterion"]})

    # --- criterion B sweep: low-cardinality rename leak at increasing purity ---
    print("\n--- Criterion B (deterministic rename), low-cardinality leak ---")
    print(f"{'purity_injected':>16} {'purity_measured':>16} {'card':>5} {'flagged':>8} {'criterion':>20}")
    for purity in [0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0]:
        rows = inject_rename_leak(base, classes, purity)
        feats = ["honest_weak", "rename_leak"]
        findings, full_f1 = check_leakage(rows, feats)
        f = next(x for x in findings if x["feature"] == "rename_leak")
        flagged = f["leak_like"]
        print(f"{purity:16.4f} {f['purity']:16.4f} {f['cardinality']:5d} {str(flagged):>8} {str(f['criterion']):>20}")
        results.append({"test": "criterion_B_sweep", "purity_injected": purity,
                         "purity_measured": f["purity"], "cardinality": f["cardinality"],
                         "flagged": flagged, "criterion": f["criterion"]})

    # --- small-sample regime: why criterion B exists separately from A ---
    # The bulk sweep above (n=6000) shows criterion A alone catches the rename leak
    # at purity>=0.90, which raises an honest question: does criterion B add
    # anything? Its docstring justification is specifically the SMALL-UNIQUE-SAMPLE
    # case (v1 dedups 14,851 rows to 127 uniques) where a 70/30 tree-based estimate
    # is small-sample noise. Test that regime directly: same rename-leak mechanism,
    # but only ~130 unique rows (v1-scale), repeated over many random splits to see
    # whether criterion A's estimate is actually unstable there while B stays exact.
    print("\n--- Small-sample regime (n=130, v1-scale): does A get noisy where B stays exact? ---")
    small_base, small_classes = make_base_rows(n=130, n_classes=8, seed=10)
    print(f"{'purity_injected':>16} {'A_flag_rate/10':>15} {'B_flag_rate/10':>15}")
    for purity in [0.90, 0.95, 0.99, 1.0]:
        a_flags, b_flags = 0, 0
        for trial in range(10):
            rows = inject_rename_leak(small_base, small_classes, purity, seed=100 + trial)
            findings, full_f1 = check_leakage(rows, ["honest_weak", "rename_leak"], seed=trial)
            f = next(x for x in findings if x["feature"] == "rename_leak")
            crit = f["criterion"] or ""
            a_flags += "A:" in crit
            b_flags += "B:" in crit
        print(f"{purity:16.2f} {a_flags:>14}/10 {b_flags:>14}/10")
        results.append({"test": "small_sample_regime", "purity_injected": purity,
                         "criterion_A_flag_rate": a_flags / 10, "criterion_B_flag_rate": b_flags / 10})

    # --- small-sample + severe imbalance regime: v1's actual skew, not just its N ---
    print("\n--- Small-sample + imbalanced (n=130, 80% dominant class, v1-shaped): ---")
    print(f"{'purity_injected':>16} {'A_flag_rate/10':>15} {'B_flag_rate/10':>15}")
    for purity in [0.90, 0.95, 0.99, 1.0]:
        a_flags, b_flags = 0, 0
        for trial in range(10):
            imb_base, imb_classes = make_imbalanced_rows(n=130, n_classes=8, seed=20 + trial)
            rows = inject_rename_leak(imb_base, imb_classes, purity, seed=200 + trial)
            findings, full_f1 = check_leakage(rows, ["honest_weak", "rename_leak"], seed=trial)
            f = next(x for x in findings if x["feature"] == "rename_leak")
            crit = f["criterion"] or ""
            a_flags += "A:" in crit
            b_flags += "B:" in crit
        print(f"{purity:16.2f} {a_flags:>14}/10 {b_flags:>14}/10")
        results.append({"test": "small_sample_imbalanced_regime", "purity_injected": purity,
                         "criterion_A_flag_rate": a_flags / 10, "criterion_B_flag_rate": b_flags / 10})

    # --- negative controls: must NOT be flagged ---
    print("\n--- Negative controls (must NOT be flagged) ---")
    rows_fp = inject_false_positive_trap(base)
    feats = ["honest_weak", "fp_trap_unique_noise"]
    findings, full_f1 = check_leakage(rows_fp, feats)
    fp = next(x for x in findings if x["feature"] == "fp_trap_unique_noise")
    hw = next(x for x in findings if x["feature"] == "honest_weak")
    print(f"honest_weak (real-strength, non-leak feature): purity={hw['purity']:.3f} "
          f"solo_f1={hw['solo_f1']:.3f} flagged={hw['leak_like']}")
    print(f"fp_trap_unique_noise (near-unique, zero signal): card={fp['cardinality']} "
          f"purity={fp['purity']:.3f} solo_f1={fp['solo_f1']:.3f} flagged={fp['leak_like']}")
    results.append({"test": "negative_control", "feature": "honest_weak",
                     "purity": hw["purity"], "solo_f1": hw["solo_f1"], "flagged": hw["leak_like"]})
    results.append({"test": "negative_control", "feature": "fp_trap_unique_noise",
                     "cardinality": fp["cardinality"], "purity": fp["purity"],
                     "solo_f1": fp["solo_f1"], "flagged": fp["leak_like"]})

    # --- verdict ---
    print("\n" + "=" * 88)
    a_sweep = [r for r in results if r["test"] == "criterion_A_sweep"]
    b_sweep = [r for r in results if r["test"] == "criterion_B_sweep"]
    a_boundary = next((r["purity_injected"] for r in a_sweep if r["flagged"]), None)
    b_boundary = next((r["purity_injected"] for r in b_sweep if r["flagged"]), None)
    fp_ok = not fp["leak_like"]
    hw_ok = not hw["leak_like"]
    print(f"Criterion A detection boundary: first flagged at injected purity >= {a_boundary}")
    print(f"Criterion B detection boundary: first flagged at injected purity >= {b_boundary}")
    print(f"False-positive traps clean: honest_weak={hw_ok}, fp_trap_unique_noise={fp_ok}")

    imb_100 = next(r for r in results if r["test"] == "small_sample_imbalanced_regime" and r["purity_injected"] == 1.0)
    imb_90 = next(r for r in results if r["test"] == "small_sample_imbalanced_regime" and r["purity_injected"] == 0.90)
    b_justified = imb_100["criterion_B_flag_rate"] > imb_100["criterion_A_flag_rate"]
    print(f"\nCriterion B justification (small+imbalanced, v1-shaped, purity=1.0 EXACT leak):")
    print(f"  criterion A alone catches it {imb_100['criterion_A_flag_rate']*10:.0f}/10 trials")
    print(f"  criterion B catches it        {imb_100['criterion_B_flag_rate']*10:.0f}/10 trials")
    print(f"  -> B is {'JUSTIFIED as a backstop' if b_justified else 'NOT adding value over A here'}"
          f" in this regime (matches/refutes the docstring's stated rationale)")
    print(f"\nRESIDUAL GAP (honest limitation, not hidden): at purity=0.90 in the same small+")
    print(f"imbalanced regime, NEITHER criterion catches it reliably (A={imb_90['criterion_A_flag_rate']*10:.0f}/10, "
          f"B={imb_90['criterion_B_flag_rate']*10:.0f}/10) — a 90%-but-not-fully-deterministic leak in a")
    print(f"small, imbalanced sample can evade both criteria. Document this as a known limitation.")

    verdict = "PASS" if (a_boundary is not None and b_boundary is not None and fp_ok and hw_ok) else "FAIL"
    print(f"\nVERDICT: {verdict}")
    print("=" * 88)

    out = {"verdict": verdict, "criterion_A_boundary": a_boundary,
           "criterion_B_boundary": b_boundary, "false_positives_clean": fp_ok and hw_ok,
           "criterion_B_justified_smallN_imbalanced_exact_leak": b_justified,
           "residual_gap_smallN_imbalanced_partial_leak": {
               "purity_0.90_criterion_A_flag_rate": imb_90["criterion_A_flag_rate"],
               "purity_0.90_criterion_B_flag_rate": imb_90["criterion_B_flag_rate"],
               "note": "a ~90%-pure (non-exact) leak in a small (~130 unique row), imbalanced "
                       "(80% dominant class) sample can evade both criteria — known limitation."},
           "results": results}
    os.makedirs("results", exist_ok=True)
    with open("results/leak_gate_synthetic_validation.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results/leak_gate_synthetic_validation.json")
    return verdict == "PASS"


# ---------------------------------------------------------------------------------
# 2. LITERATURE-DOCUMENTED LEAK: CICIDS-2017 "TCP appendix" shortcut features
#    (Engelen, Rimmer & Joosen 2021, IEEE S&P Workshops — independent of this repo,
#     claim verified by direct PDF read: quote reproduced in the module docstring)
# ---------------------------------------------------------------------------------

# Engelen et al.'s own names -> the equivalent columns in bvk/CICIDS-2017's schema
# (CICFlowMeter output naming varies slightly by mirror/version; matched by content).
ENGELEN_SHORTCUT_FEATURES = {
    "Fwd Header Len": "Fwd Header Length",
    "Total Fwd Pkt": "Total Fwd Packet",
    "Min Seg Size Fwd": "Fwd Seg Size Min",
    "Init Win bytes Fwd": "FWD Init Win Bytes",
}


def run_cicids_engelen():
    print("=" * 88)
    print("LITERATURE-DOCUMENTED LEAK — CICIDS-2017 'TCP appendix' shortcut features")
    print("(Engelen, Rimmer & Joosen, IEEE S&P Workshops (DLS) 2021 — independent finding,")
    print(" verified by direct PDF read: \"it uses four prominent but largely irrelevant")
    print(" features to crisply differentiate between appendices of different classes")
    print(" (Fwd Header Len, Total Fwd Pkt, Min Seg Size Fwd, and Init Win bytes Fwd)\")")
    print("=" * 88)
    from datasets import load_dataset
    from build_salad2 import SOURCES, canon_cicids

    src = SOURCES["cicids"]
    d = load_dataset(src["hf"], revision=src.get("revision"))
    cols = set(d["train"].column_names)

    matched = {k: v for k, v in ENGELEN_SHORTCUT_FEATURES.items() if v in cols}
    missing = {k: v for k, v in ENGELEN_SHORTCUT_FEATURES.items() if v not in cols}
    print(f"\nMatched {len(matched)}/4 named features in this HF mirror's schema:")
    for k, v in matched.items():
        print(f"  Engelen '{k}' -> this mirror's '{v}'")
    if missing:
        print(f"Not found (schema mismatch, skipped): {missing}")

    feats_current = [c for c in d["train"].column_names if c not in src["drop"]]
    # Match build_salad2.py's actual certified population EXACTLY: shuffle with
    # Python's own random.Random(seed) (NOT datasets' .shuffle(), which was found
    # 2026-07-19 to be non-reproducible across library versions/environments --
    # see the note in build_salad2.py's main()), accumulate up to 400,000
    # valid-labeled rows, take the FIRST HALF (200,000) as `tr`, the population
    # check_leakage() is actually certified against.
    n_total = len(d["train"])
    idx = list(range(n_total))
    random.Random(42).shuffle(idx)
    allr = []
    for i in idx:
        r = d["train"][i]
        y = canon_cicids(r[src["label"]])
        if y is None:
            continue
        rec = {c: r[c] for c in feats_current}
        rec[LABEL] = y
        allr.append(rec)
        if len(allr) >= 400000:
            break
    rows = allr[: int(len(allr) * 0.5)]
    print(f"\nLoaded {len(allr)} labeled rows, using first {len(rows)} (build_salad2.py's "
          f"own train-split population) x {len(feats_current)} candidate features")

    findings, full_f1 = check_leakage(rows, feats_current)
    by_name = {f["feature"]: f for f in findings}
    print(f"\n{'engelen name':<20} {'this mirror col':<24} {'purity':>7} {'solo_f1':>8} {'flagged':>8}")
    per_feature = {}
    for k, v in matched.items():
        f = by_name.get(v)
        if f is None:
            continue
        print(f"{k:<20} {v:<24} {f['purity']:7.3f} {f['solo_f1']:8.3f} {str(f['leak_like']):>8}")
        per_feature[v] = f
    print(f"(full-feature DT macro-F1 = {full_f1})")

    # NOTE ON VERDICT FRAMING: Engelen et al.'s claim is "these 4 features let an RF
    # discriminate TCP-appendix artifacts from real flows" -- a DIFFERENT, weaker
    # claim than check_leakage()'s own bar ("one feature alone reaches >=85% macro-F1
    # on the actual 8-family task"). The gate is not obligated to flag a merely-
    # informative feature as a full answer key, so a binary pass/fail on whether it
    # flags them is a category error. The fair, non-circular question is whether the
    # gate's OWN solo_f1/purity ranking is directionally consistent with Engelen's
    # finding -- i.e. do their 4 named features rank above typical/unnamed features?
    all_solo_f1 = sorted((f["solo_f1"] for f in findings), reverse=True)
    n_feats = len(all_solo_f1)
    ranks = {}
    for k, v in matched.items():
        f = by_name.get(v)
        if f is None:
            continue
        rank = sorted(all_solo_f1, reverse=True).index(f["solo_f1"]) + 1
        pctile = 100 * (1 - rank / n_feats)
        ranks[v] = {"rank": rank, "of": n_feats, "percentile": round(pctile, 1)}
        print(f"  {v}: rank {rank}/{n_feats} by solo_f1 ({pctile:.0f}th percentile)")
    any_top_quartile = any(r["percentile"] >= 75 for r in ranks.values())
    print(f"\nAt least one Engelen-named feature in the top quartile by solo_f1: {any_top_quartile}")
    verdict = ("directionally consistent (Engelen-named features rank high, even though "
               "none crosses this gate's stricter single-feature answer-key bar)"
               if any_top_quartile else
               "NOT corroborated (Engelen-named features do not even rank unusually high)")
    print(f"VERDICT: {verdict}")
    print("=" * 88)

    out = {"reference": "Engelen, Rimmer & Joosen 2021, IEEE S&P Workshops (DLS), "
                         "\"Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study\"",
           "quote_verified_by_direct_pdf_read": (
               "it uses four prominent but largely irrelevant features to crisply "
               "differentiate between appendices of different classes (Fwd Header Len, "
               "Total Fwd Pkt, Min Seg Size Fwd, and Init Win bytes Fwd)"),
           "matched_features": matched, "n_rows": len(rows),
           "full_feature_dt_macro_f1": full_f1, "per_feature": per_feature,
           "ranks_by_solo_f1": ranks, "verdict": verdict,
           "all_findings_top10": findings[:10]}
    os.makedirs("results", exist_ok=True)
    with open("results/leak_gate_cicids_engelen_validation.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results/leak_gate_cicids_engelen_validation.json")
    return any_top_quartile


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cicids-engelen", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all or a.synthetic:
        run_synthetic()
    if a.all or a.cicids_engelen:
        run_cicids_engelen()
    if not (a.all or a.synthetic or a.cicids_engelen):
        ap.print_help()
