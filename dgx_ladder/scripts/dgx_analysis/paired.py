"""paired -- paired bootstrap of the macro-F1 difference between two systems scored on the SAME rows.

Python API
    paired_bootstrap(gold, pred_a, pred_b, labels=None, B=2000, seed=0, alpha=0.05) -> dict
CLI
    python3 -m dgx_analysis paired --a <cell|dt|svc|bert|majority> --b <cell|dt|svc|bert|majority> --arm <unsw|cicids>

Method: rows are resampled with replacement JOINTLY for both systems (one index vector per resample, B resamples,
np.random.default_rng(seed)), the label set is FIXED from the full gold (macro-F1 averages over the same K labels in
every resample; a label absent from a resample scores 0 for both systems, as in the runner's zero_division=0 rule),
and the statistic is macro-F1(a) - macro-F1(b). Reported: the point difference, the percentile CI, the fraction of
resamples with difference > 0 (strictly), < 0, == 0, and an add-one bootstrap p = min(1, 2*(min(#{d<=0}, #{d>=0})+1)/(B+1))
whose smallest attainable value is 2/(B+1). This quantifies TEST-ROW sampling uncertainty only; it says nothing about
training-seed variance (one trained model per system).
"""
from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np

from .common import (ARMS, AnalysisDataError, encode, fmt, gold_labels, holm_adjust, load_cells, macro_f1_codes,
                     md_table, per_class_f1, save_report, system_preds)

ALIASES = ("dt", "svc", "majority", "bert")


def paired_bootstrap(gold: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str],
                     labels: Sequence[str] | None = None, B: int = 2000, seed: int = 0, alpha: float = 0.05,
                     return_samples: bool = False) -> dict:
    """Paired bootstrap over rows of macro-F1(a) - macro-F1(b). See the module docstring for the exact method.

    Returns dict: diff (point estimate a-b), f1_a, f1_b, ci (low, high), ci_low, ci_high, ci_level, frac_gt0,
    frac_lt0, frac_eq0, p_boot, p_boot_min, boot_sd, B, seed, n, K, labels, n_resamples_missing_label
    (+ samples when return_samples)."""
    n = len(gold)
    if not (n == len(pred_a) == len(pred_b)):
        raise ValueError(f"length mismatch: gold={n} a={len(pred_a)} b={len(pred_b)}")
    if n == 0:
        raise ValueError("empty inputs")
    labels = gold_labels(gold) if labels is None else list(labels)
    K = len(labels)
    gi, pa = encode(gold, pred_a, labels)
    _, pb = encode(gold, pred_b, labels)
    f1_a = macro_f1_codes(gi, pa, K)
    f1_b = macro_f1_codes(gi, pb, K)
    rng = np.random.default_rng(seed)
    diffs = np.empty(B, dtype=np.float64)
    missing = 0
    for b in range(B):
        idx = rng.integers(0, n, n)
        g = gi[idx]
        diffs[b] = macro_f1_codes(g, pa[idx], K) - macro_f1_codes(g, pb[idx], K)
        if np.count_nonzero(np.bincount(g[g >= 0], minlength=K) == 0):
            missing += 1
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    k_le = int(np.count_nonzero(diffs <= 0))
    k_ge = int(np.count_nonzero(diffs >= 0))
    out = {
        "diff": float(f1_a - f1_b), "f1_a": float(f1_a), "f1_b": float(f1_b),
        "ci": (float(lo), float(hi)), "ci_low": float(lo), "ci_high": float(hi), "ci_level": 1 - alpha,
        "frac_gt0": float(np.count_nonzero(diffs > 0) / B), "frac_lt0": float(np.count_nonzero(diffs < 0) / B),
        "frac_eq0": float(np.count_nonzero(diffs == 0) / B),
        "p_boot": float(min(1.0, 2.0 * (min(k_le, k_ge) + 1) / (B + 1))), "p_boot_min": 2.0 / (B + 1),
        "boot_sd": float(diffs.std(ddof=1)) if B > 1 else float("nan"),
        "B": int(B), "seed": int(seed), "n": int(n), "K": int(K), "labels": list(labels),
        "n_resamples_missing_label": int(missing),
    }
    if return_samples:
        out["samples"] = diffs
    return out


def _resolve_arm(spec_a: str, spec_b: str, arm: str | None) -> str:
    cells = {c.name: c for c in load_cells()}
    arms = set()
    for s in (spec_a, spec_b):
        if s.lower() in ALIASES or s.lower() in ("decision_tree", "linear_svc", "svm"):
            continue
        if s not in cells:
            raise AnalysisDataError(f"unknown system {s!r}: not dt|svc|majority|bert and no such cell")
        arms.add(cells[s].arm)
    if arm is None:
        if len(arms) == 1:
            return next(iter(arms))
        raise AnalysisDataError("--arm is required (both systems are baselines, or the cells' arms differ)")
    if arms and arms != {arm}:
        raise AnalysisDataError(f"--arm {arm} conflicts with the cells' arm(s) {sorted(arms)}")
    return arm


def compare(arm: str, spec_a: str, spec_b: str, B: int = 2000, seed: int = 0) -> dict:
    """Load both systems on the arm's shared test rows and run the paired bootstrap (+ per-class F1 point diffs)."""
    gold, pa = system_preds(arm, spec_a)
    gold_b, pb = system_preds(arm, spec_b)
    if gold != gold_b:
        raise AnalysisDataError("the two systems' gold sequences differ; refusing a paired comparison")
    labels = gold_labels(gold)
    res = paired_bootstrap(gold, pa, pb, labels, B=B, seed=seed)
    pca, pcb = per_class_f1(gold, pa, labels), per_class_f1(gold, pb, labels)
    res.update({"arm": arm, "a": spec_a, "b": spec_b,
                "per_class": {lab: {"a": pca[lab], "b": pcb[lab], "diff": pca[lab] - pcb[lab]} for lab in labels}})
    return res


def _verdict(r: dict, holm: float | None = None) -> str:
    """One-line reading. With `holm` (the Holm-adjusted p of a family of tests) the reading also states whether the
    difference survives the multiplicity correction, so an unadjusted CI that excludes 0 is never read as a finding
    by itself."""
    a, b = r.get("a", "A"), r.get("b", "B")
    if r["ci_low"] > 0:
        who = f"{a} > {b}"
    elif r["ci_high"] < 0:
        who = f"{b} > {a}"
    else:
        return "no resolved difference (CI spans 0)"
    if holm is None:
        return f"{who} (CI excludes 0)"
    if holm < 0.05:
        return f"{who} (CI excludes 0; Holm p {holm:.4f} < 0.05)"
    return f"{who}: CI excludes 0 but NOT significant after Holm (p {holm:.4f})"


def render_one(r: dict) -> str:
    lines = [f"# Paired bootstrap: {r['a']} vs {r['b']} ({r['arm']})", "",
             f"System A = `{r['a']}`, system B = `{r['b']}`; statistic = macro-F1(A) - macro-F1(B) on the arm's "
             f"{r['n']:,} shared test rows, K = {r['K']} gold labels. Rows resampled jointly, B = {r['B']:,} resamples, "
             f"seed {r['seed']}, label set fixed from the full gold. Test-row uncertainty only (one trained model per system; "
             f"no training-seed variance). This is ONE comparison with no multiplicity correction: if it is one of several "
             f"tests, use the Holm-adjusted `paired_default` report instead.", ""]
    lines.append(md_table(["quantity", "value"], [
        ["macro-F1 A", r["f1_a"]], ["macro-F1 B", r["f1_b"]], ["difference A-B", fmt(r["diff"], 4, True)],
        [f"{int(round(100 * r['ci_level']))}% percentile CI", f"[{fmt(r['ci_low'], 4, True)}, {fmt(r['ci_high'], 4, True)}]"],
        ["bootstrap SD of the difference", r["boot_sd"]],
        ["fraction of resamples with diff > 0", r["frac_gt0"]], ["fraction with diff < 0", r["frac_lt0"]],
        ["fraction with diff == 0", r["frac_eq0"]],
        [f"add-one bootstrap p (two-sided, B={r['B']}; smallest attainable {r['p_boot_min']:.1e})", r["p_boot"]],
        ["resamples missing >= 1 label", f"{r['n_resamples_missing_label']} of {r['B']}"],
        ["reading", _verdict(r)]], "lr"))
    lines += ["", "## Per-class F1 (point estimates on the full test set)", ""]
    lines.append(md_table(["class", "A", "B", "A-B"],
                          [[k, v["a"], v["b"], fmt(v["diff"], 4, True)] for k, v in r["per_class"].items()], "lrrr"))
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    try:
        arm = _resolve_arm(args.a, args.b, args.arm)
        r = compare(arm, args.a, args.b, B=args.B, seed=args.seed)
    except AnalysisDataError as e:
        print(f"paired: ERROR: {e}")
        return 2
    print(f"paired [{arm}] A={args.a} F1={r['f1_a']:.4f}  B={args.b} F1={r['f1_b']:.4f}  diff={r['diff']:+.4f}  "
          f"{int(round(100 * r['ci_level']))}% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
          f"frac>0={r['frac_gt0']:.3f}  B={r['B']} seed={r['seed']}  -> {_verdict(r)}")
    if not args.no_save:
        name = args.out or f"paired_{args.a}_vs_{args.b}_{arm}"
        md_p, _ = save_report(name, render_one(r), {k: v for k, v in r.items() if k != "samples"})
        print(f"report: {md_p}")
    return 0


# --------------------------------------------------------------------------------------------------------------
# default report: pre-specified comparisons only (no comparison chosen by looking at test scores)
# --------------------------------------------------------------------------------------------------------------
PRESPECIFIED_LLM = {"unsw": ["F_S3750_L2e-4_C2_Yall"], "cicids": []}   # the P22/P14 default recipe on its own arm


def default_report(B: int = 2000, seed: int = 0) -> tuple[str, dict]:
    rows, results = [], []
    for arm in ARMS:
        pairs = [("dt", "svc"), ("dt", "bert"), ("svc", "bert")]
        for cn in PRESPECIFIED_LLM.get(arm, []):   # always listed: a cell that is not synced yet shows up as an ERROR row
            pairs += [(cn, "dt"), (cn, "svc"), (cn, "bert")]
        for a, b in pairs:
            try:
                r = compare(arm, a, b, B=B, seed=seed)
            except AnalysisDataError as e:
                results.append({"arm": arm, "a": a, "b": b, "error": str(e)})
                continue
            results.append(r)
    ok = [r for r in results if "error" not in r]
    # the family is the pre-specified list: a comparison that could not be computed stays in it as a NaN p-value
    # (holm_adjust counts it toward m), so a missing cell never shrinks the correction for the others
    adj = holm_adjust([r["p_boot"] if "error" not in r else float("nan") for r in results]) if results else []
    for r, pa in zip(results, adj):
        if "error" not in r:
            r["p_boot_holm"] = pa
    lines = ["# Paired bootstrap: pre-specified comparisons", "",
             f"Each row: macro-F1(A) - macro-F1(B) on the arm's 10,000 shared test rows; rows resampled jointly, B = {B:,}, "
             f"seed {seed}, label set fixed from the full gold. Comparisons are pre-specified (every baseline pair, plus the "
             f"P22/P14 default LoRA recipe against each baseline); no comparison is picked by looking at test scores. "
             f"{len(results)} pre-specified tests ({len(ok)} computed): Holm-adjusted p is across all of them (add-one bootstrap p, smallest attainable "
             f"{2 / (B + 1):.1e}). CIs are 95% percentile intervals, not multiplicity-adjusted. Test-row uncertainty only; "
             f"one trained model per LLM cell (seed 42), so no training-seed variance is included.", ""]
    for r in results:
        if "error" in r:
            rows.append([r["arm"], r["a"], r["b"], "-", "-", "-", "-", "-", "-", f"ERROR: {r['error']}"])
        else:
            rows.append([r["arm"], r["a"], r["b"], r["f1_a"], r["f1_b"], fmt(r["diff"], 4, True),
                         f"[{fmt(r['ci_low'], 4, True)}, {fmt(r['ci_high'], 4, True)}]", r["frac_gt0"],
                         f"{r['p_boot']:.4f} / {r['p_boot_holm']:.4f}", _verdict(r, r["p_boot_holm"])])
    lines.append(md_table(["arm", "A", "B", "F1 A", "F1 B", "A-B", "95% CI", "frac>0", "p / Holm p", "reading"], rows,
                          "lllrrrlrrl"))
    data = {"B": B, "seed": seed, "n_tests": len(ok), "results": [{k: v for k, v in r.items() if k != "samples"} for r in results]}
    return "\n".join(lines), data


def run_default() -> int:
    md, data = default_report()
    md_p, _ = save_report("paired_default", md, data)
    print(f"paired_default: {data['n_tests']} comparisons -> {md_p}")
    return 0


def register(subparsers: Any) -> None:
    p = subparsers.add_parser("paired", help="paired bootstrap of macro-F1(a)-macro-F1(b) on the shared test rows")
    p.add_argument("--a", required=True, help="system A: a cell name or dt|svc|bert|majority")
    p.add_argument("--b", required=True, help="system B: a cell name or dt|svc|bert|majority")
    p.add_argument("--arm", choices=list(ARMS), help="arm (inferred from a cell when given)")
    p.add_argument("--B", type=int, default=2000, help="bootstrap resamples (default 2000)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    p.add_argument("--out", help="report name (default paired_<a>_vs_<b>_<arm>)")
    p.add_argument("--no-save", action="store_true", help="print only, do not write a report")
    p.set_defaults(func=run)
