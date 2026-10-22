"""ladder -- the tuned model-size ladder: LT_<arm>_<size>_lora_r16_s<seed> cells (sizes 0.5b/1.5b/3b/7b, seeds 42/77/123,
arms unsw/cicids; 24 cells by design) scored on the arm's shared 10,000 test rows.

Python API
    build(dirs=None, arm=None, B=2000, seed=0) -> dict      the whole analysis (jsonable apart from nothing: to_jsonable is applied on save)
    render(data) -> str                                      the markdown report
    analyse_ordering(values) -> dict                         per-seed size ranking / argmax / adjacent steps (pure)
    exact_mannwhitney(x, y) -> dict | None                   exact two-sided Mann-Whitney permutation test (pure)
CLI
    python3 -m dgx_analysis ladder [--dirs D ...] [--arm unsw|cicids] [--B 2000] [--seed 0] [--out ladder] [--no-save] [--quiet]

What it reports (per arm x size): n seeds, mean and sample SD (ddof=1) of strict macro-F1, the per-seed values, off-vocab
rate; the point difference to the canonical DT / LinearSVC / BERT-REFIT (same 10,000 test rows) AND a paired bootstrap of
each LLM cell against each baseline (paired.paired_bootstrap, rows resampled jointly, label set fixed from the full gold),
summarised as "# seeds whose 95% CI excludes 0 in favour of the LLM / of the baseline / neither" (unadjusted CIs, the
primary count) plus the same count under Holm-adjusted bootstrap p < 0.05 (Holm across ALL cell x baseline tests in the
report); a wins matrix (size x baseline); the size ladder per seed with the per-seed argmax and how many seeds agree (a
"peak" is only claimed when every seed is complete over all four sizes and they all agree); the per-seed direction of each
adjacent-size step (+ an exact Mann-Whitney test only when both sizes have >= 3 seeds, with its smallest attainable p
printed); the recipe each cell adopted from the tuning grid (config.tuned_resolved), read per cell, never assumed.

A cell that cannot be compared on the shared rows (no predictions, not a full-test cell, rows/ids off, gold not aligned
with the arm's test gold, unknown arm/model/seed, duplicate (arm, size, seed)) is NOT dropped silently and is never
imputed: it is listed under "not comparable" with its own F1 and the reason, and it takes no part in means, orderings or
bootstraps. Nothing here is written outside results/salad2/dgx/analysis/ (save_report) and nothing talks to the DGX.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .common import (ARMS, ATTRIB_DIR, LOAD_ERRORS, P22_ATTRIB_DIR, SALAD2, VERIFY_TOL, AnalysisDataError, Cell, baseline_preds, bert_baseline,
                     canonical_baselines, fmt, holm_adjust, load_cells, macro_f1, mannwhitney_min_p, md_table, mean,
                     off_vocab_rate, parse_cell_name, read_json, save_report, scan_cell_files, sd, sign_test_min_p, split_gold, test_gold)
from .paired import paired_bootstrap

# --------------------------------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------------------------------
FAMILY = "LT"
SIZES = ("0.5b", "1.5b", "3b", "7b")
SEEDS = (42, 77, 123)
BASELINES = ("dt", "svc", "bert")
BASELINE_NAME = {"dt": "DT", "svc": "LinearSVC", "bert": "BERT-REFIT"}
N_TEST = 10000
BATCH = 4                      # attrib.py BATCH
DEFAULT_EPOCHS = 3             # attrib.py EPOCHS (used when a cell has neither steps nor epochs)
TIE_TOL = 1e-12
ALPHA = 0.05
MIN_SEEDS_EXACT = 3            # an exact seed-level test is only run when both groups have >= this many seeds
# recipe fields that must be identical across a ladder (a difference is flagged); lr / epochs are handled separately
RECIPE_KEYS = ("method", "rank", "alpha", "sched", "opt", "layers", "rslora", "mask_pad", "dtype", "val", "steps",
               "tuned", "tuned_use")
SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "dgx" / "queues_snapshot"
REPORT_NAME = "ladder"


# --------------------------------------------------------------------------------------------------------------
# small pure helpers (unit-tested against hand-computed values)
# --------------------------------------------------------------------------------------------------------------
def classify_ci(ci_low: float, ci_high: float) -> str:
    """'llm' when the CI of macro-F1(LLM) - macro-F1(baseline) is entirely above 0, 'baseline' when entirely below,
    else 'neither' (a bound exactly at 0 is not an exclusion)."""
    if ci_low > 0:
        return "llm"
    if ci_high < 0:
        return "baseline"
    return "neither"


def classify_holm(diff: float, p_holm: float, alpha: float = ALPHA) -> str:
    if p_holm < alpha and diff > 0:
        return "llm"
    if p_holm < alpha and diff < 0:
        return "baseline"
    return "neither"


def wald_p(diff: float, boot_sd: float) -> float:
    """Two-sided normal-approximation p of a bootstrap difference: z = diff / bootstrap SD, p = erfc(|z| / sqrt 2).
    Used for the Holm family because the add-one bootstrap p is resolution-limited (smallest attainable 2 / (B + 1); after
    Holm over m tests nothing can be certified once m * 2 / (B + 1) >= 0.05, e.g. m = 72, B = 2000 -> 0.072). An
    approximation (assumes the bootstrap distribution is close to normal); returns 1.0 for an undefined SD."""
    if boot_sd is None or math.isnan(boot_sd):
        return 1.0
    if boot_sd <= 0:
        return 1.0 if diff == 0 else 0.0
    return float(math.erfc(abs(diff / boot_sd) / math.sqrt(2.0)))


def enforce_holm_not_stronger(tests: Sequence[dict]) -> int:
    """The Holm column uses a normal-approximation p; it must never be stronger than the unadjusted percentile interval it sits
    beside. Any test whose Holm class is a direction but whose CI class is not the same direction is reset to 'neither'
    (class_holm_raw keeps the original). Returns the number of overrides."""
    n = 0
    for t in tests:
        t["class_holm_raw"] = t["class_holm"]
        if t["class_holm"] != "neither" and t["class_holm"] != t["class_ci"]:
            t["class_holm"] = "neither"
            n += 1
    return n


def verdict_from_counts(n_llm: int, n_base: int, n_neither: int) -> str:
    n = n_llm + n_base + n_neither
    if n == 0:
        return "-"
    if n_llm == n:
        return "LLM ahead in all seeds"
    if n_base == n:
        return "baseline ahead in all seeds"
    return "mixed / unresolved"


def _midranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    ranks = np.empty(len(v), dtype=np.float64)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def exact_mannwhitney(x: Sequence[float], y: Sequence[float], max_perm: int = 2_000_000) -> dict | None:
    """Exact two-sided Mann-Whitney U (permutation) test on midranks. All C(n1+n2, n1) assignments are enumerated, so
    the p-value is exact (ties handled by midranks); p = fraction of assignments whose rank-sum deviates from its null
    mean at least as much as the observed one. None when a group is empty or the enumeration would exceed max_perm.
    p_min = 2 / C(n1+n2, n1) is the smallest p attainable with these group sizes (3 vs 3 -> 0.10)."""
    n1, n2 = len(x), len(y)
    if n1 < 1 or n2 < 1:
        return None
    N = n1 + n2
    total = math.comb(N, n1)
    if total > max_perm:
        return None
    pooled = np.round(np.asarray(list(x) + list(y), dtype=np.float64), 12)   # F1 ties = equal to 12 dp
    r = _midranks(pooled)
    obs = float(r[:n1].sum())
    mu = n1 * (N + 1) / 2.0
    sums = np.fromiter((r[list(c)].sum() for c in itertools.combinations(range(N), n1)), dtype=np.float64, count=total)
    p = float(np.count_nonzero(np.abs(sums - mu) >= abs(obs - mu) - 1e-9) / total)
    return {"U_x": obs - n1 * (n1 + 1) / 2.0, "p": p, "n_perm": int(total), "n1": n1, "n2": n2,
            "p_min": mannwhitney_min_p(n1, n2)}


def _ranking(seed_values: dict[str, float]) -> list[list[str]]:
    """Sizes grouped by (tied) F1, best first: [['1.5b'], ['3b', '7b'], ['0.5b']]."""
    items = sorted(seed_values.items(), key=lambda kv: (-kv[1], SIZES.index(kv[0]) if kv[0] in SIZES else 99))
    groups: list[list[str]] = []
    last: float | None = None
    for s, v in items:
        if last is not None and abs(v - last) <= TIE_TOL:
            groups[-1].append(s)
        else:
            groups.append([s])
        last = v
    return groups


def _ranking_str(seed_values: dict[str, float]) -> str:
    parts = []
    for grp in _ranking(seed_values):
        parts.append(" = ".join(f"{s} ({seed_values[s]:.4f})" for s in grp))
    return " > ".join(parts)


def analyse_ordering(values: dict[str, dict[int, float]]) -> dict:
    """values[size][seed] = strict macro-F1 of the comparable cell. Returns
      sizes_present / sizes_missing (relative to SIZES), seeds, complete_seeds (a cell at EVERY present size),
      per_seed {seed: {sizes, ranking, argmax (tuple, ties kept), complete}},
      argmax {n_complete, counts {argmax_tuple_str: n}, modal, n_agree, all_agree},
      peak {claimed, size, reason, edge, chance, margins, min_margin, sd_top, within_noise}
                                    -- 'claimed' only if all four sizes are present, every seed present is complete, there are >= 2
                                       seeds, they all have the same single argmax AND that size is interior (an argmax at either end
                                       of the ladder is never a peak). A claim is DESCRIPTIVE: the reason prints the chance level
                                       (1/k)^(n-1) of n seeds agreeing on one of k sizes if sizes were exchangeable and the margin
                                       over the runner-up size per seed against the across-seed SD; no between-size test is run,
      steps [ {from, to, absent_between, n_pairs, up, down, tie, seed_up..., mean_diff, mw (dict|None), note} ],
      shape  one-line reading built ONLY from the per-seed step directions."""
    sizes_present = [s for s in SIZES if values.get(s)]
    sizes_missing = [s for s in SIZES if not values.get(s)]
    seeds = sorted({sd_ for s in sizes_present for sd_ in values[s]})
    per_seed: dict[int, dict] = {}
    for sd_ in seeds:
        sv = {s: values[s][sd_] for s in sizes_present if sd_ in values[s]}
        rk = _ranking(sv) if sv else []
        per_seed[sd_] = {"sizes": list(sv), "ranking": _ranking_str(sv) if sv else "", "argmax": tuple(rk[0]) if rk else (),
                         "complete": len(sv) == len(sizes_present) and len(sizes_present) > 0, "values": sv}
    complete = [s for s in seeds if per_seed[s]["complete"]]
    counts: dict[str, int] = {}
    for s in complete:
        key = " = ".join(per_seed[s]["argmax"])
        counts[key] = counts.get(key, 0) + 1
    modal = max(counts, key=lambda k: (counts[k], k)) if counts else None
    n_agree = counts.get(modal, 0) if modal else 0
    all_agree = bool(complete) and len(counts) == 1
    peak = {"claimed": False, "size": None, "reason": "", "edge": None, "chance": None, "margins": {}, "min_margin": None,
            "sd_top": None, "within_noise": None}
    if not sizes_present:
        peak["reason"] = "no comparable cells"
    elif sizes_missing:
        peak["reason"] = f"size(s) {', '.join(sizes_missing)} absent: a peak over the ladder cannot be claimed"
    elif len(complete) < len(seeds):
        peak["reason"] = f"{len(seeds) - len(complete)} of {len(seeds)} seeds are missing a size"
    elif len(complete) < 2:
        peak["reason"] = "only one complete seed: agreement across seeds cannot be assessed"
    elif not all_agree:
        peak["reason"] = ("seeds disagree on the argmax (" + ", ".join(f"{k}: {v}/{len(complete)}" for k, v in counts.items()) +
                          f"): only {n_agree} of {len(complete)} seeds agree on {modal}")
    elif " = " in (modal or ""):
        peak["reason"] = f"seeds agree but the argmax is a tie ({modal})"
    elif modal in (sizes_present[0], sizes_present[-1]):
        peak.update(edge=True, size=modal)
        peak["reason"] = (f"{modal} is the highest size in all {len(complete)} complete seeds but it is the "
                          f"{'smallest' if modal == sizes_present[0] else 'largest'} size on the ladder: an edge of the ladder, not an interior peak "
                          "(no size beyond it was run), so no peak is claimed")
    else:
        k = len(sizes_present)
        chance = (1.0 / k) ** (len(complete) - 1)
        margins = {}
        for sd_ in complete:
            sv = per_seed[sd_]["values"]
            margins[sd_] = sv[modal] - max(v for s_, v in sv.items() if s_ != modal)
        top_sd = sd(list(values[modal].values()))
        min_m = min(margins.values())
        within = (not math.isnan(top_sd)) and min_m <= top_sd
        peak.update(claimed=True, size=modal, edge=False, chance=chance, margins=margins, min_margin=min_m, sd_top=top_sd, within_noise=within)
        peak["reason"] = (f"{modal} (an interior size) has the highest F1 in all {len(complete)} complete seeds; "
                          f"if the {k} sizes were exchangeable, {len(complete)} seeds would agree on some one size with probability "
                          f"(1/{k})^{len(complete) - 1} = {chance:.4f} (also the smallest value attainable with {len(complete)} seeds and {k} sizes"
                          f"{', and above 0.05' if chance > ALPHA else ''}); margin over the runner-up size per seed: "
                          + ", ".join(f"s{a_} {m_:+.4f}" for a_, m_ in margins.items())
                          + (f"; the smallest margin ({min_m:.4f}) is within the across-seed SD of {modal} ({top_sd:.4f}), i.e. inside seed noise"
                             if within else "" if math.isnan(top_sd) else f"; the smallest margin ({min_m:.4f}) exceeds the across-seed SD of {modal} ({top_sd:.4f})")
                          + " (test-row sampling noise between the sizes is not assessed, so a margin above the seed SD is not evidence of a real difference)"
                          + "; this is a per-seed ranking, not a tested peak: no between-size significance test was run")
    steps = []
    for a, b in zip(sizes_present[:-1], sizes_present[1:]):
        between = [s for s in SIZES if SIZES.index(a) < SIZES.index(s) < SIZES.index(b) and s in sizes_missing]
        pairs = sorted(set(values[a]) & set(values[b]))
        up = sum(1 for s in pairs if values[b][s] - values[a][s] > TIE_TOL)
        down = sum(1 for s in pairs if values[a][s] - values[b][s] > TIE_TOL)
        tie = len(pairs) - up - down
        xa, xb = list(values[a].values()), list(values[b].values())
        mw = exact_mannwhitney(xa, xb) if min(len(xa), len(xb)) >= MIN_SEEDS_EXACT else None
        steps.append({"from": a, "to": b, "absent_between": between, "n_pairs": len(pairs), "up": up, "down": down,
                      "tie": tie, "mean_from": mean(xa), "mean_to": mean(xb), "mean_diff": mean(xb) - mean(xa),
                      "n_from": len(xa), "n_to": len(xb), "mw": mw,
                      "mw_min_p": mannwhitney_min_p(len(xa), len(xb)), "sign_min_p": sign_test_min_p(len(pairs)) if pairs else None,
                      "mw_note": "" if mw else f"not computed: n_from={len(xa)}, n_to={len(xb)} (need >= {MIN_SEEDS_EXACT} seeds each)"})
    usable = [st for st in steps if st["n_pairs"] > 0]
    if not usable:
        shape = "no adjacent-size step has a seed present at both sizes: no direction to report"
    elif len(usable) < len(steps):
        shape = "at least one adjacent-size step has no seed present at both sizes: no monotone claim is made"
    elif all(st["up"] == 0 and st["down"] == 0 for st in usable):
        shape = "every step is a tie in every matched seed (F1 equal to 1e-12): no size effect is visible; no monotone claim is made"
    elif all(st["up"] == st["n_pairs"] for st in usable) and all(st["n_pairs"] >= 2 for st in usable):
        shape = "every step is up in every matched seed (increasing in size in each seed; direction counts only, not a tested trend)"
    elif all(st["down"] == st["n_pairs"] for st in usable) and all(st["n_pairs"] >= 2 for st in usable):
        shape = "every step is down in every matched seed (decreasing in size in each seed; direction counts only, not a tested trend)"
    elif any(st["n_pairs"] < 2 for st in usable):
        shape = "at least one step rests on a single matched seed: no monotone claim is made"
    else:
        shape = ("NOT monotone in every seed (the step directions are not all the same across steps and seeds; a tie counts as no "
                 "direction): no monotone claim is made")
    return {"sizes_present": sizes_present, "sizes_missing": sizes_missing, "seeds": seeds, "complete_seeds": complete,
            "per_seed": per_seed, "argmax": {"n_complete": len(complete), "counts": counts, "modal": modal,
                                             "n_agree": n_agree, "all_agree": all_agree},
            "peak": peak, "steps": steps, "shape": shape}


# --------------------------------------------------------------------------------------------------------------
# expected grid / cell checks
# --------------------------------------------------------------------------------------------------------------
def design_names() -> list[str]:
    return [f"{FAMILY}_{a}_{s}_lora_r16_s{sd_}" for a in ARMS for s in SIZES for sd_ in SEEDS]


def expected_names(snapshot_dir: Path | None = None) -> tuple[list[str], str, str | None]:
    """The LT_ names the DGX queue was built with (scripts/dgx/queues_snapshot/*/*.json); falls back to the design grid.
    Returns (names, source, note). note is set when the snapshot and the 2 x 4 x 3 design grid differ."""
    snap = SNAPSHOT_DIR if snapshot_dir is None else Path(snapshot_dir)
    names: list[str] = []
    if snap.is_dir():
        for f in sorted(snap.glob("*/*.json")):
            try:
                rows = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if isinstance(rows, list):
                names += [r["name"] for r in rows if isinstance(r, dict) and str(r.get("name", "")).startswith(FAMILY + "_")]
    design = design_names()
    if not names:
        return design, "design grid (2 arms x 4 sizes x 3 seeds); no LT_ names found in the queue snapshot", None
    names = sorted(set(names))
    note = None
    if set(names) != set(design):
        note = (f"queue snapshot has {len(names)} LT_ names vs {len(design)} in the 2 x 4 x 3 design grid; "
                f"only in snapshot: {sorted(set(names) - set(design))}; only in design: {sorted(set(design) - set(names))}")
    return names, "queue snapshot (scripts/dgx/queues_snapshot/*/*.json)", note


def _effective(c: Cell, n_train: int) -> dict:
    """Epochs / optimizer steps the runner used for this cell (attrib.py:116: total = steps or (n//4) * (epochs or 3))."""
    steps_cfg = c.steps
    ep_cfg = int(c.config.get("epochs") or 0)
    n_rows = n_train - c.val
    if steps_cfg:
        return {"epochs": None, "steps": steps_cfg, "rows_per_epoch": n_rows}
    ep = ep_cfg or DEFAULT_EPOCHS
    return {"epochs": ep, "steps": (n_rows // BATCH) * ep, "rows_per_epoch": n_rows}


def _recipe(c: Cell, n_train: int) -> dict:
    cfg, tr = c.config, (c.tuned_resolved or {})
    eff = _effective(c, n_train)
    chk = c.meta.get("eval_bs_check") if isinstance(c.meta.get("eval_bs_check"), dict) else None
    return {"method": c.method, "rank": c.rank, "alpha": c.alpha, "lr": c.lr, "epochs": eff["epochs"], "steps_total": eff["steps"],
            "sched": cfg.get("sched"), "opt": cfg.get("opt"), "layers": c.layers, "rslora": cfg.get("rslora"),
            "mask_pad": cfg.get("mask_pad"), "dtype": cfg.get("dtype"), "val": c.val, "steps": c.steps,
            "tuned": c.tuned, "tuned_use": cfg.get("tuned_use"),
            "tuned_from": tr.get("from"), "tuned_val_nll": tr.get("val_answer_nll"), "tuned_n_candidates": tr.get("n_candidates"),
            "tuned_fallback": bool(tr.get("fallback")), "has_tuned_resolved": bool(tr),
            "tr_lr": tr.get("lr"), "tr_epochs": tr.get("epochs"),
            "micro": cfg.get("micro"), "eval_bs": c.eval_bs,
            "eval_bs_check": None if chk is None else {"rows": chk.get("rows"), "identical": chk.get("identical_raw_vs_batch1")}}


OFF_VOCAB_TOL = 1e-4          # the runner rounds off_vocab_rate to 4 dp (attrib.py), so a recompute can differ by at most 5e-5


def _cell_flags(c: Cell, rec: dict, f1: float | None, ov: float | None = None) -> list[str]:
    """Warnings about one cell (never exclusions): recompute vs reported, name vs config, tuned_resolved vs config."""
    flags: list[str] = []
    if f1 is not None and c.f1_reported is not None and abs(f1 - c.f1_reported) > VERIFY_TOL:
        flags.append(f"recomputed F1 {f1:.4f} differs from the reported {c.f1_reported:.4f} by more than {VERIFY_TOL:g} (run `verify`)")
    if ov is not None and c.off_vocab is not None and abs(ov - c.off_vocab) > OFF_VOCAB_TOL:
        flags.append(f"recomputed off-vocab rate {ov:.4f} differs from the reported {c.off_vocab:.4f} by more than {OFF_VOCAB_TOL:g} (run `verify`)")
    p = c.parsed
    for key, cfgv in (("arm", c.arm), ("model", c.model), ("seed", c.seed)):
        if key in p and cfgv is not None and p[key] != cfgv:
            flags.append(f"name says {key}={p[key]!r} but config says {cfgv!r}")
    if rec["tuned"] is None:
        flags.append("no tuned= key: this cell did NOT adopt a tuned recipe (lr/epochs are the runner defaults or hand-set)")
    elif not rec["has_tuned_resolved"]:
        flags.append("tuned= is set but no tuned_resolved was recorded: cannot tell which recipe it adopted")
    if rec["tuned_fallback"]:
        flags.append("tuned_resolved.fallback = true: NO tuning results existed at launch, the cell ran the DEFAULT lr/epochs, not a tuned recipe")
    if rec["has_tuned_resolved"]:
        if rec["tr_lr"] is not None and rec["lr"] is not None and abs(float(rec["tr_lr"]) - float(rec["lr"])) > 1e-15:
            flags.append(f"tuned_resolved.lr {rec['tr_lr']} != config lr {rec['lr']}")
        if rec["tr_epochs"] is not None and rec["epochs"] is not None and int(rec["tr_epochs"]) != int(rec["epochs"]):
            flags.append(f"tuned_resolved.epochs {rec['tr_epochs']} != effective epochs {rec['epochs']}")
    chk = rec["eval_bs_check"]
    if rec["eval_bs"] and rec["eval_bs"] > 1:
        if chk is None:
            flags.append(f"eval_bs={rec['eval_bs']} but no eval_bs_check recorded (batched generation not validated)")
        elif chk.get("identical") != chk.get("rows"):
            flags.append(f"eval_bs_check: only {chk.get('identical')} of {chk.get('rows')} rows identical to batch-1 generation")
    return flags


def _not_comparable_reason(c: Cell, gold_ref: list[str] | None) -> str | None:
    if c.scored_on != "test" or c.eval_limit != 0 or c.n_scored != N_TEST:
        return f"not a full-test cell (scored_on={c.scored_on}, n_scored={c.n_scored}, eval_limit={c.eval_limit})"
    if not c.has_preds:
        return "no predictions file (only the reported F1 exists; nothing to bootstrap)"
    try:
        ids = [r.get("k", r.get("idx")) for r in c.iter_preds()]
    except Exception as e:  # noqa: BLE001
        return f"predictions file unreadable ({type(e).__name__}: {e})"
    if ids != list(range(N_TEST)):
        return (f"prediction rows/ids are not 0..{N_TEST - 1} in order ({len(ids)} rows in the preds file, {N_TEST} expected; "
                f"n_scored={c.n_scored}; ids permuted, duplicated or missing)")
    if gold_ref is None:
        return "the arm's test gold is unavailable"
    try:
        g, _ = c.gold_pred()
    except Exception as e:  # noqa: BLE001 - a half-written / malformed preds file must not take the report down
        return f"predictions file unreadable ({type(e).__name__}: {e})"
    if g != gold_ref:
        return "gold sequence differs from the arm's test gold (rows not aligned): not comparable to the baselines"
    return None


def _still_unreadable(path: Path) -> bool:
    """LOAD_ERRORS is process-global and may be stale (the file could have been completed since): re-check."""
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return False
    except Exception:  # noqa: BLE001
        return True


def _name_config_mismatch(c: Cell) -> list[str]:
    """Fields where the cell NAME (LT_<arm>_<size>_lora_r16_s<seed>) disagrees with its own config (empty when consistent)."""
    out = []
    for key, cfgv in (("arm", c.arm), ("model", c.model), ("seed", c.seed)):
        if key in c.parsed and cfgv is not None and c.parsed[key] != cfgv:
            out.append(f"name says {key}={c.parsed[key]!r}, config says {cfgv!r}")
    return out


def _own_f1(c: Cell) -> float | None:
    """The cell's F1 against its OWN gold (never comparable to the baselines); the reported value if the preds cannot be read."""
    try:
        return c.f1 if c.has_preds else c.f1_reported
    except Exception:  # noqa: BLE001
        return c.f1_reported


def _baselines(arm: str) -> tuple[dict, list[str] | None, list[str]]:
    """{name: {'f1','f1_reported','preds'|None,'note'}}, the arm's test gold (or None), notes."""
    notes: list[str] = []
    out: dict[str, dict] = {}
    try:
        gold = test_gold(arm)
    except AnalysisDataError as e:
        return {b: {"f1": None, "preds": None, "note": str(e)} for b in BASELINES}, None, [str(e)]
    try:
        canon = canonical_baselines(arm)
    except AnalysisDataError as e:
        canon = {}
        notes.append(str(e))
    for short in ("dt", "svc"):
        try:
            pred = baseline_preds(arm, short)
            out[short] = {"f1": canon.get(short, {}).get("f1"), "f1_reported": canon.get(short, {}).get("f1_reported"),
                          "preds": pred, "note": ""}
        except AnalysisDataError as e:
            out[short] = {"f1": None, "f1_reported": None, "preds": None, "note": str(e)}
            notes.append(str(e))
    try:
        b = bert_baseline(arm)
        if list(b["gold"]) != gold:
            raise AnalysisDataError(f"{arm}: BERT preds file gold differs from the baseline preds file gold")
        refit_p = SALAD2 / f"bert_{arm}_results.REFIT.json"
        refit = read_json(refit_p) if refit_p.is_file() else {}
        out["bert"] = {"f1": b["f1"], "f1_reported": b["f1_reported"], "preds": list(b["preds"]), "note": "",
                       "selection": b.get("selection"), "epochs": refit.get("epochs"), "checkpoint": b.get("checkpoint")}
    except AnalysisDataError as e:
        out["bert"] = {"f1": None, "f1_reported": None, "preds": None, "note": str(e)}
        notes.append(str(e))
    return out, gold, notes


# --------------------------------------------------------------------------------------------------------------
# the analysis
# --------------------------------------------------------------------------------------------------------------
def _stale_check(arm: str, recs: list[dict], all_cells: list[Cell]) -> list[str]:
    """Compare each adopted tuning cell with the winner the T_ grid gives NOW (same rule as the runner)."""
    out: list[str] = []
    try:
        from .tuning import select_winner
    except Exception:  # noqa: BLE001
        return ["stale-tuning check skipped: tuning module unavailable"]
    combos = sorted({r["recipe"]["tuned"] for r in recs if r["recipe"]["tuned"]})
    for key in combos:
        try:
            t_arm, t_method = key.split(":")
        except ValueError:
            out.append(f"tuned key {key!r} is not 'arm:method'")
            continue
        grid = [c for c in all_cells if c.family == "T" and c.arm == t_arm and c.method == t_method]
        if not grid:
            out.append(f"{key}: no T_ cells in the loaded directories, so the adopted recipe cannot be compared with the grid")
            continue
        win = select_winner(grid)
        if win is None:
            continue
        usable = [c for c in grid if c.val_nll is not None]
        loaded = {c.name for c in grid}
        adopted = sorted({r["recipe"]["tuned_from"] for r in recs if r["recipe"]["tuned"] == key and r["recipe"]["tuned_from"]})
        for a in adopted:
            n_c = sorted({r["recipe"]["tuned_n_candidates"] for r in recs if r["recipe"]["tuned_from"] == a and r["recipe"]["tuned_n_candidates"] is not None})
            n_c_s = ", ".join(str(x) for x in n_c) or "?"
            if a not in loaded or any(x > len(usable) for x in n_c):
                out.append(f"LOCAL GRID BEHIND {key}: cells adopted `{a}` with {n_c_s} candidates at launch, but only {len(usable)} usable T_ cells are "
                           f"synced here" + ("" if a in loaded else f" and `{a}` is not among them") + "; the DGX grid was extended after the last sync. "
                           "Sync the T_ cells before judging whether the ladder used the current winner (this is NOT a staleness finding).")
            elif a != win.name:
                out.append(f"STALE {key}: cells adopted `{a}` (grid had {n_c_s} candidates at launch) but the grid as synced now "
                           f"({len(usable)} usable T_ cells) selects `{win.name}` (lr {win.lr:g}); the ladder was NOT run with the current winner")
            elif any(x < len(usable) for x in n_c):
                out.append(f"INFO {key}: `{a}` is still the winner, but the grid has grown from {n_c_s} to {len(usable)} candidates since launch")
    return out


def _file_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except OSError:
        return None


def _shadowed(dirs: Sequence[str | Path] | None, all_cells: list[Cell]) -> list[dict]:
    """LT_ files that share a NAME with a cell in another loaded directory. load_cells keeps the first directory's copy and drops
    the rest without a word; here every dropped copy is listed with whether its bytes (JSON and preds) differ from the used copy."""
    used = {c.name: c for c in all_cells if c.family == FAMILY}
    out: list[dict] = []
    for name, path in scan_cell_files(dirs):
        c = used.get(name)
        if c is None or Path(path) == c.path:
            continue
        other_preds = Path(path).with_name(name + ".preds.jsonl")
        same_json = _file_digest(Path(path)) == _file_digest(c.path)
        same_preds = (other_preds.is_file() == c.preds_path.is_file()) and (not c.preds_path.is_file() or _file_digest(other_preds) == _file_digest(c.preds_path))
        out.append({"name": name, "used": str(c.path), "shadowed": str(path), "identical": bool(same_json and same_preds)})
    return sorted(out, key=lambda r: (r["name"], r["shadowed"]))


def _dirs_scanned(dirs: Sequence[str | Path] | None) -> list[dict]:
    ds = [ATTRIB_DIR, P22_ATTRIB_DIR] if dirs is None else [Path(d) for d in dirs]
    return [{"path": str(d), "exists": Path(d).is_dir()} for d in ds]


def _tuning_provenance(recs: list[dict], all_cells: list[Cell]) -> dict:
    """Which model / seed / hold-out the adopted tuning cells were run on (from the T_ cells themselves)."""
    by_name = {c.name: c for c in all_cells}
    names = sorted({r["recipe"]["tuned_from"] for r in recs if r["recipe"]["tuned_from"]})
    found = [by_name[n] for n in names if n in by_name]
    edges: dict[str, list[str]] = {}
    try:
        from .tuning import _edge_flags
        for c in found:
            grid = [g for g in all_cells if g.family == "T" and g.arm == c.arm and g.method == c.method and g.val_nll is not None]
            edges[c.name] = [f["kind"] for f in _edge_flags(c, grid) if f["severity"] == "warn"]
    except Exception:  # noqa: BLE001 - a diagnostic only; never take the report down
        edges = {}
    return {"adopted": names, "found": [c.name for c in found],
            "models": sorted({c.model for c in found if c.model}), "seeds": sorted({c.seed for c in found if c.seed is not None}),
            "val_rows": sorted({c.val for c in found if c.val}), "missing": [n for n in names if n not in by_name],
            "edge_flags": edges}


def build(dirs: Sequence[str | Path] | None = None, arm: str | None = None, B: int = 2000, seed: int = 0,
          snapshot_dir: Path | None = None) -> dict:
    all_cells = load_cells(dirs)
    # an --arm filter must not make a cell vanish because its config arm disagrees with its NAME: keep it if either says `arm`
    lt_all = [c for c in all_cells if c.family == FAMILY and (arm is None or arm in (c.arm, c.parsed.get("arm")) or (c.arm is None and c.parsed.get("arm") is None))]
    exp_names, exp_source, exp_note = expected_names(snapshot_dir)
    if arm is not None:
        exp_names = [n for n in exp_names if f"_{arm}_" in n]
    arms = [arm] if arm else list(ARMS)

    base: dict[str, dict] = {}
    gold_ref: dict[str, list[str] | None] = {}
    base_notes: dict[str, list[str]] = {}
    n_train: dict[str, int] = {}
    for a in arms:
        base[a], gold_ref[a], base_notes[a] = _baselines(a)
        try:
            n_train[a] = len(split_gold(a, "train"))
        except AnalysisDataError:
            n_train[a] = 5000

    comparable: list[dict] = []
    not_comparable: list[dict] = []
    scanned = {str(Path(d).resolve()) for d in ([ATTRIB_DIR, P22_ATTRIB_DIR] if dirs is None else dirs)}
    for path, msg in LOAD_ERRORS:                     # LT_ files that exist but could not be parsed (e.g. caught mid-sync)
        pp = Path(path)
        if pp.stem.startswith(FAMILY + "_") and str(pp.parent.resolve()) in scanned and pp.is_file() and _still_unreadable(pp):
            pn = parse_cell_name(pp.stem)
            if (arm is None or pn.get("arm") == arm) and pp.stem not in {n["name"] for n in not_comparable}:
                not_comparable.append({"name": pp.stem, "arm": pn.get("arm"), "size": pn.get("model"), "seed": pn.get("seed"),
                                       "f1_reported": None, "f1": None, "reason": f"result JSON unreadable ({msg})"})
    seen: dict[tuple, str] = {}
    # cells whose name agrees with their config claim a slot first; a name/config clash never wins a slot by sort order alone
    for c in sorted(lt_all, key=lambda c: (bool(_name_config_mismatch(c)), c.name)):
        a, size, sd_ = c.arm, c.model, c.seed
        item = {"name": c.name, "arm": a, "size": size, "seed": sd_, "f1_reported": c.f1_reported}
        if a not in arms or size not in SIZES or sd_ is None:
            item.update(reason=f"cannot be placed on the ladder (arm={a!r}, model={size!r}, seed={sd_!r})", f1=_own_f1(c))
            not_comparable.append(item)
            continue
        key = (a, size, sd_)
        rec = _recipe(c, n_train[a])
        reason = _not_comparable_reason(c, gold_ref[a])
        if reason is None and key in seen:
            reason = f"duplicate (arm, size, seed): `{seen[key]}` already holds this slot"
            mm = _name_config_mismatch(c)
            if mm:
                reason += f" ({'; '.join(mm)}: this cell's config places it in the same slot)"
        if reason is not None:
            item.update(reason=reason, f1=_own_f1(c), recipe=rec)
            not_comparable.append(item)
            continue
        seen[key] = c.name
        g, p = c.gold_pred()
        labels = sorted(set(gold_ref[a]))
        f1 = macro_f1(g, p, labels)
        ov = off_vocab_rate(g, p, labels)
        comparable.append({"name": c.name, "arm": a, "size": size, "seed": sd_, "f1": f1, "f1_reported": c.f1_reported,
                           "off_vocab": ov, "off_vocab_reported": c.off_vocab, "recipe": rec,
                           "resumed_from_step": int(c.meta.get("resumed_from_step") or 0),
                           "resume_recorded": "resumed_from_step" in c.meta,
                           "flags": _cell_flags(c, rec, f1, ov), "cmp": {}, "_pred": p})

    # ---- paired bootstrap of every comparable cell against every baseline (one shared seed => same resamples) ----
    tests: list[dict] = []
    for rec in comparable:
        a = rec["arm"]
        for bname in BASELINES:
            bl = base[a].get(bname, {})
            if bl.get("preds") is None or gold_ref[a] is None:
                rec["cmp"][bname] = {"available": False, "note": bl.get("note") or "baseline unavailable"}
                continue
            r = paired_bootstrap(gold_ref[a], rec["_pred"], bl["preds"], sorted(set(gold_ref[a])), B=B, seed=seed)
            d = {"available": True, "baseline_f1": bl["f1"], "diff": r["diff"], "p_z": wald_p(r["diff"], r["boot_sd"]), "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                 "frac_gt0": r["frac_gt0"], "frac_lt0": r["frac_lt0"], "p_boot": r["p_boot"], "p_boot_min": r["p_boot_min"],
                 "boot_sd": r["boot_sd"], "class_ci": classify_ci(r["ci_low"], r["ci_high"]),
                 "n_resamples_missing_label": r["n_resamples_missing_label"]}
            rec["cmp"][bname] = d
            tests.append(d)
    adj_boot = holm_adjust([t["p_boot"] for t in tests]) if tests else []
    adj_z = holm_adjust([t["p_z"] for t in tests]) if tests else []
    for t, pb, pz in zip(tests, adj_boot, adj_z):
        t["p_boot_holm"] = pb                       # resolution-limited (see wald_p): kept for the record
        t["p_holm"] = pz                            # Holm on the Wald-z p: this one classifies
        t["class_holm"] = classify_holm(t["diff"], pz)
    n_holm_override = enforce_holm_not_stronger(tests)
    for rec in comparable:
        rec.pop("_pred", None)

    # ---- per arm x size groups, wins matrices, orderings ----
    arm_out: dict[str, dict] = {}
    for a in arms:
        recs = [r for r in comparable if r["arm"] == a]
        sizes: dict[str, dict] = {}
        values: dict[str, dict[int, float]] = {}
        for s in SIZES:
            rs = sorted([r for r in recs if r["size"] == s], key=lambda r: r["seed"])
            if not rs:
                continue
            f1s = [r["f1"] for r in rs]
            values[s] = {r["seed"]: r["f1"] for r in rs}
            per_b: dict[str, dict] = {}
            for bname in BASELINES:
                cs = [(r["seed"], r["cmp"].get(bname, {})) for r in rs]
                ok = [(sd_, d) for sd_, d in cs if d.get("available")]
                counts = {k: sum(1 for _, d in ok if d["class_ci"] == k) for k in ("llm", "baseline", "neither")}
                holm = {k: sum(1 for _, d in ok if d["class_holm"] == k) for k in ("llm", "baseline", "neither")}
                bf = base[a].get(bname, {}).get("f1")
                per_b[bname] = {"baseline_f1": bf, "n": len(ok),
                                "diff_mean": (mean(f1s) - bf) if (bf is not None) else None,
                                "diff_per_seed": {sd_: d["diff"] for sd_, d in ok},
                                "ci_counts": counts, "holm_counts": holm,
                                "verdict": verdict_from_counts(counts["llm"], counts["baseline"], counts["neither"]),
                                "verdict_holm": verdict_from_counts(holm["llm"], holm["baseline"], holm["neither"])}
            sizes[s] = {"n": len(rs), "seeds": [r["seed"] for r in rs], "f1_per_seed": {r["seed"]: r["f1"] for r in rs},
                        "mean": mean(f1s), "sd": sd(f1s), "min": min(f1s), "max": max(f1s),
                        "off_vocab_per_seed": {r["seed"]: r["off_vocab"] for r in rs},
                        "off_vocab_mean": mean(r["off_vocab"] for r in rs), "off_vocab_max": max(r["off_vocab"] for r in rs),
                        "vs": per_b}
        order = analyse_ordering(values)
        best_cell = max(recs, key=lambda r: r["f1"]) if recs else None
        n_tied_best = sum(1 for r in recs if best_cell is not None and r is not best_cell and abs(r["f1"] - best_cell["f1"]) <= TIE_TOL)
        arm_out[a] = {"n_cells": len(recs), "n_cells_tied_with_best": n_tied_best, "sizes": sizes, "ordering": order,
                      "best_cell": None if best_cell is None else {"name": best_cell["name"], "f1": best_cell["f1"]},
                      "recipe": _arm_recipe(recs), "stale": _stale_check(a, recs, all_cells),
                      "tuning_provenance": _tuning_provenance(recs, all_cells), "baseline_notes": base_notes.get(a, [])}
    # Holm across every adjacent-size Mann-Whitney test in the report
    mw_steps = [st for a in arm_out.values() for st in a["ordering"]["steps"] if st["mw"]]
    if mw_steps:
        for st, pa in zip(mw_steps, holm_adjust([st["mw"]["p"] for st in mw_steps])):
            st["mw"]["p_holm"] = pa
    present = {r["name"] for r in comparable}
    listed = present | {n["name"] for n in not_comparable}
    expected_set = set(exp_names)
    coverage = {"source": exp_source, "note": exp_note, "n_expected": len(exp_names), "expected": exp_names,
                "n_comparable": len(present & expected_set), "n_present_files": len(listed & expected_set),
                "missing": sorted(expected_set - listed),
                "present_not_comparable": sorted(n["name"] for n in not_comparable if n["name"] in expected_set),
                "unexpected": sorted(listed - expected_set)}
    grid_models = sorted({c.model for c in all_cells if c.family == "T" and c.method == "lora" and c.model})
    caveats = build_caveats(base, arm_out, comparable, not_comparable, n_train, gold_ref, B, seed, len(tests), len(mw_steps), grid_models, n_holm_override)
    shadowed = _shadowed(dirs, all_cells)
    data = {"params": {"B": B, "seed": seed, "arm": arm, "dirs": None if dirs is None else [str(d) for d in dirs], "dirs_scanned": _dirs_scanned(dirs),
                       "family": FAMILY, "alpha": ALPHA, "n_tests": len(tests), "n_mw_tests": len(mw_steps),
                       "n_holm_overridden_to_neither": n_holm_override},
            "coverage": coverage,
            "baselines": {a: {b: {k: v for k, v in d.items() if k != "preds"} for b, d in base[a].items()} for a in arms},
            "arms": arm_out, "cells": comparable, "not_comparable": not_comparable, "shadowed": shadowed, "caveats": caveats}
    data["summary"] = summary_lines(data)
    return data


def _arm_recipe(recs: list[dict]) -> dict:
    """Distinct adopted recipes (per cell, never assumed), constancy of the other recipe fields, execution per size."""
    if not recs:
        return {"n": 0, "adopted": [], "constant": {}, "varying": {}, "execution": {}, "heterogeneous": False, "fallback_cells": []}
    # a recipe is what was trained (adopted cell, lr, epochs, fallback); the size of the candidate pool at launch is provenance, not recipe
    keyf = lambda r: (r["recipe"]["tuned_from"], r["recipe"]["lr"], r["recipe"]["epochs"], r["recipe"]["tuned_fallback"])
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        groups.setdefault(keyf(r), []).append(r)
    adopted = []
    for (frm, lr, ep, fb), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ncs = sorted({r["recipe"]["tuned_n_candidates"] for r in rs if r["recipe"]["tuned_n_candidates"] is not None})
        nlls = sorted({r["recipe"]["tuned_val_nll"] for r in rs if r["recipe"]["tuned_val_nll"] is not None})
        adopted.append({"from": frm, "lr": lr, "epochs": ep, "steps_total": rs[0]["recipe"]["steps_total"], "fallback": fb,
                        "n_candidates": ncs[0] if len(ncs) == 1 else (ncs or None), "val_nll": nlls[0] if len(nlls) == 1 else (nlls or None),
                        "n_cells": len(rs),
                        "cells": {s: sorted(r["seed"] for r in rs if r["size"] == s) for s in SIZES if any(r["size"] == s for r in rs)}})
    constant, varying = {}, {}
    for k in RECIPE_KEYS:
        vals = sorted({json.dumps(r["recipe"].get(k)) for r in recs})
        if len(vals) == 1:
            constant[k] = json.loads(vals[0])
        else:
            varying[k] = [json.loads(v) for v in vals]
    ex: dict[str, dict] = {}
    for s in SIZES:
        rs = [r for r in recs if r["size"] == s]
        if not rs:
            continue
        chk_rows = sum((r["recipe"]["eval_bs_check"] or {}).get("rows") or 0 for r in rs)
        chk_id = sum((r["recipe"]["eval_bs_check"] or {}).get("identical") or 0 for r in rs)
        ex[s] = {"n": len(rs), "micro": sorted({r["recipe"]["micro"] for r in rs}, key=str), "eval_bs": sorted({r["recipe"]["eval_bs"] for r in rs}),
                 "dtype": sorted({r["recipe"]["dtype"] for r in rs}, key=str),
                 "eval_bs_check_identical": chk_id, "eval_bs_check_rows": chk_rows,
                 "n_without_check": sum(1 for r in rs if r["recipe"]["eval_bs_check"] is None)}
    return {"n": len(recs), "adopted": adopted, "constant": constant, "varying": varying, "execution": ex,
            "heterogeneous": len(adopted) > 1 or bool(varying),
            "fallback_cells": sorted(r["name"] for r in recs if r["recipe"]["tuned_fallback"])}


def build_caveats(base, arm_out, comparable, not_comparable, n_train, gold_ref, B, seed, n_tests, n_mw, grid_models=(), n_override=0) -> list[str]:
    """The header block. Every number is derived from the data/files at run time; facts that are not recorded in a file
    are attributed to their source and labelled as such."""
    c: list[str] = []
    arms = list(base)
    n_tr = ", ".join(f"{a} {n_train[a]:,}" for a in arms)
    bert_ep = ", ".join(f"{a} {base[a].get('bert', {}).get('epochs') or '?'}" for a in arms)
    bert_sel = next((base[a]["bert"].get("selection") for a in arms if base[a].get("bert", {}).get("selection")), None)
    c.append("**Baseline exposure.** DT and LinearSVC are one deterministic fit each on the arm's whole training file "
             f"(`train_file` in `<arm>_baselines.json` = `data/salad2_<arm>/salad2_train.json`; rows counted here: {n_tr}), "
             "TF-IDF over the rendered flow-text prompt. BERT-REFIT is NOT trained on all of those rows: its result file "
             f"(`bert_<arm>_results.REFIT.json`) records the selection protocol ('{bert_sel or 'not recorded'}') and the epochs "
             f"({bert_ep}) but not the number of training rows; the P13 manuscript (Section VI-B) states it was trained from "
             "4,500 rows per arm (the other 500 held out for selection). Treat BERT's exposure as 4,500 rows, not 5,000.")
    n_lt = len(comparable)
    n_val0 = sum(1 for r in comparable if r["recipe"]["val"] == 0 and not r["recipe"]["steps"])
    resumed = sorted((r["name"], r.get("resumed_from_step")) for r in comparable if r.get("resumed_from_step"))
    if resumed:
        resume_txt = (f"{len(resumed)} of {n_lt} comparable cells resumed from a checkpoint after an interruption (`resumed_from_step` in the "
                      "cell's own JSON: " + ", ".join(f"{nm} at step {st}" for nm, st in resumed) + "); a resume restores adapter, optimizer "
                      "and scheduler state and replays the epoch shuffles, but is not bit-identical to an uninterrupted run")
    else:
        resume_txt = "no comparable cell records a resume (`resumed_from_step` is 0 or absent in every cell JSON)"
    unrecorded = sorted(r["name"] for r in comparable if not r.get("resume_recorded"))
    if unrecorded:
        resume_txt += (f"; the field is absent from {len(unrecorded)} cell(s) that predate it ({', '.join(unrecorded)}), which are counted as "
                       "not resumed - that rests on the lane's driver log, which is not part of these result files")
    lead = (f"{n_val0} of {n_lt} comparable LT cells have `val = 0` and `steps = 0`, so each trains" if n_lt else
            "No LT cell exists yet; by the runner's rules a cell with `val = 0` and `steps = 0` (checked per cell here once cells exist) trains")
    c.append("**LLM exposure (matched on distinct rows, not on passes).** " + lead + " on all 5,000 training rows for E epochs: "
             "(5000 // 4) x E = 1,250 x E optimizer steps (`scripts/dgx/attrib.py:85-90, 116`), reshuffled every epoch with "
             "`Random(seed + epoch)` (`attrib.py:127`), in one training loop per cell: " + resume_txt + ". So the batch-replay defect of P13's MLX ladder (20 of 24 cells replayed the same "
             "first 100 batches, about 400 distinct rows seen six times: the P13 manuscript, Section VI-B) does not apply here, and "
             "the DT/SVC baselines and the LLM see the same set of distinct rows. The LLM sees each row E times, DT/SVC once, BERT a "
             "few epochs over 4,500 rows: 'matched exposure' means the same rows, not the same number of passes."
             + ("" if n_val0 == n_lt else f" WARNING: {n_lt - n_val0} cell(s) do NOT satisfy this (see the recipe table)."))
    tp = {a: arm_out[a]["tuning_provenance"] for a in arms if arm_out[a]["n_cells"]}
    models = sorted({m for v in tp.values() for m in v["models"]})
    seeds_t = sorted({s for v in tp.values() for s in v["seeds"]})
    if tp and models:
        vals_t = sorted({v for x in tp.values() for v in x["val_rows"]})
        val_s = ", ".join(f"{v:,}" for v in vals_t) or "?"
        edge = sorted({f"`{k}` ({', '.join(fl)})" for x in tp.values() for k, fl in x.get("edge_flags", {}).items() if fl})
        c.append(f"**Tuned on {'/'.join(models)} only.** The adopted lr / epochs were selected on validation answer-NLL of {'/'.join(models)} "
                 f"tuning cells (seed(s) {', '.join(map(str, seeds_t))}, one run per grid point, {val_s} held-out TRAIN rows, the model trained on "
                 "the rest) and adopted UNCHANGED for every size on the ladder: the optimum can move with model size, so a size trend here mixes "
                 "size with a recipe selected on the smallest model, which need not be right for the larger ones."
                 + (f" The adopted grid point sits at an edge of the tried grid ({'; '.join(edge)}), so it was not bracketed and a better setting "
                    "may lie beyond it (see `tuning.md`)." if edge else "")
                 + " Also, each LT cell takes 1,250 x E steps against 1,125 x E for the grid point it adopted (about 11% more, because it "
                 "trains on 5,000 rows, not 4,500).")
    else:
        gm = ", ".join(grid_models) if grid_models else "none found"
        c.append("**Tuning provenance.** No LT cell records an adopted tuning cell that is present in the loaded directories, so which "
                 f"model the recipe was tuned on cannot be read from a cell. The LoRA T_ tuning cells that ARE loaded were run on: {gm}. "
                 "Whatever the grid model, the adopted lr / epochs would be applied unchanged to every size on the ladder (a recipe tuned "
                 "on the smallest model need not be optimal for 7B), and each LT cell takes 1,250 x E steps against 1,125 x E for the "
                 "adopted grid point (about 11% more, since it trains on 5,000 rows, not 4,500).")
    c.append("**Execution differs by size (not part of the recipe).** `micro` and `eval_bs` vary across sizes (see the execution table); "
             "with `mask_pad = 0` the loss is taken over padding too and padding and the loss mean are computed per micro-batch "
             "(`attrib.py:118-141`), so `micro = 2` is not numerically identical to `micro = 0`. Size is therefore partly confounded "
             "with micro-batching; `eval_bs_check` (batched vs batch-1 generation) is shown per size.")
    c.append(f"**Uncertainty is partial.** Paired-bootstrap CIs (B = {B:,}, seed {seed}, rows resampled jointly, label set fixed from the "
             "full gold, the SAME resamples for every test) measure test-row sampling of ONE trained model against ONE deterministic baseline "
             f"fit; they do not include training-seed variance. Seed spread comes from up to {len(SEEDS)} training seeds per size (the design; n per size is in "
             "every table) on a SINGLE fixed 5,000-row training "
             "file (it excludes variance from re-drawing the training rows), and seeds are matched across sizes by seed number only (the "
             "different models share nothing but the data-order seed), so 'per seed' orderings are matched-index comparisons, not true pairings.")
    c.append(f"**Small n.** With {len(SEEDS)} seeds per size (the design) an exact seed-level test cannot go below p = "
             f"{mannwhitney_min_p(3, 3):.2f} (Mann-Whitney) or {sign_test_min_p(3):.2f} (paired sign test), so no adjacent-size difference "
             "can be significant at 0.05 on seed direction alone; the step counts are descriptive. Exact tests are run only when both sizes "
             f"have >= {MIN_SEEDS_EXACT} seeds.")
    if n_tests:
        pmin = 2 / (B + 1)
        c.append(f"**Multiplicity.** {n_tests} cell x baseline bootstrap tests in this report. The seed counts by CI use unadjusted 95% "
                 "intervals (the primary, requested summary). The Holm-adjusted counts beside them use Holm over all "
                 f"{n_tests} tests on a normal-approximation p (z = difference / bootstrap SD): the add-one bootstrap p has resolution "
                 f"2/(B+1) = {pmin:.1e}, so after Holm over {n_tests} tests its smallest attainable adjusted p is {n_tests * pmin:.3f}"
                 + (" (>= 0.05: it cannot certify any test at this B, which is why the z p is used)" if n_tests * pmin >= ALPHA else
                    " (< 0.05: usable, but shown only in the JSON)") +
                 f". A Holm-adjusted direction is never allowed to be stronger than the unadjusted CI beside it: {n_override} test(s) whose z-based "
                 f"Holm p < 0.05 while the percentile CI spans 0 were reset to 'neither'. {n_mw} adjacent-size Mann-Whitney tests are Holm-adjusted separately.")
    else:
        c.append("**Multiplicity.** No bootstrap test was run (no comparable cells). When there are cells the report gives unadjusted-CI seed "
                 "counts as the primary summary and Holm-adjusted counts beside them (Holm over every cell x baseline test, on a "
                 "normal-approximation p from the bootstrap SD because the add-one bootstrap p is resolution-limited).")
    sup = {}
    for a in arms:
        if gold_ref.get(a):
            cnt: dict[str, int] = {}
            for g in gold_ref[a]:
                cnt[g] = cnt.get(g, 0) + 1
            lab = min(cnt, key=lambda k: (cnt[k], k))
            sup[a] = (len(cnt), lab, cnt[lab])
    if sup:
        c.append("**Macro-F1 is noisy on rare classes.** Macro-F1 averages over the gold-present classes, so a class with few test rows moves "
                 "it a lot: " + "; ".join(f"{a}: K = {k}, smallest class `{lab}` has {n} of {N_TEST:,} test rows" for a, (k, lab, n) in sup.items()) + ".")
    return c


def summary_lines(data: dict) -> list[str]:
    out: list[str] = []
    cov = data["coverage"]
    out.append(f"{cov['n_comparable']} of {cov['n_expected']} expected LT cells are present and comparable on the shared 10,000 test rows "
               f"({len(cov['missing'])} missing, {len(cov['present_not_comparable'])} present but not comparable).")
    if data.get("shadowed"):
        n_diff = sum(1 for r in data["shadowed"] if not r["identical"])
        out.append(f"WARNING {len(data['shadowed'])} LT_ file(s) share a name with a cell in an earlier directory and were NOT used (the first directory's copy "
                   f"wins; {n_diff} of the {len(data['shadowed'])} differ in content from the copy used): see 'Shadowed duplicates'.")
    for a, ad in data["arms"].items():
        if ad["n_cells"] == 0:
            out.append(f"{a}: no comparable LT cells yet.")
            continue
        o = ad["ordering"]
        bl = data["baselines"].get(a, {})
        blt = ", ".join(f"{BASELINE_NAME[b]} {fmt(bl.get(b, {}).get('f1'))}" for b in BASELINES)
        best = ad["best_cell"]
        n_tied = ad.get("n_cells_tied_with_best", 0)
        out.append(f"{a}: {ad['n_cells']} comparable cells over sizes {', '.join(o['sizes_present'])}; best single LLM cell "
                   f"{best['name']} = {best['f1']:.4f}{f' (tied with {n_tied} other cell(s) to 1e-12)' if n_tied else ''} against {blt} "
                   "(the best cell is picked on the test scores: descriptive, post-hoc).")
        rc0 = ad["recipe"]
        k = rc0["constant"]
        if rc0["adopted"]:
            top_r = max(rc0["adopted"], key=lambda x: x["n_cells"])
            out.append(f"{a}: recipe = {k.get('method', '?')} r{k.get('rank', '?')} alpha {k.get('alpha', '?')}, {k.get('opt', '?')} + {k.get('sched', '?')}, "
                       f"lr {'?' if top_r['lr'] is None else format(top_r['lr'], 'g')}, {top_r['epochs']} epochs = {top_r['steps_total']} steps, adopted from "
                       f"`{top_r['from'] or 'no tuning cell (fallback)'}` in {top_r['n_cells']} of {rc0['n']} cells (read from each cell's "
                       f"config.tuned_resolved; see the recipe table).")
        tot = {"llm": 0, "baseline": 0, "neither": 0}
        toth = {"llm": 0, "baseline": 0, "neither": 0}
        for sd_ in ad["sizes"].values():
            for b in BASELINES:
                for k in tot:
                    tot[k] += sd_["vs"][b]["ci_counts"][k]
                    toth[k] += sd_["vs"][b]["holm_counts"][k]
        out.append(f"{a}: of {sum(tot.values())} cell x baseline paired-bootstrap tests, the 95% CI excludes 0 in favour of the LLM in "
                   f"{tot['llm']}, in favour of the baseline in {tot['baseline']}, neither in {tot['neither']} (unadjusted); under Holm-adjusted "
                   f"p < 0.05 (Wald-z p from the bootstrap SD, Holm over all {data['params']['n_tests']} tests): LLM {toth['llm']}, "
                   f"baseline {toth['baseline']}, neither {toth['neither']}.")
        means = {s: v["mean"] for s, v in ad["sizes"].items()}
        mx = max(means.values())
        tops = [s for s in means if abs(means[s] - mx) <= TIE_TOL]
        top = " = ".join(tops) + (" (exact tie)" if len(tops) > 1 else "")
        out.append(f"{a}: highest seed-mean size = {top} ({mx:.4f}, n = {', '.join(str(ad['sizes'][s]['n']) for s in tops)} seed(s); post-hoc); peak claim: "
                   f"{'YES (descriptive only), ' if o['peak']['claimed'] else 'NO, '}{o['peak']['reason']}.")
        out.append(f"{a}: shape: {o['shape']}.")
        rc = ad["recipe"]
        if rc["fallback_cells"]:
            out.append(f"{a}: WARNING {len(rc['fallback_cells'])} cell(s) ran the FALLBACK recipe (no tuning results at launch), not a tuned one: "
                       f"{', '.join(rc['fallback_cells'])}.")
        if rc["heterogeneous"]:
            out.append(f"{a}: WARNING the cells do not share one recipe ({len(rc['adopted'])} distinct adopted lr/epochs/tuning-cell combinations"
                       f"{'; varying fields: ' + ', '.join(rc['varying']) if rc['varying'] else ''}): a size comparison mixes recipes.")
        for s in ad["stale"]:
            if s.startswith("STALE"):
                out.append(f"{a}: {s}")
            elif s.startswith("LOCAL GRID BEHIND"):
                out.append(f"{a}: NOTE {s}")
    return out


# --------------------------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------------------------
def _seeds_compact(cells: dict[str, list[int]]) -> str:
    return "; ".join(f"{s}: {','.join(map(str, v))}" for s, v in cells.items())


def _cmp_cell(d: dict) -> str:
    if not d.get("available"):
        return "n/a"
    tag = {"llm": "LLM", "baseline": "base", "neither": "ns"}[d["class_ci"]]
    return f"{d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] {tag}, f>0 {d['frac_gt0']:.3f}"


def render(data: dict) -> str:
    P = data["params"]
    cov = data["coverage"]
    L: list[str] = ["# Tuned model-size ladder (LT_ cells)", ""]
    L.append(f"Test set: each arm's 10,000 shared test rows (never validation). Cells: `LT_<arm>_<size>_lora_r16_s<seed>` scored on the "
             f"full test set; strict macro-F1 = mean per-class F1 over the gold-present labels, zero_division = 0. n = **{cov['n_comparable']} "
             f"comparable cells** of {cov['n_expected']} expected (expected set from the {cov['source']}). Paired bootstrap: B = {P['B']:,}, seed {P['seed']}, "
             f"{P['n_tests']} cell x baseline tests.")
    L.append("")
    L += ["## Summary", ""] + [f"- {s}" for s in data["summary"]] + ["", "## Read this first (caveats)", ""]
    L += [f"{i}. {c}" for i, c in enumerate(data["caveats"], 1)]
    L += ["", "## Coverage", ""]
    L += ["Directories scanned: " + "; ".join(f"`{d['path']}`" + ("" if d["exists"] else " (DOES NOT EXIST)") for d in P.get("dirs_scanned", [])) + ".", ""]
    if cov.get("note"):
        L += [f"> {cov['note']}", ""]
    if cov["n_comparable"] == 0 and not cov["present_not_comparable"]:
        L += [f"**No LT cell exists in the loaded directories yet** ({cov['n_expected']} expected). Nothing below is measured; the tables that "
              "follow list the reference baselines and the expected grid so the report is ready when the cells sync.", ""]
    exp_by: dict[tuple, dict] = {}
    for n in cov["expected"]:
        pn = parse_cell_name(n)
        if pn.get("arm") and pn.get("model") and pn.get("seed") is not None:
            exp_by.setdefault((pn["arm"], pn["model"]), {})[pn["seed"]] = n
    comp = {(r["arm"], r["size"], r["seed"]) for r in data["cells"]}
    nonc = {(r["arm"], r["size"], r["seed"]) for r in data["not_comparable"]}
    rows = []
    seeds_all = sorted({sd_ for v in exp_by.values() for sd_ in v})
    for (a, s), byseed in sorted(exp_by.items(), key=lambda kv: (kv[0][0], SIZES.index(kv[0][1]) if kv[0][1] in SIZES else 99)):
        rows.append([a, s] + [("ok" if (a, s, sd_) in comp else "NOT COMPARABLE" if (a, s, sd_) in nonc else ("missing" if sd_ in byseed else "not expected"))
                              for sd_ in seeds_all])
    if rows:
        L += [md_table(["arm", "size"] + [f"s{x}" for x in seeds_all], rows, "ll" + "l" * len(seeds_all)), ""]
    if cov["unexpected"]:
        L += [f"LT_ cells present but not in the expected grid: {', '.join(cov['unexpected'])}", ""]

    L += ["## Reference baselines (canonical, same 10,000 rows)", "",
          "n = 10,000 test rows per arm; one deterministic fit per baseline (no seed spread). DT / LinearSVC: `<arm>_baseline_preds.json`; BERT-REFIT: "
          "`<arm>_bert_preds.json` (validation-selected epochs, trained on 4,500 rows: see caveat 1).", ""]
    brow = []
    for a, bl in data["baselines"].items():
        for b in BASELINES:
            d = bl.get(b, {})
            brow.append([a, BASELINE_NAME[b], d.get("f1"), d.get("f1_reported"), d.get("note") or ""])
    L += [md_table(["arm", "baseline", "macro-F1 (recomputed)", "reported", "note"], brow, "llrrl"), ""]

    for a, ad in data["arms"].items():
        L += [f"## Arm: {a}", ""]
        if ad["n_cells"] == 0:
            L += ["No comparable LT cells for this arm.", ""]
            continue
        L += _render_recipe(a, ad)
        L += _render_results(a, ad, data)
        L += _render_ordering(a, ad)
    if data["not_comparable"]:
        L += ["## Present but NOT comparable (excluded from every mean, ordering and bootstrap)", ""]
        L += ["The F1 column is the cell's score on ITS OWN scored rows (validation, a subsample or a partial file): it is not a test result and is "
              "not comparable to the baselines.", ""]
        L += [md_table(["cell", "F1 on its own rows (NOT a test result)", "reason"],
                       [[r["name"], r.get("f1"), r["reason"]] for r in data["not_comparable"]], "lrl"), ""]
    if data.get("shadowed"):
        L += ["## Shadowed duplicates (same LT_ name in more than one scanned directory; only the first directory's copy is used)", ""]
        L += [md_table(["cell", "used", "not used", "content"],
                       [[r["name"], r["used"], r["shadowed"], "identical" if r["identical"] else "DIFFERS"] for r in data["shadowed"]], "llll"), ""]
    flagged = [r for r in data["cells"] if r["flags"]]
    if flagged:
        L += ["## Per-cell warnings (cells stay in the analysis)", ""]
        L += [f"- `{r['name']}`: " + "; ".join(r["flags"]) for r in flagged]
        L.append("")
    L += ["## Method", "",
          "- Strict macro-F1 recomputed from `NAME.preds.jsonl` (`common.macro_f1`), not copied from the JSON; a >6e-4 discrepancy is flagged.",
          "- A cell enters the analysis only if it is a full-test cell (10,000 rows, ids 0..9999 in order, gold identical to the arm's test gold).",
          "- Point difference = LLM F1 - baseline F1 (canonical `<arm>_baseline_preds.json` DT/LinearSVC; BERT = validation-selected REFIT). "
          "Paired bootstrap = `paired.paired_bootstrap` (rows resampled jointly, percentile 95% CI, label set fixed from the full gold).",
          "- Classification: `LLM` = whole CI above 0; `base` = whole CI below 0; `ns` = CI spans 0 (a bound exactly at 0 counts as spanning). "
          "Wins-matrix cell = `#seeds LLM / #seeds baseline / #seeds neither`, then the verdict word.",
          "- Ordering: sizes ranked per seed by F1; argmax over the sizes that seed has; a peak is claimed only when all four sizes are present, "
          "every seed is complete, there are >= 2 seeds, all agree on one size and that size is interior (an argmax at 0.5b or 7b is an edge of "
          "the ladder, never a peak). The claim is descriptive: the reason prints the chance level (1/k)^(n-1) of n seeds agreeing on one of k "
          "exchangeable sizes and the margin over the runner-up size against the across-seed SD; no between-size significance test is run. "
          "Step directions are counted per matched seed (up = the larger model scores higher by more than 1e-12).",
          "- Verdict words in the wins matrix are shorthand for the seed counts beside them; the unadjusted table and the Holm table can give "
          "different words, and the Holm one is the multiplicity-corrected reading.",
          "- Pre-specified: every comparable cell against each of DT / LinearSVC / BERT-REFIT, and every adjacent-size step. Post-hoc (selected by looking "
          "at the test scores, descriptive only): 'best single cell' and 'highest seed-mean size'.",
          "- Holm family for the cell x baseline tests = all of them in the report; Holm p uses z = difference / bootstrap SD (two-sided normal), "
          "because the add-one bootstrap p (smallest attainable 2 / (B + 1)) is resolution-limited after Holm; both are stored in the JSON.",
          "- Exact Mann-Whitney: permutation on midranks over all C(n1+n2, n1) assignments, two-sided; run only when both sizes have >= 3 seeds.",
          ""]
    return "\n".join(L)


def _render_recipe(a: str, ad: dict) -> list[str]:
    rc = ad["recipe"]
    L = [f"### {a}: recipe each cell adopted (`config.tuned_resolved`, read per cell)", ""]
    c = rc["constant"]
    if c:
        L.append("Constant across all cells: " + ", ".join(f"`{k}={v}`" for k, v in c.items()) + ".")
    if rc["varying"]:
        L.append("")
        L.append("**WARNING: recipe fields differ between cells:** " + "; ".join(f"`{k}` in {v}" for k, v in rc["varying"].items()) + ".")
    L.append("")
    L.append(md_table(["adopted from (T_ cell)", "lr", "epochs", "total steps", "val NLL", "grid size at launch", "fallback", "cells", "which"],
                      [[x["from"] or "(none recorded)", "-" if x["lr"] is None else f"{x['lr']:g}", x["epochs"], x["steps_total"],
                        fmt(x["val_nll"]) if not isinstance(x["val_nll"], list) else "/".join(fmt(v) for v in x["val_nll"]),
                        ", ".join(map(str, x["n_candidates"])) if isinstance(x["n_candidates"], list) else x["n_candidates"], "YES" if x["fallback"] else "no", x["n_cells"], _seeds_compact(x["cells"])]
                       for x in rc["adopted"]], "lrrrrrlrl"))
    if len(rc["adopted"]) > 1:
        L += ["", f"**{len(rc['adopted'])} distinct adopted recipes in this arm**: the tuning grid changed while the ladder was being queued, "
                  "so cells launched at different times used different lr/epochs. A size comparison mixes recipes."]
    if rc["fallback_cells"]:
        L += ["", "**FALLBACK cells (default lr/epochs, no tuning results existed when they launched):** " + ", ".join(f"`{x}`" for x in rc["fallback_cells"])]
    for s in ad["stale"]:
        L += ["", f"- {s}"]
    tp = ad["tuning_provenance"]
    if tp["adopted"]:
        L += ["", f"Adopted tuning cell(s) ran on model(s) **{', '.join(tp['models']) or 'unknown (not found)'}**, seed(s) "
                  f"{', '.join(map(str, tp['seeds'])) or '?'}, validation hold-out {', '.join(map(str, tp['val_rows'])) or '?'} rows"
                  + (f"; not found among the loaded cells: {', '.join(tp['missing'])}" if tp["missing"] else "") + "."]
    L += ["", "Execution per size (not recipe; sizes differ here):", ""]
    L.append(md_table(["size", "cells", "micro", "eval_bs", "dtype", "eval_bs_check identical / rows", "cells w/o check"],
                      [[s, e["n"], ",".join(map(str, e["micro"])), ",".join(map(str, e["eval_bs"])), ",".join(map(str, e["dtype"])),
                        f"{e['eval_bs_check_identical']} / {e['eval_bs_check_rows']}", e["n_without_check"]] for s, e in rc["execution"].items()],
                      "lllllrr"))
    return L + [""]


def _render_results(a: str, ad: dict, data: dict) -> list[str]:
    L = [f"### {a}: strict macro-F1 by size (n seeds, mean, sample SD)", ""]
    rows = []
    for s, v in ad["sizes"].items():
        per = ", ".join(f"s{sd_} {x:.4f}" for sd_, x in v["f1_per_seed"].items())
        rows.append([s, v["n"], v["mean"], "n=1, no SD" if v["n"] < 2 else v["sd"], per, v["off_vocab_mean"], v["off_vocab_max"]])
    L.append(md_table(["size", "n seeds", "mean F1", "SD (ddof=1)", "per seed", "off-vocab mean", "off-vocab max"], rows, "lrrrlrr"))
    L += ["", f"### {a}: point difference to the canonical baselines (mean LLM F1 - baseline F1) and paired bootstrap per cell", ""]
    bl = data["baselines"][a]
    rows = []
    for s, v in ad["sizes"].items():
        rows.append([s, v["n"], v["mean"]] + [fmt(v["vs"][b]["diff_mean"], 4, True) for b in BASELINES])
    L.append(md_table(["size", "n", "mean F1"] + [f"vs {BASELINE_NAME[b]} ({fmt(bl.get(b, {}).get('f1'))})" for b in BASELINES], rows, "lrrrrr"))
    L += ["", "Per cell (difference LLM - baseline [95% percentile CI, test rows resampled jointly]; `LLM` / `base` / `ns` = CI excludes 0 for the LLM / for the baseline / spans 0; "
          "`f>0` = fraction of the bootstrap resamples in which the LLM cell scores higher):", ""]
    crow = []
    for r in sorted((r for r in data["cells"] if r["arm"] == a), key=lambda r: (SIZES.index(r["size"]), r["seed"])):
        crow.append([r["size"], r["seed"], r["f1"]] + [_cmp_cell(r["cmp"].get(b, {})) for b in BASELINES])
    L.append(md_table(["size", "seed", "F1"] + [f"vs {BASELINE_NAME[b]}" for b in BASELINES], crow, "lrrlll"))
    L += ["", f"### {a}: wins matrix (# seeds whose 95% CI excludes 0 in favour of the LLM / of the baseline / neither; unadjusted CIs)", ""]
    wrow = []
    for s, v in ad["sizes"].items():
        wrow.append([s, v["n"]] + [f"{v['vs'][b]['ci_counts']['llm']} / {v['vs'][b]['ci_counts']['baseline']} / {v['vs'][b]['ci_counts']['neither']} -- {v['vs'][b]['verdict']} (unadjusted)"
                                   + ("" if v["n"] >= 3 else f" [only {v['n']} seed{'s' if v['n'] != 1 else ''}]")
                                   for b in BASELINES])
    L.append(md_table(["size", "n seeds"] + [BASELINE_NAME[b] for b in BASELINES], wrow, "lrlll"))
    L += ["", f"Same counts under Holm-adjusted p < 0.05 (Wald-z p from the bootstrap SD, Holm across all {data['params']['n_tests']} tests in the report; "
          "where the two tables give different verdict words, the Holm one is the multiplicity-corrected reading):", ""]
    hrow = []
    for s, v in ad["sizes"].items():
        hrow.append([s, v["n"]] + [f"{v['vs'][b]['holm_counts']['llm']} / {v['vs'][b]['holm_counts']['baseline']} / {v['vs'][b]['holm_counts']['neither']} -- {v['vs'][b]['verdict_holm']} (Holm)"
                                   + ("" if v["n"] >= 3 else f" [only {v['n']} seed{'s' if v['n'] != 1 else ''}]")
                                   for b in BASELINES])
    L.append(md_table(["size", "n seeds"] + [BASELINE_NAME[b] for b in BASELINES], hrow, "lrlll"))
    return L + [""]


def _render_ordering(a: str, ad: dict) -> list[str]:
    o = ad["ordering"]
    L = [f"### {a}: ladder ordering per seed", ""]
    if o["sizes_missing"]:
        L += [f"Sizes absent from this arm: {', '.join(o['sizes_missing'])} (orderings are over the sizes present: {', '.join(o['sizes_present'])}).", ""]
    rows = [[f"s{sd_}", "yes" if v["complete"] else f"NO ({len(v['sizes'])}/{len(o['sizes_present'])} sizes)", " = ".join(v["argmax"]), v["ranking"]]
            for sd_, v in o["per_seed"].items()]
    L.append(md_table(["seed", "complete", "argmax size", "ranking (F1)"], rows, "llll"))
    ag = o["argmax"]
    L += ["", f"Per-seed argmax agreement: {ag['n_agree']} of {ag['n_complete']} complete seeds agree on "
              f"{ag['modal'] if ag['modal'] else 'n/a'}" + (" (" + ", ".join(f"{k}: {v}" for k, v in ag["counts"].items()) + ")" if ag["counts"] else "") + ".",
          "", f"**Peak claim: {'YES (descriptive only) -- ' if o['peak']['claimed'] else 'NO -- '}{o['peak']['reason']}.**", "",
          f"Adjacent-size steps, direction per matched seed (up = the larger model scores higher). **{o['shape']}.**", ""]
    rows = []
    for st in o["steps"]:
        mw = st["mw"]
        rows.append([f"{st['from']} -> {st['to']}" + (f" ({', '.join(st['absent_between'])} absent)" if st["absent_between"] else ""),
                     st["n_pairs"], f"{st['up']} / {st['down']} / {st['tie']}",
                     fmt(st["mean_diff"], 4, True) + ("" if st["n_from"] == st["n_to"] == st["n_pairs"] else
                                                      f" (difference of group means: n = {st['n_from']} vs {st['n_to']} seeds, {st['n_pairs']} matched)"),
                     f"{mw['p']:.4f} (Holm {mw['p_holm']:.4f}; min attainable {mw['p_min']:.3f}; {mw['n_perm']} permutations)" if mw else st["mw_note"],
                     "-" if not st["n_pairs"] else f"{st['sign_min_p']:.3f}"])
    L.append(md_table(["step", "matched seeds", "up / down / tie", "mean difference", "exact Mann-Whitney p (two-sided)", "min sign-test p"], rows, "lrrrll"))
    return L + [""]


# --------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    dirs = getattr(args, "dirs", None)
    if dirs is not None:
        bad = [d for d in dirs if not Path(d).is_dir()]
        if bad:
            print(f"ladder: ERROR: not a directory: {', '.join(bad)}")
            return 2
    data = build(dirs, arm=getattr(args, "arm", None), B=args.B, seed=args.seed)
    md = render(data)
    if not getattr(args, "quiet", False):
        for s in data["summary"]:
            print(f"ladder: {s}")
    if not getattr(args, "no_save", False):
        md_p, _ = save_report(args.out, md, data)
        print(f"report: {md_p}")
    return 0


def register(subparsers: Any) -> None:
    p = subparsers.add_parser("ladder", help="tuned model-size ladder (LT_ cells): per-size stats, paired bootstrap vs DT/SVC/BERT, per-seed ordering")
    p.add_argument("--dirs", nargs="*", help="cell directories (default: dgx/attrib and dgx/p22_attrib)")
    p.add_argument("--arm", choices=list(ARMS), help="only this arm")
    p.add_argument("--B", type=int, default=2000, help="bootstrap resamples per cell x baseline test (default 2000)")
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed (default 0)")
    p.add_argument("--out", default=REPORT_NAME, help=f"report name (default: {REPORT_NAME})")
    p.add_argument("--no-save", action="store_true", help="print only, do not write a report")
    p.add_argument("--quiet", action="store_true", help="do not print the summary lines")
    p.set_defaults(func=run)


def run_default() -> int:
    return run(argparse.Namespace(dirs=None, arm=None, B=2000, seed=0, out=REPORT_NAME, no_save=False, quiet=False))
