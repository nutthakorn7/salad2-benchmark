"""tuning -- validation-tuning grid (T_<arm>_<method>_r<rank>_lr<lr>_e<epochs>_v500) analysis.

The T_ cells are scored on 500 held-out TRAIN rows (attrib.py val=500, Random(1234)); the test set is never touched.
Selection rule (attrib.py resolve_tuned): among the group's T_<arm>_<method>_*.json files, lowest `val_answer_nll`;
files are visited in sorted filename order and a later file replaces the incumbent only on a STRICT improvement, so
ties go to the first file in sorted order. The adopted epochs = round(steps / 1125).

Reports per (arm, method): the grid sorted by validation NLL next to validation F1, the lr x epochs coverage matrix,
the winner, edge flags (winner at an lr extreme, at the maximum epochs, at the edge of its own epoch slice with
untried neighbours) that say the grid may need extending, and a warning when NLL and F1 disagree on the winner
(with a paired bootstrap on the shared validation rows to show whether that disagreement is more than noise).
Also checks any cell that adopted `tuned=` against the winner the grid gives NOW (stale tuning).
"""
from __future__ import annotations

import argparse
from typing import Any

from .common import Cell, epochs_from_steps, fmt, gold_labels, group_by, load_cells, md_table, save_report
from .paired import paired_bootstrap

VAL_NOTE = ("validation: 500 held-out TRAIN rows (fixed hold-out, Random(1234)); the test set is never used here. "
            "Validation F1 on 500 rows is noisy for the rare classes (see the per-group support note).")


def select_winner(cells: list[Cell]) -> Cell | None:
    """attrib.py resolve_tuned, exactly: sorted by filename, strict '<' on val_answer_nll, skip cells without NLL."""
    best: Cell | None = None
    for c in sorted(cells, key=lambda c: c.name + ".json"):
        if c.val_nll is None:
            continue
        if best is None or c.val_nll < best.val_nll:  # type: ignore[operator]
            best = c
    return best


def _lr_str(lr: float) -> str:
    return f"{lr:g}"


def _edge_flags(winner: Cell, cells: list[Cell]) -> list[dict]:
    """Grid-edge diagnostics for the winner. Each flag: {'kind', 'severity', 'message'}."""
    flags: list[dict] = []
    lrs = sorted({c.lr for c in cells if c.lr is not None})
    eps = sorted({epochs_from_steps(c.steps) for c in cells if c.steps})
    w_lr, w_ep = winner.lr, epochs_from_steps(winner.steps)
    if len(lrs) > 1 and w_lr is not None:
        if w_lr == lrs[-1]:
            flags.append({"kind": "lr-upper-edge", "severity": "warn",
                          "message": f"winner lr {_lr_str(w_lr)} is the LARGEST lr in the grid {[_lr_str(x) for x in lrs]}: "
                                     f"a higher lr may be better -- extend the grid upward"})
        if w_lr == lrs[0]:
            flags.append({"kind": "lr-lower-edge", "severity": "warn",
                          "message": f"winner lr {_lr_str(w_lr)} is the SMALLEST lr in the grid: extend the grid downward"})
    if len(eps) > 1 and w_ep == eps[-1]:
        flags.append({"kind": "max-epochs-edge", "severity": "warn",
                      "message": f"winner trains for {w_ep} epochs = the MAXIMUM tried ({eps}): more epochs may help -- "
                                 f"extend the epoch axis"})
    if len(eps) > 1 and w_ep == eps[0]:
        flags.append({"kind": "min-epochs-edge", "severity": "info",
                      "message": f"winner trains for {w_ep} epochs = the minimum tried ({eps}); fewer epochs untested"})
    # edges within the winner's own epoch slice / lr slice (the grid is not a full factorial)
    slice_lrs = sorted({c.lr for c in cells if epochs_from_steps(c.steps) == w_ep and c.lr is not None})
    if len(slice_lrs) > 1 and w_lr is not None:
        untried_up = [x for x in lrs if x > slice_lrs[-1]]
        untried_dn = [x for x in lrs if x < slice_lrs[0]]
        if w_lr == slice_lrs[-1] and untried_up and w_lr != lrs[-1]:
            flags.append({"kind": "lr-slice-upper-edge", "severity": "warn",
                          "message": f"at {w_ep} epochs the winner has the highest lr tried ({_lr_str(w_lr)}); "
                                     f"higher lrs {[_lr_str(x) for x in untried_up]} exist in the grid but were NOT run at "
                                     f"{w_ep} epochs"})
        if w_lr == slice_lrs[0] and untried_dn and w_lr != lrs[0]:
            flags.append({"kind": "lr-slice-lower-edge", "severity": "warn",
                          "message": f"at {w_ep} epochs the winner has the lowest lr tried ({_lr_str(w_lr)}); lower lrs "
                                     f"{[_lr_str(x) for x in untried_dn]} were NOT run at {w_ep} epochs"})
    slice_eps = sorted({epochs_from_steps(c.steps) for c in cells if c.lr == w_lr and c.steps})
    if len(slice_eps) > 1 and w_ep == slice_eps[-1] and w_ep != eps[-1]:
        flags.append({"kind": "epochs-slice-edge", "severity": "info",
                      "message": f"at lr {_lr_str(w_lr)} the winner has the most epochs tried ({slice_eps})"})
    return flags


def _untried(cells: list[Cell]) -> list[tuple[float, int]]:
    lrs = sorted({c.lr for c in cells if c.lr is not None})
    eps = sorted({epochs_from_steps(c.steps) for c in cells if c.steps})
    have = {(c.lr, epochs_from_steps(c.steps)) for c in cells}
    return [(lr, e) for lr in lrs for e in eps if (lr, e) not in have]


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (average ranks for ties); nan when degenerate."""
    keep = [i for i in range(len(x)) if x[i] == x[i] and y[i] == y[i]]   # drop pairs with an unknown value
    x, y = [x[i] for i in keep], [y[i] for i in keep]
    if len(x) < 3:
        return float("nan")
    from scipy.stats import spearmanr  # scipy is available locally; kept lazy so the rest stays numpy-only
    r = spearmanr(x, y)
    return float(getattr(r, "statistic", r[0]))


def analyse_group(arm: str, method: str, cells: list[Cell], B: int, seed: int) -> dict:
    cells = sorted(cells, key=lambda c: c.name + ".json")
    scored_val = [c for c in cells if c.scored_on == "val" and c.val_nll is not None]
    excluded = [c.name for c in cells if c not in scored_val]
    out: dict[str, Any] = {"arm": arm, "method": method, "n_cells": len(cells), "n_usable": len(scored_val),
                           "excluded": excluded, "cells": [], "flags": [], "notes": [],
                           "models": sorted({str(c.model) for c in cells}), "seeds": sorted({c.seed for c in cells if c.seed is not None}),
                           "ranks": sorted({c.rank for c in cells if c.rank is not None})}
    if not scored_val:
        out["notes"].append("no usable validation cells")
        return out
    winner = select_winner(scored_val)
    by_nll = sorted(scored_val, key=lambda c: (c.val_nll, c.name + ".json"))
    # F1 is recomputed from the preds; a cell whose preds are missing/truncated shows its JSON value instead (the winner
    # is chosen on NLL alone, exactly like the runner, so it does not depend on the preds) and is named in the notes
    partial = [c.name for c in scored_val if not c.preds_complete]
    f1s = {c.name: (lambda v: float("nan") if v is None else float(v))(c.f1 if c.preds_complete else c.f1_reported)
           for c in scored_val}
    by_f1 = sorted(scored_val, key=lambda c: (-(f1s[c.name] if f1s[c.name] == f1s[c.name] else -1.0), c.name))
    nll_rank = {c.name: i + 1 for i, c in enumerate(by_nll)}
    f1_rank = {c.name: i + 1 for i, c in enumerate(by_f1)}
    for c in by_nll:
        out["cells"].append({"name": c.name, "lr": c.lr, "epochs": epochs_from_steps(c.steps), "steps": c.steps,
                             "rank": c.rank, "val_nll": c.val_nll, "val_f1": f1s[c.name], "off_vocab": c.off_vocab,
                             "nll_rank": nll_rank[c.name], "f1_rank": f1_rank[c.name], "winner": c is winner})
    w_ep = epochs_from_steps(winner.steps)
    out["winner"] = {"name": winner.name, "lr": winner.lr, "epochs": w_ep, "steps": winner.steps,
                     "val_nll": winner.val_nll, "val_f1": f1s[winner.name],
                     "resolved": {"lr": winner.lr, "epochs": w_ep}}
    if partial:
        out["notes"].append(f"preds missing or truncated for {partial}: their val F1 is the JSON value, not recomputed, and "
                            f"any NLL-vs-F1 bootstrap involving them is skipped (run `verify`)")
    ties = [c.name for c in scored_val if c.val_nll == winner.val_nll and c is not winner]
    if ties:
        out["notes"].append(f"NLL tie with the winner (4-dp rounding): {ties}; the runner takes the first in sorted "
                            f"filename order = {winner.name}")
    out["flags"] = _edge_flags(winner, scored_val)
    out["untried_grid_points"] = [{"lr": lr, "epochs": e} for lr, e in _untried(scored_val)]
    out["lrs"] = sorted({c.lr for c in scored_val if c.lr is not None})
    out["epochs_grid"] = sorted({epochs_from_steps(c.steps) for c in scored_val})
    # NLL vs F1 agreement
    best_f1 = by_f1[0]
    f1_ties = [c.name for c in by_f1 if f1s[c.name] == f1s[best_f1.name]]
    out["f1_winner"] = {"name": best_f1.name, "val_f1": f1s[best_f1.name], "val_nll": best_f1.val_nll, "ties": f1_ties}
    out["spearman_nll_vs_f1"] = _spearman([c.val_nll for c in scored_val], [f1s[c.name] for c in scored_val])
    disagree = winner.name not in f1_ties
    out["nll_f1_disagree"] = disagree
    if disagree:
        gw, gp = winner.gold_pred(), best_f1.gold_pred()
        boot = None
        if winner.preds_complete and best_f1.preds_complete and gw[0] == gp[0]:
            labs = gold_labels(gw[0])
            boot = paired_bootstrap(gw[0], gp[1], gw[1], labs, B=B, seed=seed)  # (best-F1 cell) - (NLL winner)
        out["disagreement"] = {"nll_winner": winner.name, "f1_winner": best_f1.name,
                               "f1_gap": f1s[best_f1.name] - f1s[winner.name],
                               "nll_gap": best_f1.val_nll - winner.val_nll,  # >= 0 by construction (winner has the lowest NLL)
                               "boot": None if boot is None else {k: boot[k] for k in
                                                                    ("diff", "ci_low", "ci_high", "frac_gt0", "B", "seed", "n")}}
    # support note (validation rows are the same for every cell of an arm)
    ref = winner if winner.preds_complete else next((c for c in scored_val if c.preds_complete), None)
    g = ref.gold_pred()[0] if ref is not None else []
    counts = {lab: g.count(lab) for lab in gold_labels(g)}
    out["val_support"] = counts
    out["rare_val_classes"] = sorted([lab for lab, k in counts.items() if k < 5])
    out["val_rows"] = len(g)
    return out


def check_stale_tuning(groups: dict[tuple[str, str], dict], all_cells: list[Cell]) -> list[dict]:
    """Cells that adopted tuned='arm:method' at run time vs the winner the CURRENT grid would give."""
    rows = []
    for c in all_cells:
        t = c.tuned
        if not t:
            continue
        tr = c.tuned_resolved or {}
        try:
            arm, method = t.split(":")
        except ValueError:
            rows.append({"cell": c.name, "tuned": t, "status": "malformed tuned key"})
            continue
        g = groups.get((arm, method))
        cur = g.get("winner") if g else None
        row = {"cell": c.name, "tuned": t, "resolved_from": tr.get("from"), "resolved_lr": tr.get("lr"),
               "resolved_epochs": tr.get("epochs"), "n_candidates_then": tr.get("n_candidates"),
               "n_candidates_now": g.get("n_usable") if g else 0, "current_winner": cur["name"] if cur else None,
               "fallback": bool(tr.get("fallback"))}
        if tr.get("fallback"):
            row["status"] = "FALLBACK (ran with default lr/epochs; tuning grid was empty then)"
        elif cur is None:
            row["status"] = "no current grid"
        elif tr.get("from") != cur["name"]:
            row["status"] = "STALE: adopted a different cell than the current winner"
        else:
            row["status"] = "consistent"
        rows.append(row)
    return rows


def build(arm: str | None = None, method: str | None = None, B: int = 2000, seed: int = 0) -> tuple[dict, list[Cell]]:
    all_cells = load_cells()
    tcells = [c for c in all_cells if c.family == "T"]
    grouped = group_by(tcells, lambda c: (c.arm, c.method))
    groups: dict[tuple[str, str], dict] = {}
    for (a, m), cs in sorted(grouped.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        if (arm and a != arm) or (method and m != method):
            continue
        groups[(a, m)] = analyse_group(a, m, cs, B, seed)
    return groups, all_cells


def render(groups: dict[tuple[str, str], dict], stale: list[dict], B: int) -> str:
    L = ["# Validation tuning grid (T_ cells)", "", f"**{VAL_NOTE}**", "",
         "Selection rule (attrib.py `resolve_tuned`): lowest `val_answer_nll`; ties go to the first file in sorted "
         "filename order; adopted epochs = round(steps / 1125).", ""]
    if not groups:
        L.append("No T_ cells found.")
        return "\n".join(L)
    for (arm, method), g in groups.items():
        L += [f"## {arm} / {method}  (n = {g['n_usable']} usable cells of {g['n_cells']})", "",
              f"Model(s) {', '.join(g['models'])}; rank(s) {', '.join(map(str, g['ranks']))}; training seed(s) "
              f"{', '.join(map(str, g['seeds']))} (one run per grid point: NLL/F1 differences between grid points include "
              f"seed noise that this design cannot separate).", ""]
        if not g.get("winner"):
            L += ["No usable validation cells.", ""]
            continue
        w = g["winner"]
        L.append(f"**Winner by validation NLL:** `{w['name']}` -- lr {_lr_str(w['lr'])}, {w['epochs']} epochs "
                 f"({w['steps']} steps), val NLL {w['val_nll']:.4f}, val F1 {w['val_f1']:.4f}. "
                 f"`tuned=\"{arm}:{method}\"` would adopt lr={_lr_str(w['lr'])}, epochs={w['epochs']}.")
        L.append("")
        # coverage matrix lr x epochs (val NLL, winner starred)
        lrs, eps = g["lrs"], g["epochs_grid"]
        cellmap = {(c["lr"], c["epochs"]): c for c in g["cells"]}
        rows = []
        for lr in lrs:
            r = [_lr_str(lr)]
            for e in eps:
                c = cellmap.get((lr, e))
                r.append("-" if c is None else (f"{c['val_nll']:.4f}*" if c["winner"] else f"{c['val_nll']:.4f}"))
            rows.append(r)
        L.append("Validation NLL by lr (rows) and epochs (columns); `-` = not run; `*` = winner:")
        L.append("")
        L.append(md_table(["lr \\ epochs"] + [f"e{e}" for e in eps], rows, "l" + "r" * len(eps)))
        L.append("")
        L.append("Grid sorted by validation NLL (validation set, not test):")
        L.append("")
        rows = [[c["nll_rank"], c["name"], _lr_str(c["lr"]), c["epochs"], c["val_nll"], c["val_f1"], c["f1_rank"],
                 fmt(c["val_nll"] - w["val_nll"], 4, True), "yes" if c["winner"] else ""] for c in g["cells"]]
        L.append(md_table(["NLL rank", "cell", "lr", "epochs", "val NLL", "val F1", "F1 rank", "NLL - best", "winner"],
                          rows, "lllrrrrrl"))
        L.append("")
        # flags
        L.append("Checks:")
        L.append("")
        any_flag = False
        for f in g["flags"]:
            any_flag = any_flag or f["severity"] == "warn"
            L.append(f"- [{f['severity'].upper()}] {f['message']}")
        if g["untried_grid_points"]:
            pts = ", ".join(f"(lr {_lr_str(p['lr'])}, e{p['epochs']})" for p in g["untried_grid_points"])
            L.append(f"- [INFO] grid is not a full factorial; untried points: {pts}")
        for nte in g["notes"]:
            L.append(f"- [NOTE] {nte}")
        if g["nll_f1_disagree"]:
            d = g["disagreement"]
            sp = g["spearman_nll_vs_f1"]
            s = (f"- [WARN] NLL and val F1 DISAGREE on the winner: NLL picks `{d['nll_winner']}` (F1 {g['winner']['val_f1']:.4f}), "
                 f"best val F1 is `{d['f1_winner']}` ({g['f1_winner']['val_f1']:.4f}, NLL +{d['nll_gap']:.4f} worse). ")
            b = d.get("boot")
            if b:
                s += (f"Paired bootstrap of that F1 gap on the {b['n']} shared validation rows (B={b['B']}, seed {b['seed']}): "
                      f"{b['diff']:+.4f}, 95% CI [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}], frac>0 = {b['frac_gt0']:.3f}. ")
                s += ("The gap is within row-sampling noise." if b["ci_low"] <= 0 <= b["ci_high"]
                      else "The F1 gap is NOT explained by row-sampling noise alone.")
            L.append(s)
            any_flag = True
        else:
            L.append("- [OK] NLL winner is also (one of) the best validation F1 cell(s).")
        if g["spearman_nll_vs_f1"] == g["spearman_nll_vs_f1"]:
            rho = g["spearman_nll_vs_f1"]
            how = ("the two criteria order the grid almost identically" if rho <= -0.8 else
                   "the two criteria mostly agree" if rho <= -0.5 else
                   "the two criteria only weakly agree" if rho < 0 else "the two criteria do NOT agree (lower NLL does not go with higher F1)")
            L.append(f"- Spearman rank correlation of val NLL vs val F1 over the {g['n_usable']} cells: {rho:+.3f} "
                     f"(-1 = identical ordering, 0 = unrelated; descriptive, n is small and no p-value is reported): {how}.")
        if g["rare_val_classes"]:
            L.append(f"- [NOTE] val F1 rests on {g['val_rows']} rows; classes with < 5 validation rows: "
                     f"{', '.join(g['rare_val_classes'])} (their per-class F1 is close to a coin flip).")
        if g["excluded"]:
            L.append(f"- [WARN] excluded (not scored on validation, or no val_answer_nll): {g['excluded']}")
        L.append("")
        if any_flag:
            L.append("**Action:** at least one flag above says the tuned value may not be a true optimum.")
            L.append("")
    L += ["## Downstream cells that adopted `tuned=...`", ""]
    if stale:
        L.append(md_table(["cell", "tuned", "resolved from", "lr", "epochs", "candidates then", "candidates now",
                           "current winner", "status"],
                          [[s["cell"], s["tuned"], s.get("resolved_from") or "-", s.get("resolved_lr"),
                            s.get("resolved_epochs"), s.get("n_candidates_then"), s.get("n_candidates_now"),
                            s.get("current_winner") or "-", s["status"]] for s in stale], "lllrrrrll"))
    else:
        L.append("None of the synced cells uses `tuned=` (the tuned ladder LT_ cells have not been synced yet), so "
                 "there is nothing to check for stale tuning.")
    return "\n".join(L)


def _jsonable_groups(groups: dict) -> dict:
    return {f"{a}:{m}": g for (a, m), g in groups.items()}


def run(args: argparse.Namespace) -> int:
    groups, all_cells = build(args.arm, args.method, B=args.B, seed=args.seed)
    stale = check_stale_tuning(groups, all_cells)
    md = render(groups, stale, args.B)
    data = {"note": VAL_NOTE, "groups": _jsonable_groups(groups), "stale_tuning": stale}
    md_p, _ = save_report(args.out, md, data)
    for (arm, method), g in groups.items():
        w = g.get("winner")
        if not w:
            print(f"tuning {arm}/{method}: no usable cells")
            continue
        edge = [f["kind"] for f in g["flags"] if f["severity"] == "warn"]
        print(f"tuning {arm}/{method}: n={g['n_usable']} winner={w['name']} (lr {_lr_str(w['lr'])}, e{w['epochs']}, "
              f"NLL {w['val_nll']:.4f}, F1 {w['val_f1']:.4f}) edge-flags={edge or 'none'} "
              f"nll_f1_disagree={g['nll_f1_disagree']}")
    print(f"report: {md_p}")
    return 0


def register(subparsers: Any) -> None:
    p = subparsers.add_parser("tuning", help="validation-tuning grid: winner by val NLL, grid-edge flags, NLL-vs-F1 check")
    p.add_argument("--arm", help="only this arm")
    p.add_argument("--method", choices=["lora", "oft"], help="only this method")
    p.add_argument("--B", type=int, default=2000, help="bootstrap resamples for the NLL-vs-F1 disagreement check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="tuning", help="report name (default: tuning)")
    p.set_defaults(func=run)


def run_default() -> int:
    return run(argparse.Namespace(arm=None, method=None, B=2000, seed=0, out="tuning"))
