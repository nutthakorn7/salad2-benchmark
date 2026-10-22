#!/usr/bin/env python3
"""P13 fact sheet: every number the SALAD-v2 manuscript quotes about the LLM ladders, computed from stored result files.

    python3 scripts/p13_fact_sheet.py            # writes results/salad2/dgx/analysis/p13_fact_sheet.{json,md}
    python3 scripts/p13_fact_sheet.py --check papers/p13-salad-v2/main.tex
        # every 4-dp number in the ladder tables of main.tex must appear in the fact sheet (exit 1 otherwise)

Sources (read only):
  * torch / DGX tuned ladder: results/salad2/dgx/analysis/ladder.json (written by `python3 -m dgx_analysis ladder`,
    which recomputes F1 from the per-row predictions and runs the paired bootstrap) + each LT_ cell's own JSON
  * MLX ladder (untuned, 4-bit): results/salad2/full_test/<arm>_<size>_s<seed>_full.json (10,000-row test)
  * same-rows DT / LinearSVC for the MLX cells: results/salad2/p13_same_rows/same_rows_out.json
  * data-file checksums: data/salad2_<arm>/salad2_{train,test}.json vs papers/p13-salad-v2/ARTIFACT.md
Nothing is written outside results/salad2/dgx/analysis/ (override with DGX_ANALYSIS_OUT).

`--check` is a numeric-token check only: it verifies that every four-decimal number printed in the manuscript occurs somewhere in the
fact sheet (or in the older, separately sourced legacy_numbers.txt, when present). It does not verify which seed, table cell or sign a
number belongs to, nor any counted claim ("five of six", Holm counts); those are read from the generated fact sheet by hand.
The manuscript source and the built corpus are not part of the public artifact, so there `--check` is for the authors and the
checksum section of the fact sheet is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SALAD2 = ROOT / "results" / "salad2"
ANALYSIS = Path(os.environ["DGX_ANALYSIS_OUT"]) if os.environ.get("DGX_ANALYSIS_OUT") else SALAD2 / "dgx" / "analysis"
LADDER_DIR = SALAD2 / "dgx" / "analysis"          # where `python3 -m dgx_analysis ladder` writes ladder.json
ATTRIB = SALAD2 / "dgx" / "attrib"
LADDER = ANALYSIS / "ladder.json" if (ANALYSIS / "ladder.json").exists() else LADDER_DIR / "ladder.json"   # read the ladder report written next to the output when there is one
SAME_ROWS = SALAD2 / "p13_same_rows" / "same_rows_out.json"
FULL_TEST = SALAD2 / "full_test"
SIZES = ["0.5b", "1.5b", "3b", "7b"]
ARMS = ["unsw", "cicids"]
SEEDS = [42, 77, 123]


def mean_sd(v):
    return (statistics.fmean(v), statistics.stdev(v)) if len(v) > 1 else (statistics.fmean(v), None)


def r4(x):
    return None if x is None else round(float(x), 4)


def torch_ladder() -> dict:
    d = json.loads(LADDER.read_text())
    out: dict = {"B": d["params"]["B"], "bootstrap_seed": d["params"]["seed"], "n_cells": len(d["cells"]),
                 "n_expected": d["coverage"]["n_expected"], "baselines": {}, "groups": {}, "recipe": {}, "execution": {},
                 "cells": [], "size_steps": {}}
    for arm in ARMS:
        out["baselines"][arm] = {k: {"f1": r4(v["f1"])} for k, v in d["baselines"][arm].items()}
        out["baselines"][arm]["bert"]["epochs"] = d["baselines"][arm]["bert"].get("epochs")
    for c in d["cells"]:
        j = json.loads((ATTRIB / f"{c['name']}.json").read_text())
        cfg = j["config"]
        out["cells"].append({
            "name": c["name"], "arm": c["arm"], "size": c["size"], "seed": c["seed"], "f1": r4(c["f1"]), "f1_raw": float(c["f1"]),
            "off_vocab": r4(c["off_vocab"]), "resumed_from_step": j.get("resumed_from_step", 0),
            "trainable_params": j.get("trainable_params"), "train_seconds": j.get("train_seconds"),
            "eval_bs": c["recipe"]["eval_bs"], "micro": c["recipe"]["micro"], "torch": j.get("torch"),
            "device": j.get("device"), "eval_bs_check": c["recipe"]["eval_bs_check"],
            "lr": c["recipe"]["lr"], "epochs": c["recipe"]["epochs"], "steps_total": c["recipe"]["steps_total"],
            "tuned_from": c["recipe"]["tuned_from"], "alpha": c["recipe"]["alpha"], "rank": c["recipe"]["rank"],
            "mask_pad": c["recipe"]["mask_pad"], "opt": c["recipe"]["opt"], "sched": c["recipe"]["sched"],
            "dtype": c["recipe"]["dtype"], "eval_limit": cfg.get("eval_limit"),
            "cmp": {b: {k: c["cmp"][b][k] for k in ("diff", "ci_low", "ci_high", "class_ci", "class_holm", "p_boot",
                                                      "p_holm", "p_z", "boot_sd")} for b in ("dt", "svc", "bert")}})
    for arm in ARMS:
        for size in SIZES:
            cs = sorted((c for c in out["cells"] if c["arm"] == arm and c["size"] == size), key=lambda c: c["seed"])
            if not cs:
                continue
            f = [c["f1"] for c in cs]
            m, s = mean_sd([c["f1_raw"] for c in cs])          # unrounded values, as dgx_analysis ladder does
            g = {"n": len(cs), "seeds": [c["seed"] for c in cs], "f1": f, "mean": r4(m), "sd": r4(s),
                 "off_vocab_mean": r4(statistics.fmean(c["off_vocab"] for c in cs)),
                 "off_vocab_max": r4(max(c["off_vocab"] for c in cs)), "vs": {}}
            for b in ("dt", "svc", "bert"):
                cls = [c["cmp"][b]["class_ci"] for c in cs]
                hol = [c["cmp"][b]["class_holm"] for c in cs]
                g["vs"][b] = {"mean_diff": r4(m - float(d["baselines"][arm][b]["f1"])),
                              "per_seed": [{"seed": c["seed"], "diff": r4(c["cmp"][b]["diff"]),
                                            "ci": [r4(c["cmp"][b]["ci_low"]), r4(c["cmp"][b]["ci_high"])],
                                            "class_ci": c["cmp"][b]["class_ci"], "class_holm": c["cmp"][b]["class_holm"]}
                                           for c in cs],
                              "unadjusted": {k: cls.count(k) for k in ("llm", "baseline", "neither")},
                              "holm": {k: hol.count(k) for k in ("llm", "baseline", "neither")}}
            out["groups"][f"{arm}/{size}"] = g
        cs = [c for c in out["cells"] if c["arm"] == arm]
        out["recipe"][arm] = {k: sorted({str(c[k]) for c in cs}) for k in
                              ("lr", "epochs", "steps_total", "rank", "alpha", "opt", "sched", "dtype", "mask_pad", "tuned_from")}
        for size in SIZES:
            sc = [c for c in cs if c["size"] == size]
            if sc:
                out["execution"][f"{arm}/{size}"] = {
                    "micro": sorted({c["micro"] for c in sc}), "eval_bs": sorted({c["eval_bs"] for c in sc}),
                    "eval_bs_check_identical": sum(c["eval_bs_check"]["identical"] for c in sc),
                    "eval_bs_check_rows": sum(c["eval_bs_check"]["rows"] for c in sc),
                    "resumed_cells": [c["name"] for c in sc if c["resumed_from_step"]],
                    "trainable_params": sorted({c["trainable_params"] for c in sc})}
        o = d["arms"][arm]["ordering"]
        out["size_steps"][arm] = {"steps": o.get("steps"), "shape": o.get("shape"), "peak": o.get("peak")}
    return out


def mlx_ladder() -> dict:
    same = json.loads(SAME_ROWS.read_text()) if SAME_ROWS.exists() else None
    out = {"cells": [], "groups": {}, "same_rows_available": same is not None}
    base = json.loads(LADDER.read_text())["baselines"]      # the same canonical DT / LinearSVC / BERT-REFIT numbers
    for arm in ARMS:
        for size in SIZES:
            for seed in SEEDS:
                p = FULL_TEST / f"{arm}_{size}_s{seed}_full.json"
                if not p.exists():
                    continue
                j = json.loads(p.read_text())
                # regime: UNSW s42 0.5B-3B = one uninterrupted 600-step run; UNSW 7B s42 = first 300 steps (+ replayed 300);
                # every other cell = six replayed 100-step chunks (main.tex "delivered identically" paragraph)
                regime = "steps100"
                if arm == "unsw" and seed == 42:
                    regime = "steps300" if size == "7b" else "steps600"
                cell = {"name": f"{arm}_{size}_s{seed}", "arm": arm, "size": size, "seed": seed, "n": j.get("n"),
                        "f1": r4(j["strict_macro_f1"]), "regime": regime}
                if same is not None:
                    sr = same[f"{arm}_s{seed}_{regime}"]
                    cell.update({"same_rows_n": sr["n"], "same_rows_dt": r4(sr["dt"]), "same_rows_svc": r4(sr["svc"]),
                                 "beats_same_rows_dt": cell["f1"] > sr["dt"], "beats_same_rows_svc": cell["f1"] > sr["svc"]})
                out["cells"].append(cell)
    for arm in ARMS:
        dt, svc = r4(base[arm]["dt"]["f1"]), r4(base[arm]["svc"]["f1"])
        cs = [c for c in out["cells"] if c["arm"] == arm]
        out.setdefault("arm_totals", {})[arm] = {
            "n_cells": len(cs), "dt_5000": dt, "svc_5000": svc,
            "best_cell_f1": max(c["f1"] for c in cs),
            "same_rows_beats_dt": sum(1 for c in cs if c.get("beats_same_rows_dt")),
            "same_rows_beats_svc": sum(1 for c in cs if c.get("beats_same_rows_svc")),
            "same_rows_dt_range": [min(c["same_rows_dt"] for c in cs), max(c["same_rows_dt"] for c in cs)] if same else None,
            "same_rows_svc_range": [min(c["same_rows_svc"] for c in cs), max(c["same_rows_svc"] for c in cs)] if same else None}
        for size in SIZES:
            sc = sorted((c for c in cs if c["size"] == size), key=lambda c: c["seed"])
            if sc:
                m, s = mean_sd([c["f1"] for c in sc])
                out["groups"][f"{arm}/{size}"] = {"n": len(sc), "f1": [c["f1"] for c in sc], "mean": r4(m), "sd": r4(s)}
    return out


def tuning_grid() -> dict:
    """The LoRA tuning cells the adopted lr / epochs were selected from (one run per grid point, 500 held-out train rows).
    Per arm: every T_<arm>_lora_r16_* cell, the winner (minimum validation answer-token NLL), the number of distinct
    (lr, epochs) configurations, any configuration run more than once (e.g. an eval-batch-1 retrain), and at which epoch
    counts a learning rate above the winner's was tried."""
    out: dict = {}
    for arm in ARMS:
        cells = []
        for p in sorted(ATTRIB.glob(f"T_{arm}_lora_r16_*.json")):
            j = json.loads(p.read_text())
            cfg = j["config"]
            per_epoch = (5000 - cfg["val"]) // 4                       # batch 4 over the rows that remain after the validation hold-out
            cells.append({"name": p.stem, "lr": float(cfg["lr"]), "epochs": round(cfg["steps"] / per_epoch), "steps": cfg["steps"],
                          "val_nll": j.get("val_answer_nll"), "eval_bs": j.get("eval_bs")})
        win = min(cells, key=lambda c: c["val_nll"])
        configs: dict = {}
        for c in cells:
            configs.setdefault((c["lr"], c["epochs"]), []).append(c)
        out[arm] = {"n_entries": len(cells), "n_distinct_configs": len(configs), "winner": win["name"], "winner_nll": win["val_nll"],
                    "winner_lr": win["lr"], "winner_epochs": win["epochs"], "max_epochs_in_grid": max(c["epochs"] for c in cells),
                    "epochs_with_higher_lr_than_winner": sorted({c["epochs"] for c in cells if c["lr"] > win["lr"]}),
                    "repeated_configs": {f"lr{lr:g}_e{e}": [{"name": c["name"], "val_nll": c["val_nll"], "eval_bs": c["eval_bs"]} for c in v]
                                         for (lr, e), v in configs.items() if len(v) > 1},
                    "cells": sorted(cells, key=lambda c: c["val_nll"])}
    return out


def resume_summary() -> dict:
    """Which tuned-ladder cells resumed from a checkpoint, from each cell's own `resumed_from_step` (absent in cells that predate it)."""
    resumed, absent = {}, []
    for p in sorted(ATTRIB.glob("LT_*.json")):
        j = json.loads(p.read_text())
        if "resumed_from_step" not in j:
            absent.append(p.stem)
        elif j["resumed_from_step"]:
            resumed[p.stem] = j["resumed_from_step"]
    return {"n_resumed": len(resumed), "resumed": resumed, "field_absent": absent}


def checksums() -> dict:
    """SHA-256 of the four built files vs the manifest. Skipped (empty dict) when the built corpus or the manifest is not on this
    machine: the corpus is not part of the public artifact."""
    manifests = [ROOT / "papers" / "p13-salad-v2" / "ARTIFACT.md", ROOT.parent / "ARTIFACT.md"]
    manifest = next((m.read_text() for m in manifests if m.exists()), None)
    files = {f"{arm}/{split}": ROOT / "data" / f"salad2_{arm}" / f"salad2_{split}.json" for arm in ARMS for split in ("train", "test")}
    if manifest is None or not all(p.exists() for p in files.values()):
        return {}
    return {k: {"sha256": (h := hashlib.sha256(p.read_bytes()).hexdigest()), "in_manifest": h in manifest} for k, p in files.items()}


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:                      # e.g. a ladder report redirected to a temporary directory
        return str(p)


def build() -> dict:
    return {"torch_ladder": torch_ladder(), "tuning_grid": tuning_grid(), "resume": resume_summary(), "mlx_ladder": mlx_ladder(), "checksums": checksums(),
            "sources": {"torch": _rel(LADDER), "mlx": "results/salad2/full_test/*_full.json",
                        "same_rows": _rel(SAME_ROWS)}}


def render(fs: dict) -> str:
    L = ["# P13 fact sheet (generated by scripts/p13_fact_sheet.py; do not edit by hand)", ""]
    t = fs["torch_ladder"]
    L += [f"## Torch tuned ladder: {t['n_cells']} of {t['n_expected']} designed cells, bootstrap B={t['B']}", ""]
    L += ["| group | n | per-seed F1 | mean | SD | DT (LLM/base/ns) | SVC | BERT | Holm DT | Holm SVC | Holm BERT |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, g in t["groups"].items():
        f = lambda b, w: "/".join(str(g["vs"][b][w][x]) for x in ("llm", "baseline", "neither"))  # noqa: E731
        L.append(f"| {k} | {g['n']} | {', '.join(f'{x:.4f}' for x in g['f1'])} | {g['mean']:.4f} | {g['sd']:.4f} | "
                 f"{f('dt', 'unadjusted')} | {f('svc', 'unadjusted')} | {f('bert', 'unadjusted')} | "
                 f"{f('dt', 'holm')} | {f('svc', 'holm')} | {f('bert', 'holm')} |")
    L += ["", "Baselines: " + "; ".join(f"{a}: " + ", ".join(f"{k} {v['f1']:.4f}" for k, v in b.items()) for a, b in t["baselines"].items()), ""]
    L += ["## Recipe per arm", ""] + [f"- {a}: {json.dumps(r)}" for a, r in t["recipe"].items()]
    L += ["", "## Execution per arm/size", ""] + [f"- {k}: {json.dumps(v)}" for k, v in t["execution"].items()]
    L += ["", "## Tuning grid (LoRA, one run per point)", ""]
    for a, g in fs["tuning_grid"].items():
        L.append(f"- {a}: {g['n_entries']} entries / {g['n_distinct_configs']} distinct configs; winner {g['winner']} (NLL {g['winner_nll']}); "
                 f"max epochs in grid {g['max_epochs_in_grid']}; epochs at which a higher lr was tried: {g['epochs_with_higher_lr_than_winner']}; "
                 f"repeated: {json.dumps(g['repeated_configs'])}")
    r = fs["resume"]
    L += ["", f"## Resumes: {r['n_resumed']} cells resumed {json.dumps(r['resumed'])}; field absent (predates it): {r['field_absent']}"]
    L += ["", "## MLX ladder (full 10,000-row test) and same-rows baselines", ""]
    for k, g in fs["mlx_ladder"]["groups"].items():
        L.append(f"- {k}: n={g['n']} per-seed {g['f1']} mean {g['mean']} sd {g['sd']}")
    L += ["", "Same-rows totals: " + json.dumps(fs["mlx_ladder"].get("arm_totals")), ""]
    L += ["## Data-file checksums vs ARTIFACT.md", ""] + ([f"- {k}: in manifest = {v['in_manifest']}" for k, v in fs["checksums"].items()]
                                                           or ["- skipped: the built corpus (or the manifest) is not on this machine"])
    return "\n".join(L) + "\n"


def numbers_in_fact_sheet(fs: dict) -> set[str]:
    nums: set[str] = set()

    def walk(x):
        if isinstance(x, float):
            nums.add(f"{x:.4f}")
            nums.add(f"{abs(x):.4f}")                      # the manuscript prints signed differences as $-$0.0052
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(fs)
    return nums


def check(tex_path: Path, fs: dict) -> int:
    """Every `d.dddd` token in the LLM-ladder tables / results text of the manuscript must be a number of the fact sheet
    or one of the older, separately sourced numbers listed in LEGACY (classical/BERT/gate statistics and follow-ups)."""
    tex = tex_path.read_text()
    legacy_path = ROOT / "papers" / "p13-salad-v2" / "legacy_numbers.txt"
    legacy = set(legacy_path.read_text().split()) if legacy_path.exists() else set()
    known = numbers_in_fact_sheet(fs) | legacy
    bad = sorted({m for m in re.findall(r"(?<![\d.])(\d\.\d{4})(?!\d)", tex) if m not in known})
    for m in bad:
        print(f"NOT IN FACT SHEET: {m}")
    print(f"checked {len(set(re.findall(r'(?<![\d.])(\d\.\d{4})(?!\d)', tex)))} distinct 4-dp numbers, {len(bad)} unmatched")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="TEX")
    a = ap.parse_args()
    fs = build()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "p13_fact_sheet.json").write_text(json.dumps(fs, indent=1))
    (ANALYSIS / "p13_fact_sheet.md").write_text(render(fs))
    print(f"wrote {ANALYSIS / 'p13_fact_sheet.md'}")
    if fs["checksums"] and not all(v["in_manifest"] for v in fs["checksums"].values()):
        print("WARNING: a data file's SHA-256 is not in ARTIFACT.md")
    return check(Path(a.check), fs) if a.check else 0


if __name__ == "__main__":
    sys.exit(main())
