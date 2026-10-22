"""dgx_analysis.common -- shared, numpy-only foundation for the DGX (attrib.py) result analyses.

PUBLIC API (stable -- other modules build on these signatures; extend, do not change)
====================================================================================

Paths / constants
    REPO, SALAD2, ATTRIB_DIR, P22_ATTRIB_DIR, ANALYSIS_DIR, DATA_DIR      pathlib.Path
    ARMS = ("unsw", "cicids");  MODELS = {"0.5b": "Qwen/Qwen2.5-0.5B-Instruct", ...}
    BATCH = 4;  T_STEPS_PER_EPOCH = 1125      (attrib.py: 4500 train rows // BATCH when val=500)
    VERIFY_TOL = 6e-4                         (tolerance used to compare a 4-dp JSON number with a recomputed one)

Metrics (identical to the runner's sklearn call f1_score(gold, pred, labels=sorted(set(gold)), average="macro",
zero_division=0); predictions outside `labels` are ignored for the FP counts, exactly as in sklearn)
    gold_labels(gold) -> list[str]                          sorted(set(gold))
    macro_f1(gold, pred, labels=None) -> float              mean of per-class F1 over `labels` (default: gold labels)
    per_class_f1(gold, pred, labels=None) -> dict[str,float]
    off_vocab_rate(gold, pred, labels=None) -> float        fraction of pred not in `labels`
    encode(gold, pred, labels) -> (gi, pi)                  int arrays, -1 == not in labels
    macro_f1_codes(gi, pi, K) -> float                      macro-F1 on encoded arrays (fast path, used by bootstraps)

Statistics helpers
    mean(xs) -> float;  sd(xs) -> float (sample SD, ddof=1; nan when n<2);  sem(xs) -> float
    mean_sd(xs) -> (mean, sd, n);  fmt_mean_sd(xs, nd=4) -> "0.4327 +- 0.0123 (n=3)"
    mannwhitney_min_p(n1, n2) -> float      smallest two-sided EXACT Mann-Whitney p (all-ordered): 2/C(n1+n2,n1)
    sign_test_min_p(n) -> float             smallest two-sided EXACT paired sign-test p: min(1, 2/2**n)
    holm_adjust(pvals) -> list[float]       Holm step-down adjusted p-values (same order as input; a NaN p counts toward
                                            the family size, sorts last and returns NaN)
    group_by(items, key) -> dict            key = callable | attribute name | dict key; insertion ordered

Cell parsing
    parse_cell_name(name) -> dict           {'name','family','arm','model','method','rank','seed','steps','lr',
                                             'scale','layers','epochs','val','n_train','tags', ...} (missing = absent)
    FAMILIES                                {'T','F','M','LT','P','O','R','D','S','L','X'}

Cells (JSON + preds)
    Cell                                    wraps one NAME.json (+ NAME.preds.jsonl); read-only
        .name .family .path .preds_path .meta .config .parsed .source_dir .legacy(bool: p22_attrib schema)
        .arm .model(str key like '0.5b') .model_id .method .rank .alpha .lr .steps .epochs .seed .layers
        .val(int) .eval_limit .eval_bs .scored_on('test'|'val'|None) .n_scored .off_vocab .val_nll .per_class
        .tuned (str|None) .tuned_resolved (dict|None; read from config OR top level)
        .f1_reported (JSON, 4 dp)  .f1 (recomputed from preds at full precision when preds exist, else reported)
        .is_full_test (scored on the full 10,000-row test set)  .has_preds  .preds_complete (rows == n_scored)
        .iter_preds() -> iterator of dict rows (uncached; keys k|idx, gold, pred, raw)
        .preds() -> list[dict] (cached)  .gold_pred() -> (gold, pred) lists (cached)  .row_ids() -> list[int]
        .release() drops caches
    load_cells(dirs=None, families=None, arm=None, predicate=None) -> list[Cell]   sorted by name; default dirs =
                                            [ATTRIB_DIR, P22_ATTRIB_DIR]; unreadable files go to LOAD_ERRORS
    load_cell(name, dirs=None) -> Cell | None
    scan_cell_files(dirs=None) -> list[(name, json_path)]           every NAME.json (no parsing)
    LOAD_ERRORS: list[(path, message)]      files that could not be parsed by load_cells (verify reports them)

Baselines / test rows (all on the SAME 10,000 test rows, in test-file order)
    test_gold(arm) -> list[str]                                   gold of the arm's test set
    arm_labels(arm) -> list[str]                                  sorted(set(test_gold(arm)))
    baseline_preds(arm, which) -> list[str]                       which: dt | svc | majority | bert
    canonical_baselines(arm) -> {'dt': {...}, 'svc': {...}, 'majority': {...}}     reported + recomputed macro-F1
    bert_baseline(arm) -> {'f1_reported', 'f1', 'checkpoint', 'preds', ...}         validation-selected REFIT
    matched_baselines(arm=None) -> list[dict]                     p6_matched_baselines.json 'cells' (optionally filtered)
    system_preds(arm, spec) -> (gold, pred)                       spec = dt|svc|majority|bert|<full-test cell name>
    parse_label(text) -> str                                      the runner's label parser
    split_gold(arm, split) -> list[str]                           gold parsed from data/salad2_<arm>/salad2_<split>.json
    val_holdout_gold(arm, n_val) -> list[str]                     gold of the runner's Random(1234) validation hold-out
    epochs_from_steps(steps) -> int                               round(steps / T_STEPS_PER_EPOCH) (runner's tuned rule)

Reports
    fmt(x, nd=4, signed=False) -> str;  md_table(headers, rows, align=None, nd=4) -> str
    to_jsonable(obj) -> obj                 numpy -> python, nan/inf -> None
    save_report(name, md, data=None, out_dir=None) -> (md_path, json_path)   writes <name>.md and <name>.json under
                                            results/salad2/dgx/analysis/ (name must be a bare file stem); the env var
                                            DGX_ANALYSIS_OUT redirects ANALYSIS_DIR (tests, dry runs)

Module protocol for the CLI (see cli.py): a subcommand module defines
    register(subparsers)  -> adds its subparser and calls parser.set_defaults(func=run)
    run(args) -> int      exit code (0 ok)
    run_default() -> int  optional: what `python3 -m dgx_analysis all` runs

Conventions: macro-F1 over gold-present labels; sample SD (ddof=1); never impute missing data; never write outside
scripts/dgx_analysis/ and results/salad2/dgx/analysis/.
"""
from __future__ import annotations

import functools
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

__all__ = [
    "REPO", "SALAD2", "ATTRIB_DIR", "P22_ATTRIB_DIR", "ANALYSIS_DIR", "DATA_DIR", "ARMS", "MODELS", "BATCH",
    "T_STEPS_PER_EPOCH", "VERIFY_TOL", "FAMILIES", "LOAD_ERRORS", "AnalysisDataError",
    "gold_labels", "macro_f1", "per_class_f1", "off_vocab_rate", "encode", "macro_f1_codes",
    "mean", "sd", "sem", "mean_sd", "fmt_mean_sd", "mannwhitney_min_p", "sign_test_min_p", "holm_adjust", "group_by",
    "parse_cell_name", "Cell", "load_cells", "load_cell", "scan_cell_files", "clear_cache",
    "test_gold", "arm_labels", "baseline_preds", "canonical_baselines", "bert_baseline", "matched_baselines",
    "system_preds", "parse_label", "split_gold", "val_holdout_gold", "epochs_from_steps",
    "fmt", "md_table", "to_jsonable", "save_report", "read_json",
]

# --------------------------------------------------------------------------------------------------------------
# paths / constants
# --------------------------------------------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PKG_DIR.parent
REPO = SCRIPTS_DIR.parent
SALAD2 = REPO / "results" / "salad2"
ATTRIB_DIR = SALAD2 / "dgx" / "attrib"
P22_ATTRIB_DIR = SALAD2 / "dgx" / "p22_attrib"
# Reports go here; the env var lets tests / dry runs redirect them without touching the real report directory.
ANALYSIS_DIR = Path(os.environ["DGX_ANALYSIS_OUT"]) if os.environ.get("DGX_ANALYSIS_OUT") else SALAD2 / "dgx" / "analysis"
DATA_DIR = REPO / "data"

ARMS = ("unsw", "cicids")
MODELS = {"0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
          "3b": "Qwen/Qwen2.5-3B-Instruct", "7b": "Qwen/Qwen2.5-7B-Instruct"}
_MODEL_KEY_BY_ID = {v: k for k, v in MODELS.items()}
BATCH = 4
T_STEPS_PER_EPOCH = 4500 // BATCH  # 1125, attrib.py resolve_tuned
VERIFY_TOL = 6e-4
FAMILIES = {"T", "F", "M", "LT", "P", "O", "R", "D", "S", "L", "X"}
_LAYER_FROM_TOKEN = {"all": "all", "t8": "top8"}
_BASELINE_ALIASES = {"dt": "decision_tree", "decision_tree": "decision_tree", "svc": "linear_svc",
                     "linear_svc": "linear_svc", "svm": "linear_svc", "majority": "majority_class_floor",
                     "majority_class_floor": "majority_class_floor", "bert": "bert"}

LOAD_ERRORS: list[tuple[str, str]] = []


class AnalysisDataError(RuntimeError):
    """Raised for missing/inconsistent inputs a caller cannot sensibly continue without."""


# --------------------------------------------------------------------------------------------------------------
# json cache
# --------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=512)
def _read_json_cached(path: str, mtime_ns: int, size: int) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_json(path: str | Path) -> Any:
    """Cached json.load, keyed on (path, mtime, size) like the preds caches so an edited or re-synced file is re-read
    (a path-only key served stale metadata after an in-place edit). The returned object is shared -- treat it as read-only."""
    p = Path(path)
    st = p.stat()   # FileNotFoundError like open() when the file is absent
    return _read_json_cached(str(p), st.st_mtime_ns, st.st_size)


def clear_cache() -> None:
    """Drop the JSON, prediction and test/baseline caches (call after files change on disk)."""
    _read_json_cached.cache_clear()
    _split_gold_cached.cache_clear()
    _canonical_cached.cache_clear()
    _load_preds_cached.cache_clear()
    _load_gold_pred_cached.cache_clear()


def _stat_key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return str(path), st.st_mtime_ns, st.st_size


@functools.lru_cache(maxsize=8)
def _load_preds_cached(path: str, mtime_ns: int, size: int) -> tuple[dict, ...]:
    with open(path, "r", encoding="utf-8") as fh:
        return tuple(json.loads(line) for line in fh if line.strip())


@functools.lru_cache(maxsize=96)
def _load_gold_pred_cached(path: str, mtime_ns: int, size: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    g, p = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                g.append(r["gold"])
                p.append(r["pred"])
    return tuple(g), tuple(p)


# --------------------------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------------------------
def gold_labels(gold: Iterable[str]) -> list[str]:
    """Label set of a gold sequence, exactly as the runner builds it: sorted(set(gold))."""
    return sorted(set(gold))


def encode(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Integer codes 0..K-1 for gold and pred; anything not in `labels` is -1."""
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred length mismatch: {len(gold)} vs {len(pred)}")
    idx = {lab: i for i, lab in enumerate(labels)}
    gi = np.fromiter((idx.get(g, -1) for g in gold), dtype=np.int64, count=len(gold))
    pi = np.fromiter((idx.get(p, -1) for p in pred), dtype=np.int64, count=len(pred))
    return gi, pi


def _f1_from_codes(gi: np.ndarray, pi: np.ndarray, K: int) -> np.ndarray:
    true_sum = np.bincount(gi[gi >= 0], minlength=K).astype(np.float64)
    pred_sum = np.bincount(pi[pi >= 0], minlength=K).astype(np.float64)
    tp = np.bincount(gi[(gi == pi) & (gi >= 0)], minlength=K).astype(np.float64)
    den = true_sum + pred_sum
    f1 = np.zeros(K, dtype=np.float64)
    ok = den > 0
    f1[ok] = 2.0 * tp[ok] / den[ok]           # zero_division=0 where a class has no gold and no predictions
    return f1


def macro_f1_codes(gi: np.ndarray, pi: np.ndarray, K: int) -> float:
    """Macro-F1 over K classes from encoded arrays (see encode). F1_c = 2TP/(gold_c + pred_c), 0 if that is 0."""
    if K <= 0:
        return float("nan")
    return float(_f1_from_codes(gi, pi, K).mean())


def per_class_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None) -> dict[str, float]:
    labels = gold_labels(gold) if labels is None else list(labels)
    gi, pi = encode(gold, pred, labels)
    f1 = _f1_from_codes(gi, pi, len(labels))
    return {lab: float(f1[i]) for i, lab in enumerate(labels)}


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None) -> float:
    """Strict macro-F1, identical to sklearn f1_score(gold, pred, labels=labels or sorted(set(gold)),
    average='macro', zero_division=0)."""
    labels = gold_labels(gold) if labels is None else list(labels)
    gi, pi = encode(gold, pred, labels)
    return macro_f1_codes(gi, pi, len(labels))


def off_vocab_rate(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None) -> float:
    """Fraction of predictions whose label is not in `labels` (default: the gold label set)."""
    labs = set(gold_labels(gold) if labels is None else labels)
    n = len(pred)
    return float(sum(1 for p in pred if p not in labs) / n) if n else float("nan")


# --------------------------------------------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------------------------------------------
def mean(xs: Iterable[float]) -> float:
    a = [float(x) for x in xs]
    return float(sum(a) / len(a)) if a else float("nan")


def sd(xs: Iterable[float]) -> float:
    """Sample standard deviation (ddof=1); nan when fewer than two values."""
    a = np.asarray([float(x) for x in xs], dtype=np.float64)
    return float(a.std(ddof=1)) if a.size > 1 else float("nan")


def sem(xs: Iterable[float]) -> float:
    a = [float(x) for x in xs]
    return sd(a) / math.sqrt(len(a)) if len(a) > 1 else float("nan")


def mean_sd(xs: Iterable[float]) -> tuple[float, float, int]:
    a = [float(x) for x in xs]
    return mean(a), sd(a), len(a)


def fmt_mean_sd(xs: Iterable[float], nd: int = 4) -> str:
    m, s, n = mean_sd(xs)
    if n == 0:
        return "n/a (n=0)"
    if n == 1:
        return f"{m:.{nd}f} (n=1, no SD)"
    return f"{m:.{nd}f} +- {s:.{nd}f} (n={n})"


def mannwhitney_min_p(n1: int, n2: int) -> float:
    """Smallest attainable two-sided exact Mann-Whitney U p-value: 2 / C(n1+n2, n1) (n=3 vs 3 -> 0.10)."""
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return min(1.0, 2.0 / math.comb(n1 + n2, n1))


def sign_test_min_p(n: int) -> float:
    """Smallest attainable two-sided exact paired sign-test p-value: 2 / 2**n (n=3 pairs -> 0.25)."""
    if n <= 0:
        return float("nan")
    return min(1.0, 2.0 / 2 ** n)


def holm_adjust(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, returned in the input order (monotone, capped at 1).

    A NaN p-value (an untestable comparison) still counts toward the family size m -- dropping it would shrink the
    correction, the anti-conservative direction -- is ordered LAST so it cannot disturb the other tests' ranks (NaN
    comparisons make a plain sort order arbitrary), and comes back as NaN."""
    m = len(pvals)
    order = sorted((i for i in range(m) if pvals[i] == pvals[i]), key=lambda i: pvals[i])
    adj = [float("nan")] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def group_by(items: Iterable[Any], key: Callable[[Any], Any] | str) -> dict[Any, list[Any]]:
    """Group into an insertion-ordered dict. `key` may be a callable, an attribute name, or a dict key."""
    if isinstance(key, str):
        name = key

        def kf(x: Any) -> Any:
            if isinstance(x, dict):
                return x.get(name)
            return getattr(x, name)
    else:
        kf = key
    out: dict[Any, list[Any]] = {}
    for it in items:
        out.setdefault(kf(it), []).append(it)
    return out


# --------------------------------------------------------------------------------------------------------------
# cell-name parsing
# --------------------------------------------------------------------------------------------------------------
_RE_SIZE = re.compile(r"(\d+(?:\.\d+)?)b")
_RE_NUM = r"[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?"


def parse_cell_name(name: str) -> dict:
    """Best-effort token parse of a cell name. The JSON `config` stays authoritative; verify cross-checks the two.

    Families: T_<arm>_<method>_r<rank>_lr<lr>_e<epochs>_v<val>; F_S<steps>_L<lr>_C<scale>_Y<all|t8>;
    M_<F-style tokens>_maskpad_s<seed>; LT_<arm>_<size>_lora_r16_s<seed>; P_<arm>_<size>_<lora|oft>_s<seed>;
    R_<arm>_r<rank>_s<seed>; S_<arm>_n<N>_s<seed>; X0_mlxemu_r8_s42 / X1_t600_r8_s42; O_, D_, L_ (free-form).
    A cross-arm eval cell ends in `_to_<arm>` (XA_LT_cicids_0.5b_lora_r16_s42_to_unsw): the arm before it is the arm the
    adapter was TRAINED on ('arm'), the one after `to` is the arm it was SCORED on ('eval_arm').
    Unrecognised tokens land in 'tags'."""
    tokens = name.split("_")
    head = tokens[0]
    fam_m = re.fullmatch(r"([A-Z]+)(\d*)", head)
    family = fam_m.group(1) if fam_m and fam_m.group(1) in FAMILIES else "?"
    out: dict[str, Any] = {"name": name, "family": family, "tags": []}
    if family == "X" and fam_m and fam_m.group(2):
        out["variant_no"] = int(fam_m.group(2))
    rest = tokens[1:] if family != "?" else tokens
    skip_next = False
    for i, t in enumerate(rest):
        if skip_next:
            skip_next = False
            continue
        m: re.Match | None
        if t == "to" and i + 1 < len(rest) and rest[i + 1] in ARMS:
            out["eval_arm"] = rest[i + 1]
            skip_next = True
        elif t in ARMS:
            out["arm"] = t
        elif _RE_SIZE.fullmatch(t):
            out["model"] = t
        elif t in ("lora", "oft"):
            out["method"] = t
        elif (m := re.fullmatch(r"r(\d+)", t)):
            out["rank"] = int(m.group(1))
        elif (m := re.fullmatch(r"s(\d+)", t)):
            out["seed"] = int(m.group(1))
        elif (m := re.fullmatch(r"S(\d+)", t)):
            out["steps"] = int(m.group(1))
        elif (m := re.fullmatch(r"n(\d+)", t)):
            out["n_train"] = int(m.group(1))
        elif (m := re.fullmatch(r"e(\d+)", t)):
            out["epochs"] = int(m.group(1))
        elif (m := re.fullmatch(r"v(\d+)", t)):
            out["val"] = int(m.group(1))
        elif (m := re.fullmatch(r"lr(" + _RE_NUM + ")", t)):
            out["lr"] = float(m.group(1))
        elif (m := re.fullmatch(r"L(" + _RE_NUM + ")", t)):
            out["lr"] = float(m.group(1))
        elif (m := re.fullmatch(r"C(" + _RE_NUM + ")", t)):
            out["scale"] = float(m.group(1))
        elif (m := re.fullmatch(r"Y(all|t8)", t)):
            out["layers"] = _LAYER_FROM_TOKEN[m.group(1)]
        elif family == "X" and (m := re.fullmatch(r"t(\d+)", t)):
            out["steps"] = int(m.group(1))
            out["tags"].append(t)
        else:
            out["tags"].append(t)
    return out


def epochs_from_steps(steps: int | float) -> int:
    """attrib.py resolve_tuned: epochs = round(steps / (4500 // BATCH)) = round(steps / 1125)."""
    return int(round(float(steps) / T_STEPS_PER_EPOCH))


# --------------------------------------------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------------------------------------------
_CORE_KEYS = {"name", "config", "model_id", "trainable_params", "strict_macro_f1", "per_class_f1", "off_vocab_rate",
              "n_scored", "scored_on", "val_answer_nll", "train_seconds", "eval_seconds", "loss_first50",
              "loss_last50", "device", "torch", "host"}


class Cell:
    """One experiment cell: NAME.json (+ NAME.preds.jsonl). Handles both the attrib.py schema and the legacy
    p22_attrib.py schema (top-level arm/model/dtype/n_test, preds keyed by 'idx')."""

    def __init__(self, name: str, json_path: Path, meta: dict, source_dir: Path | None = None):
        self.name = name
        self.path = Path(json_path)
        self.source_dir = Path(source_dir) if source_dir else self.path.parent
        self.meta = meta
        self.preds_path = self.path.with_name(name + ".preds.jsonl")
        self.parsed = parse_cell_name(name)
        self.family = self.parsed["family"]
        self._preds: list[dict] | None = None
        self._gp: tuple[list[str], list[str]] | None = None
        self._f1: float | None = None

    def __repr__(self) -> str:
        return f"Cell({self.name!r}, f1={self.f1_reported}, on={self.scored_on}, n={self.n_scored})"

    # ---- metadata (config authoritative, name parse / legacy fallbacks) ----
    @property
    def config(self) -> dict:
        c = self.meta.get("config")
        return c if isinstance(c, dict) else {}

    @property
    def legacy(self) -> bool:
        return "n_test" in self.meta and "n_scored" not in self.meta

    def _cfg(self, key: str, default: Any = None) -> Any:
        v = self.config.get(key, None)
        return default if v is None else v

    @property
    def arm(self) -> str | None:
        return self._cfg("arm") or self.meta.get("arm") or self.parsed.get("arm")

    @property
    def scored_arm(self) -> str | None:
        """The arm whose test set this cell was scored on. Equal to `arm` except for an eval-only cross-arm cell, whose
        adapter was trained on `arm` but scored on config.eval_arm. Only an eval-only cell may name a different arm."""
        if self.meta.get("eval_only") is True and self._cfg("eval_arm") in ARMS:
            return self._cfg("eval_arm")
        return self.arm

    @property
    def model_id(self) -> str | None:
        mid = self.meta.get("model_id")
        if mid:
            return mid
        m = self.meta.get("model")
        return m if isinstance(m, str) and "/" in m else MODELS.get(self._cfg("model", ""), None)

    @property
    def model(self) -> str | None:
        k = self._cfg("model")
        if k in MODELS:
            return k
        return _MODEL_KEY_BY_ID.get(self.model_id or "", self.parsed.get("model"))

    @property
    def method(self) -> str | None:
        return self._cfg("method") or ("lora" if self.legacy else None) or self.parsed.get("method")

    @property
    def rank(self) -> int | None:
        v = self._cfg("rank")
        return int(v) if v is not None else self.parsed.get("rank")

    @property
    def alpha(self) -> float | None:
        v = self._cfg("alpha")
        return None if v is None else float(v)

    @property
    def lr(self) -> float | None:
        v = self._cfg("lr")
        return float(v) if v is not None else self.parsed.get("lr")

    @property
    def steps(self) -> int:
        return int(self._cfg("steps", 0) or 0)

    @property
    def epochs(self) -> int:
        """Configured epochs, else steps / ((n_train - val) // 4) with n_train from an S_ name (default 5000); 0 if unknown. NB the runner's tuning rule
        (epochs_from_steps) is round(steps / 1125), valid for the val=500 tuning cells."""
        e = int(self._cfg("epochs", 0) or 0)
        if e:
            return e
        if self.steps:  # train rows = parsed n_train (S_ sweep cells) else the standard 5,000, minus the val hold-out
            n_rows = int(self.parsed.get("n_train") or 5000) - self.val
            return int(round(self.steps / max(n_rows // BATCH, 1)))
        return int(self.parsed.get("epochs", 0) or 0)

    @property
    def seed(self) -> int | None:
        v = self._cfg("seed")
        return int(v) if v is not None else self.parsed.get("seed")

    @property
    def layers(self) -> str | None:
        return self._cfg("layers") or self.parsed.get("layers")

    @property
    def val(self) -> int:
        return int(self._cfg("val", 0) or 0)

    @property
    def eval_limit(self) -> int:
        return int(self._cfg("eval_limit", 0) or 0)

    @property
    def eval_bs(self) -> int:
        return int(self.meta.get("eval_bs") or self._cfg("eval_bs", 1) or 1)

    @property
    def scored_on(self) -> str | None:
        s = self.meta.get("scored_on")
        if s:
            return s
        return "test" if "n_test" in self.meta else None

    @property
    def n_scored(self) -> int | None:
        n = self.meta.get("n_scored", self.meta.get("n_test"))
        return None if n is None else int(n)

    @property
    def f1_reported(self) -> float | None:
        v = self.meta.get("strict_macro_f1")
        return None if v is None else float(v)

    @property
    def off_vocab(self) -> float | None:
        v = self.meta.get("off_vocab_rate")
        return None if v is None else float(v)

    @property
    def val_nll(self) -> float | None:
        v = self.meta.get("val_answer_nll")
        return None if v is None else float(v)

    @property
    def per_class(self) -> dict:
        pc = self.meta.get("per_class_f1")
        return pc if isinstance(pc, dict) else {}

    @property
    def tuned(self) -> str | None:
        """The `tuned="arm:method"` key the cell was launched with (config first, then top level)."""
        return self.config.get("tuned") or self.meta.get("tuned") or None

    @property
    def tuned_resolved(self) -> dict | None:
        """What the runner adopted from the tuning grid at run time. attrib.py stores it inside `config`; a top-level
        copy is accepted too, so either result-JSON layout works."""
        return self.meta.get("tuned_resolved") or self.config.get("tuned_resolved") or None

    @property
    def is_full_test(self) -> bool:
        return self.scored_on == "test" and self.eval_limit == 0 and self.n_scored == 10000

    @property
    def has_preds(self) -> bool:
        return self.preds_path.is_file()

    @property
    def preds_complete(self) -> bool:
        """True when the preds file exists and holds exactly `n_scored` rows. `f1` is recomputed from whatever rows
        exist, so an analysis that ranks or contrasts cells should skip (and name) cells where this is False; `verify`
        reports the same defect as a FAIL."""
        if not self.has_preds or self.n_scored is None:
            return False
        try:
            return len(self.gold_pred()[0]) == self.n_scored
        except (OSError, ValueError, KeyError):
            return False

    # ---- predictions ----
    def iter_preds(self) -> Iterator[dict]:
        """Stream the preds.jsonl rows (dict with k|idx, gold, pred, raw). Uncached, low memory."""
        if not self.has_preds:
            return
        with open(self.preds_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def preds(self) -> list[dict]:
        """All rows (with `raw`), cached across Cell objects by (path, mtime, size) in a small LRU (8 files)."""
        if self._preds is None:
            self._preds = list(_load_preds_cached(*_stat_key(self.preds_path))) if self.has_preds else []
        return self._preds

    def gold_pred(self) -> tuple[list[str], list[str]]:
        """(gold, pred) label lists; cached across Cell objects by (path, mtime, size) (96 files). Do not mutate."""
        if self._gp is None:
            if self.has_preds:
                g, p = _load_gold_pred_cached(*_stat_key(self.preds_path))
                self._gp = (list(g), list(p))
            else:
                self._gp = ([], [])
        return self._gp

    def row_ids(self) -> list[int | None]:
        return [r.get("k", r.get("idx")) for r in self.preds()]

    def release(self) -> None:
        """Drop this object's references (the shared LRU caches may still hold the data)."""
        self._preds = None
        self._gp = None

    @property
    def f1(self) -> float | None:
        """Macro-F1 recomputed from the predictions (full precision) when they exist, else the reported 4-dp value."""
        if self._f1 is None and self.has_preds:
            g, p = self.gold_pred()
            if g:
                self._f1 = macro_f1(g, p)
        return self._f1 if self._f1 is not None else self.f1_reported


def scan_cell_files(dirs: Sequence[str | Path] | None = None) -> list[tuple[str, Path]]:
    """Every NAME.json under `dirs` (no parsing): [(name, path)], sorted by (name, path)."""
    dirs = [ATTRIB_DIR, P22_ATTRIB_DIR] if dirs is None else [Path(d) for d in dirs]
    out = []
    for d in dirs:
        if not Path(d).is_dir():
            continue
        for p in sorted(Path(d).glob("*.json")):
            out.append((p.stem, p))
    return sorted(out, key=lambda t: (t[0], str(t[1])))


def load_cells(dirs: Sequence[str | Path] | None = None, families: Iterable[str] | None = None,
               arm: str | None = None, predicate: Callable[[Cell], bool] | None = None) -> list[Cell]:
    """Load cells sorted by name. Duplicate names across dirs keep the first directory's copy (verify flags them).
    Unreadable/invalid JSON files are skipped and recorded in LOAD_ERRORS."""
    fams = set(families) if families is not None else None
    seen: set[str] = set()
    cells: list[Cell] = []
    dirs_l = [ATTRIB_DIR, P22_ATTRIB_DIR] if dirs is None else [Path(d) for d in dirs]
    for d in dirs_l:
        if not Path(d).is_dir():
            continue
        for p in sorted(Path(d).glob("*.json")):
            name = p.stem
            if name in seen:
                continue
            try:
                meta = read_json(p)
                if not isinstance(meta, dict):
                    raise ValueError("top-level JSON is not an object")
            except Exception as e:  # noqa: BLE001 - never crash the analysis on one bad file
                err = (str(p), f"{type(e).__name__}: {e}")
                if err not in LOAD_ERRORS:  # load_cells is called repeatedly; record each failure once
                    LOAD_ERRORS.append(err)
                continue
            seen.add(name)
            c = Cell(name, p, meta, d)
            if fams is not None and c.family not in fams:
                continue
            if arm is not None and c.arm != arm:
                continue
            if predicate is not None and not predicate(c):
                continue
            cells.append(c)
    return sorted(cells, key=lambda c: c.name)


def load_cell(name: str, dirs: Sequence[str | Path] | None = None) -> Cell | None:
    for c in load_cells(dirs):
        if c.name == name:
            return c
    return None


# --------------------------------------------------------------------------------------------------------------
# data files / baselines
# --------------------------------------------------------------------------------------------------------------
def parse_label(text: str) -> str:
    """attrib.py parse_label, verbatim behaviour."""
    m = re.search(r"Attack Category:\s*([A-Za-z][A-Za-z0-9_\- ]*)", text)
    return m.group(1).strip() if m else text.strip().split("\n")[0].strip()


@functools.lru_cache(maxsize=8)
def _split_gold_cached(arm: str, split: str) -> tuple[str, ...]:
    p = DATA_DIR / f"salad2_{arm}" / f"salad2_{split}.json"
    if not p.is_file():
        raise AnalysisDataError(f"missing data file {p}")
    with open(p, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    return tuple(parse_label({x["from"]: x["value"] for x in r["conversations"]}.get("gpt", "")) for r in rows)


def split_gold(arm: str, split: str) -> list[str]:
    """Gold labels parsed from data/salad2_<arm>/salad2_<split>.json (split = 'train' | 'test'), file order."""
    return list(_split_gold_cached(arm, split))


def val_holdout_gold(arm: str, n_val: int) -> list[str]:
    """Gold of the validation rows attrib.py holds out of TRAIN when val=n_val: indices
    Random(1234).sample(range(len(train)), n_val), kept in file order. Never touches the test set."""
    gold = split_gold(arm, "train")
    hold = set(random.Random(1234).sample(range(len(gold)), n_val))
    return [g for i, g in enumerate(gold) if i in hold]


@functools.lru_cache(maxsize=8)
def _canonical_cached(arm: str) -> dict:
    p = SALAD2 / f"{arm}_baseline_preds.json"
    if not p.is_file():
        raise AnalysisDataError(f"missing {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_gold(arm: str) -> list[str]:
    """Gold of the arm's 10,000-row test set (from <arm>_baseline_preds.json; equals data test-file order)."""
    return list(_canonical_cached(arm)["gold"])


def arm_labels(arm: str) -> list[str]:
    return gold_labels(test_gold(arm))


def _bert_json(arm: str) -> dict:
    p = SALAD2 / f"{arm}_bert_preds.json"
    if not p.is_file():
        raise AnalysisDataError(f"missing {p}")
    return read_json(p)


def baseline_preds(arm: str, which: str) -> list[str]:
    """Predictions of a baseline on the arm's 10,000 test rows. which: dt | svc | majority | bert."""
    key = _BASELINE_ALIASES.get(which.lower())
    if key is None:
        raise AnalysisDataError(f"unknown baseline {which!r} (use dt|svc|majority|bert)")
    if key == "bert":
        return list(_bert_json(arm)["bert"])
    return list(_canonical_cached(arm)[key])


def canonical_baselines(arm: str) -> dict:
    """{'dt'|'svc'|'majority': {'key','f1_reported','f1','n'}}: reported score from <arm>_baselines.json and the
    macro-F1 recomputed here from the stored predictions."""
    rep_p = SALAD2 / f"{arm}_baselines.json"
    rep = read_json(rep_p) if rep_p.is_file() else {}
    gold = test_gold(arm)
    labels = gold_labels(gold)
    out = {}
    for short, key in (("dt", "decision_tree"), ("svc", "linear_svc"), ("majority", "majority_class_floor")):
        preds = _canonical_cached(arm).get(key)
        out[short] = {"key": key, "f1_reported": rep.get(key),
                      "f1": macro_f1(gold, preds, labels) if preds else None, "n": len(gold)}
    return out


def bert_baseline(arm: str) -> dict:
    """Validation-selected REFIT BERT: {'f1_reported','f1','checkpoint','preds','gold','n','val_f1_per_epoch'}."""
    b = _bert_json(arm)
    refit_p = SALAD2 / f"bert_{arm}_results.REFIT.json"
    refit = read_json(refit_p) if refit_p.is_file() else {}
    gold = list(b["gold"])
    pred = list(b["bert"])
    return {"f1_reported": refit.get("macro_f1", b.get("macro_f1")), "f1": macro_f1(gold, pred),
            "checkpoint": b.get("checkpoint"), "preds": pred, "gold": gold, "n": len(gold),
            "val_f1_per_epoch": refit.get("val_f1_per_epoch"), "selection": refit.get("selection")}


def matched_baselines(arm: str | None = None) -> list[dict]:
    """p6_matched_baselines.json 'cells' (DT/SVC trained on the same rows the LLM sweep used), optionally per arm."""
    p = SALAD2 / "p6_matched_baselines.json"
    if not p.is_file():
        return []
    cells = read_json(p).get("cells", [])
    return [c for c in cells if arm is None or c.get("arm") == arm]


def system_preds(arm: str, spec: str) -> tuple[list[str], list[str]]:
    """(gold, pred) on the arm's 10,000 shared test rows for a baseline alias or a full-test cell name."""
    gold = test_gold(arm)
    if spec.lower() in _BASELINE_ALIASES:
        if _BASELINE_ALIASES[spec.lower()] == "bert" and list(_bert_json(arm)["gold"]) != gold:
            raise AnalysisDataError(f"{arm}: BERT preds file gold differs from the baseline preds file gold")
        return gold, baseline_preds(arm, spec)
    cell = load_cell(spec)
    if cell is None:
        raise AnalysisDataError(f"unknown system {spec!r}: not dt|svc|majority|bert and no such cell")
    if cell.arm != arm:
        raise AnalysisDataError(f"cell {spec} is arm {cell.arm}, requested arm {arm}")
    if not cell.is_full_test:
        raise AnalysisDataError(f"cell {spec} is not a full-test cell (scored_on={cell.scored_on}, n={cell.n_scored}, "
                                f"eval_limit={cell.eval_limit}); paired analysis needs the shared 10,000 test rows")
    if not cell.has_preds:
        raise AnalysisDataError(f"cell {spec} has no predictions file")
    g, p = cell.gold_pred()
    if g != gold:
        raise AnalysisDataError(f"cell {spec}: gold sequence differs from the arm's test gold (rows not aligned)")
    # Equal gold sequences do not prove the rows line up: two rows with the same gold label can be swapped without
    # changing the gold sequence yet mispairing every paired comparison. The row id (k / idx) must be 0..n-1 in order.
    ids = cell.row_ids()
    if ids != list(range(len(gold))):
        first = next((i for i, v in enumerate(ids) if v != i), -1)
        raise AnalysisDataError(f"cell {spec}: row ids are not 0..{len(gold) - 1} in order (first deviation at row "
                                f"{first}); refusing to pair it with other systems by position")
    return gold, list(p)


# --------------------------------------------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------------------------------------------
def fmt(x: Any, nd: int = 4, signed: bool = False) -> str:
    """Format a number for tables; None/nan -> '-' ; ints stay ints; strings pass through."""
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return f"{int(x):+d}" if signed else str(int(x))
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return "-"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{float(x):+.{nd}f}" if signed else f"{float(x):.{nd}f}"
    return str(x)


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]], align: Sequence[str] | str | None = None,
             nd: int = 4) -> str:
    """GitHub-flavoured markdown table. align: per-column 'l'|'r'|'c' (string like 'lrr' or list); default: text left,
    numbers right (decided from the first row)."""
    rows = [list(r) for r in rows]
    ncol = len(headers)
    if align is None:
        first = rows[0] if rows else []
        align = ["r" if (i < len(first) and isinstance(first[i], (int, float, np.integer, np.floating))
                         and not isinstance(first[i], bool)) else "l" for i in range(ncol)]
    align = list(align)
    while len(align) < ncol:
        align.append("l")
    sep = {"l": ":---", "r": "---:", "c": ":---:"}

    def cell(v: Any) -> str:
        return fmt(v, nd).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "| " + " | ".join(sep.get(a, ":---") for a in align[:ncol]) + " |"]
    for r in rows:
        r = r + [""] * (ncol - len(r))
        lines.append("| " + " | ".join(cell(v) for v in r[:ncol]) + " |")
    return "\n".join(lines)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/paths/sets to JSON-safe python; nan/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in (sorted(obj, key=str) if isinstance(obj, set) else obj)]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_report(name: str, md: str, data: Any = None, out_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Write <name>.md and <name>.json under results/salad2/dgx/analysis/ (or `out_dir`, used by tests).
    `name` must be a bare file stem. The JSON is `data` (jsonable-converted) with a `_meta` block added."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", name) or ".." in name:
        raise ValueError(f"report name must be a bare file stem, got {name!r}")
    out = Path(out_dir) if out_dir is not None else ANALYSIS_DIR
    out.mkdir(parents=True, exist_ok=True)
    md_path, js_path = out / f"{name}.md", out / f"{name}.json"
    payload = to_jsonable(data) if data is not None else {}
    if isinstance(payload, dict):
        payload.setdefault("_meta", {"report": name, "package": "dgx_analysis"})
    md_path.write_text(md if md.endswith("\n") else md + "\n", encoding="utf-8")
    js_path.write_text(json.dumps(payload, indent=1, allow_nan=False, sort_keys=False) + "\n", encoding="utf-8")
    return md_path, js_path
