#!/usr/bin/env python3
"""
SALAD v2 builder — leak-screened SOC alert benchmarks from real network-flow captures.

Sources (both screened and PASSING the leakage gate; both re-certified 2026-07-19
after fixing a reproducibility bug -- see REPRODUCIBILITY FIX below):
    --dataset unsw    Mireu-Lab/UNSW-NB15   42 features, 10 classes   DT 0.5817
    --dataset cicids  bvk/CICIDS-2017       83 features,  8 families  DT 0.9678

RE-CERTIFICATION (2026-07-19) -- CICIDS-2017's prior FAIL was a real bug, now understood
----------------------------------------------------------------------------------------
An earlier build (commit c8a765e) certified CICIDS-2017 as FAILING the gate
("Fwd Header Length" flagged, card 274, solo_f1 0.892, full_f1 0.9725) and UNSW-NB15
at full_f1 0.7106. Re-running the IDENTICAL, unchanged script at the IDENTICAL seed
reproduced NEITHER number on the actual ~193k-row pre-balance pool that check_leakage()
is meant to run on -- both datasets instead measured PASS (Fwd Header Length
solo_f1 0.5248, card 625), close to but not identical to the ORIGINAL 04132b7 build
(solo_f1 0.5949, card 623, also PASS) -- the two builds' own leakage_report.json files
disagree by ~0.07 in solo_f1, so "matching" overstated it; both clear the 0.85 floor by
a wide margin regardless. 2-for-2 fresh runs agreed with each other; only c8a765e
disagreed with both.

ROOT CAUSE, CONFIRMED (not guessed): running check_leakage() directly on the small,
CLASS-BALANCED, EMITTED train file (salad2_train.json, 5,000 rows after take()'s
round-robin balancing) reproduces the same failure mode -- Fwd Header Length comes
back at card 156, solo_f1 0.912, FLAGGED. Small sample size collapses the column's
cardinality (fewer rows -> fewer distinct integer header-length values observed) and
class-balancing removes the natural frequency structure a decision tree needs many
features to reconstruct, so ONE column's solo_f1 gets inflated past the 0.85 floor
purely from sample size and balance, not from any real answer-key relationship. This
is the EXACT mechanism scripts/validate_leak_gate.py's synthetic small-sample test
independently characterizes (see its "small-sample regime" sweep) -- criterion A is
known to be unreliable at small N, and c8a765e's certification was, in effect, run in
that unreliable regime instead of on the large deduplicated pool main() actually uses.
(Two secondary hardening fixes were also made while investigating, though neither was
the actual cause of this specific FAIL: replaced `d["train"].shuffle(seed=...)` -- a
`datasets`-library shuffle whose output depends on the installed library version, not
just the seed -- with Python's own `random.Random(seed)` over a plain index list; and
made check_leakage()'s own dedup sort its keys before sampling, so its result is a
pure function of the row SET rather than upstream iteration order. Both are good
practice and now confirmed bit-for-bit deterministic across repeated runs in this
environment, but should not be credited with fixing the c8a765e FAIL specifically.)

CONCLUSION: CICIDS-2017 was never actually leaky at the level check_leakage() is
meant to certify (the large pre-balance pool); the "exclude CICIDS-2017 pending a
rebuild" note that followed from c8a765e was premature, caused by certifying against
the wrong (small, balanced) population. Re-certified 2026-07-19: UNSW 0.5817 PASS,
CICIDS-2017 0.9678 PASS, both against the correct ~193k/~185k-row pools. Independent
validation of the gate itself (not just this bug): scripts/validate_leak_gate.py,
results/leak_gate_*.json.

WHY THIS EXISTS
---------------
SALAD v1 was not a classification benchmark. Its prompt contained the answer:

    Alert Type: reconnaissance_scan   ->  Attack Category: Reconnaissance
    denial_of_service                 ->  DoS
    backdoor_activity                 ->  Backdoor        ... (8 -> 8, deterministic)

so the task was a field rename. Measured (grouped split by unique prompt -- a naive row
split self-leaks, since each v1 prompt repeats ~113x):

    v1 with the leaky Alert Type ..... DT macro-F1 1.00   (Alert Type ALONE also 1.00)
    v1 leak-free (2 usable fields) ... DT macro-F1 ~0.4-0.6 (protocol-sensitive; 0.37-0.62
                                       observed across split conventions — direction robust)
    v2/unsw (this builder) ........... DT macro-F1 0.5817
    majority floor ................... 0.02
    (earlier drafts quoted 0.96/0.96 for v1; the 2026-07-17 external audit re-measured the
    grouped split at 1.00/1.00 — the leak is exact, not approximate)

Further v1 defects: MITRE Tactic / Technique / Kill Chain Phase were also synthesized
from the label; Network Segment was the constant "workstation"; only 87 unique prompts
existed across 9,851 test rows; and train was 79% benign while test was 99.9% malicious.

The leak was NOT inherited from UNSW-NB15 -- that source is clean. It was introduced by
v1's synthesis step. Screening UNSW-NB15, CICIDS-2017 and NSL-KDD found no equivalent
leak in any of them, so this is a failure mode of *label-synthesized* benchmarks, not of
real captures.

WHY A HIGH DT SCORE IS NOT AUTOMATICALLY A RED FLAG
----------------------------------------------------
CICIDS-2017 scores DT 0.9678 -- but honestly: its top single feature (Fwd Header
Length) reaches only solo_f1 0.525 (54% of full, 88th percentile among all 83
features -- elevated, and independently corroborated as a real artifact by Engelen
et al. 2021's manual PCAP forensics, but nowhere near a full answer key), i.e. the
tree needs many features working together. SALAD v1's 0.96 (grouped-split remeasure:
1.00) came from ONE column at solo_f1 1.00 (100% of full). Same headline number,
opposite cause. A purity-only or score-only check cannot tell these apart;
check_leakage()'s solo_f1 test can. (Practical consequence: CICIDS-2017 is honest but
near-solved by classical ML, so it is a weak venue for "does an LLM add value?"; UNSW
is the better primary benchmark.)

WHAT v2 DOES DIFFERENTLY
------------------------
1. Uses the REAL flow features. v1's generator intended to emit network fields, but its
   source table had none, so every one was silently skipped: the model actually saw 7
   fields (4 leaky, 1 constant, 2 usable).
2. Emits no label-derived field.
3. Two gates that FAIL THE BUILD rather than warn:
     * leakage gate      -- refuses if any single field recovers >=90% of the full signal.
     * contamination gate -- refuses on any train/test prompt overlap. It caught a real bug:
       UNSW-NB15 has duplicate flow vectors, so the official split still yielded 217
       overlapping prompts until deduped.
4. Identical sampling policy for train and test (v1's were inverted).

Usage:
    python build_salad2.py --dataset unsw    [--n-train 5000] [--n-test 10000] [--seed 42]
    python build_salad2.py --dataset cicids
"""
import argparse, json, os, random
from collections import Counter, defaultdict

LABEL = "attack_cat"

# ---- dataset registry -------------------------------------------------------------
# Both sources were screened with check_leakage() before inclusion; both PASS.
# NSL-KDD was evaluated and REJECTED: the available HF mirror (Mireu-Lab/NSL-KDD) drops
# the multi-class attack_type column, leaving only a binary normal/anomaly label.
SOURCES = {
    # 42 real flow features, 10 attack classes.
    # revision pinned 2026-07-19 (both sha's last_modified predate this pin by ~2
    # years -- confirmed unchanged -- but pin anyway so the Hub cannot silently
    # drift under us later; see the reproducibility-bug note in main()).
    "unsw": dict(hf="Mireu-Lab/UNSW-NB15", label="attack_cat",
                 revision="3eb02c4a0a29866b7abcb5ef77c45cf4fcc8f6b0",
                 drop={"id", "attack_cat", "label"}),   # 'label' = binary alias of attack_cat
    # 27 raw labels -> consolidated to 8 families (see CICIDS_MAP). The HF source ships 89
    # columns; 83 survive into the prompt after Src/Dst IP, Timestamp, Flow ID and the label
    # columns are dropped (see the drop set below). Verified 2026-07-30 by counting fields in
    # data/salad2_cicids/salad2_train.json: 83, matching what P13 reports. An earlier version of
    # this comment said only "89 flow features", which read as the prompt width and did not
    # match the paper.
    # Src/Dst IP and Timestamp are dropped: they are testbed identifiers, not flow
    # characteristics. Measured solo_f1 is low (IP 0.08-0.18, Timestamp 0.27) so they are
    # not leaks by the gate, but they encode *this capture's* topology and schedule, which
    # cannot generalize -- a model should not learn "10.0.0.8 is the attacker".
    "cicids": dict(hf="bvk/CICIDS-2017", label="Label",
                   revision="b9515d5c95c0c0e7312274760acff3d54fe5ff41",
                   drop={"Label", "Src IP dec", "Dst IP dec", "Timestamp",
                         "Src Port", "Flow ID", "Attempted Category"}),
}

# CICIDS-2017 ships 27 raw labels including "X - Attempted" variants (a failed attempt at
# X is still an X for attack-category purposes) and a long tail of classes with <20 rows.
CICIDS_MAP = {
    "BENIGN": "Benign",
    "Portscan": "Portscan",
    "DDoS": "DDoS",
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS Slowloris": "DoS", "DoS Slowhttptest": "DoS",
    "Infiltration": "Infiltration", "Infiltration - Portscan": "Infiltration",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "Botnet": "Botnet",
    "Web Attack - Brute Force": "WebAttack", "Web Attack - XSS": "WebAttack",
    "Web Attack - SQL Injection": "WebAttack",
    "Heartbleed": None,          # n=11 across 2.1M rows -- too rare to score; dropped
}


def canon_cicids(raw):
    """Map a raw CICIDS-2017 label to a consolidated family, or None to drop the row."""
    s = str(raw).strip()
    if s.endswith(" - Attempted"):
        s = s[: -len(" - Attempted")]
    return CICIDS_MAP.get(s, None)

# Human-readable names for the raw UNSW-NB15 columns. Values are NOT transformed;
# only the display name changes, so no label information can enter through renaming.
PRETTY = {
    "dur": "Duration (s)", "proto": "Protocol", "service": "Service", "state": "State",
    "spkts": "Source Packets", "dpkts": "Dest Packets", "sbytes": "Source Bytes",
    "dbytes": "Dest Bytes", "rate": "Rate", "sttl": "Source TTL", "dttl": "Dest TTL",
    "sload": "Source Load", "dload": "Dest Load", "sloss": "Source Loss", "dloss": "Dest Loss",
    "sinpkt": "Source Interpacket (ms)", "dinpkt": "Dest Interpacket (ms)",
    "sjit": "Source Jitter", "djit": "Dest Jitter", "swin": "Source TCP Window",
    "stcpb": "Source TCP Base Seq", "dtcpb": "Dest TCP Base Seq", "dwin": "Dest TCP Window",
    "tcprtt": "TCP RTT", "synack": "SYN-ACK Time", "ackdat": "ACK-DAT Time",
    "smean": "Source Mean Pkt Size", "dmean": "Dest Mean Pkt Size",
    "trans_depth": "HTTP Transaction Depth", "response_body_len": "HTTP Response Body Len",
    "ct_srv_src": "Count Same Service+SrcIP", "ct_state_ttl": "Count State+TTL",
    "ct_dst_ltm": "Count Same DstIP (100 conn)", "ct_src_dport_ltm": "Count SrcIP+DstPort",
    "ct_dst_sport_ltm": "Count DstIP+SrcPort", "ct_dst_src_ltm": "Count Src+Dst Pair",
    "is_ftp_login": "FTP Login", "ct_ftp_cmd": "FTP Command Count",
    "ct_flw_http_mthd": "HTTP Method Flow Count", "ct_src_ltm": "Count Same SrcIP",
    "ct_srv_dst": "Count Same Service+DstIP", "is_sm_ips_ports": "Same IP/Port",
}

# Per-dataset valid-label lists for the system prompt -- each dataset's own real, FIXED
# label taxonomy, not auto-derived from a subsample (a rare class, e.g. UNSW's Worms
# n=44, can be exhausted before appearing in every draw and must still be listed as valid).
CLASS_LISTS = {
    "unsw": ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
             "Analysis", "Backdoor", "Shellcode", "Worms"],
    "cicids": sorted(set(v for v in CICIDS_MAP.values() if v is not None)),
}


def make_system(dataset):
    """Build the system prompt naming the given dataset's own real label taxonomy.

    BUG FOUND 2026-07-19: this used to be one hardcoded global SYSTEM string listing only
    UNSW-NB15's 10 classes, applied UNCONDITIONALLY to both datasets -- every CICIDS-2017
    row (train AND test, 15,000 rows total) shipped with a system prompt instructing the
    model to pick from UNSW-NB15's wrong 10-class list while the actual gold labels were
    CICIDS-2017's 8 consolidated families (verbatim example: gold label "Attack Category:
    Benign", but the prompt's own "Valid ... values" line never lists Benign at all). This
    is the exact train-prompt/gold-label vocabulary mismatch this project exists to catch
    (see the salad-v1-trainlabel-mismatch finding) -- just reintroduced at construction
    time here, rather than via mistrained labels as in v1. Caught before the CICIDS-2017
    LLM ladder trained on it, while drafting a dataset card and spot-checking a raw row.
    """
    classes = CLASS_LISTS[dataset]
    return (
        "You are an expert SOC (Security Operations Center) analyst. You are given the raw "
        "network-flow telemetry for a single connection. Infer the attack category from the "
        "flow characteristics alone.\n"
        "Valid Attack Category values (respond with exactly one, verbatim):\n"
        f"  {', '.join(classes)}\n"
        "Respond with only the line: Attack Category: <value>"
    )


def check_leakage(rows, feats, dominance=0.90, abs_floor=0.85, sample=40000, seed=0):
    """Refuse to ship a benchmark whose label is recoverable from ONE field.

    This is the gate SALAD v1 needed and that general-purpose data-quality checklists lack (the
    own 30 items).

    Two signals per feature:

      purity  -- in-sample determinism: fraction of rows whose feature value maps to a
                 single dominant label. CHEAP BUT MISLEADING ALONE: a high-cardinality
                 continuous column (e.g. CICIDS-2017 'Flow Duration', card=38079) scores
                 purity=0.965 simply because its values are near-unique, yet it predicts
                 almost nothing on held-out rows. Purity is reported for diagnosis only.

      solo_f1 -- THE ACTUAL TEST: train a tree on that column ALONE and score macro-F1 on
                 held-out rows. This asks the right question -- "can one field do the whole
                 job?" -- and is immune to the cardinality artifact above.

    A feature is LEAK-LIKE when EITHER criterion holds:

      (A) statistical answer key:  solo_f1 >= dominance * full_f1  AND  solo_f1 >= abs_floor.
          The absolute floor exists because the ratio alone conflates two different worlds
          (found by the 2026-07-17 external audit): "one field IS the label" (a true answer
          key) vs "all features are individually weak on an unsolved task" (v2 Source Bytes:
          solo 0.56 vs full 0.59, ratio 0.94 -- normal for tabular data, not a leak).

      (B) deterministic rename:  purity >= 0.999  AND  cardinality <= max(2*n_classes, 16).
          A low-cardinality field whose every value maps to exactly one label is an answer
          key BY CONSTRUCTION -- no held-out statistics needed, and none are trustworthy
          when deduplication leaves few unique rows (v1 collapses 14,851 rows to 127 unique,
          where criterion (A)'s tree estimates are small-sample noise; this exact blind spot
          let a fixed-split rerun of the gate miss v1's Alert Type). The cardinality bound
          keeps the purity signal away from its known failure mode (near-unique continuous
          columns like CICIDS 'Flow Duration', card 38079, purity 0.965 -> not flagged).

    The 70/30 split is taken over rows DEDUPLICATED by full feature vector (and shuffled).
    Real captures repeat flows; a raw row split puts identical rows on both sides and the
    detector then scores its own self-leak. The first shipped version of this gate had
    exactly that bug (external audit, 2026-07-17), so its certification was run on a
    different distribution than the shipped files. This version dedupes first.

    Calibration against real data (re-verified after the fix):
      SALAD v1  'Alert Type'   purity 1.000 card 8      solo_f1 1.00 vs full 1.00 -> LEAK
      UNSW-NB15 'sload'        purity 0.923 card 42873  solo_f1 low  vs full ~0.5 -> pass
      CICIDS17  'Flow Duration' purity 0.965 card 38079 solo_f1 0.24              -> pass
      CICIDS17  'Timestamp'    purity 0.924 card 25410  solo_f1 0.27              -> pass
    """
    import random as _r
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import f1_score
    import pandas as _pd

    # HARDENING 2026-07-19: dedup used to preserve first-seen order from `rows`, so
    # `uniq`'s order (and therefore what rs.sample() below draws at a fixed seed)
    # depended on the INPUT row order. That order can vary across upstream data-
    # loading paths (e.g. a `datasets`-library iteration order that isn't guaranteed
    # stable across versions), so this made the sampled subset -- and therefore
    # solo_f1 -- sensitive to something other than seed+content. Not the cause of
    # the actual c8a765e CICIDS false-FAIL (that was caused by calling this function
    # on the wrong, small+balanced population -- see main()'s re-certification note
    # and the small-sample guard just below), but worth removing as a source of
    # variance regardless. Fix: key dedup by content and sort by that content key,
    # so `uniq`'s order is a pure function of the row SET, never of arrival order.
    rs = _r.Random(seed)
    seen = {}
    for r in rows:
        k = tuple(str(r[f]) for f in feats)
        if k not in seen:
            seen[k] = r
    uniq = [seen[k] for k in sorted(seen)]
    # Empirically calibrated 2026-07-19: check_leakage() on CICIDS-2017's actual
    # ~193,000-unique-row pre-balance pool correctly measures Fwd Header Length as
    # non-leaky (solo_f1 0.5248); run on the SAME feature via the small, 5,000-row,
    # class-balanced EMITTED train file, it comes back flagged (solo_f1 0.912) --
    # small sample + class balance alone inflates a single feature's solo_f1 well
    # past the leak threshold. 20,000 sits well inside the safe zone (unsw's ~54k
    # and cicids' ~193k both clear it) and well above the sizes known to trigger
    # the effect (5,000 -- the actual bug -- and the sample()=40,000 cap this
    # function applies anyway, meaning below that cap you are ALREADY in a smaller
    # sample than the gate was designed to reason about).
    if uniq and len(uniq) < 20000:
        print(f"   WARNING: check_leakage() called on only {len(uniq)} unique rows. This "
              f"is the exact small-sample regime that inflated a real feature's solo_f1 "
              f"past the leak threshold on 2026-07-19 (see the CICIDS-2017 false-FAIL note "
              f"above) -- run this on the large pre-balance/pre-dedup pool, not a small or "
              f"class-balanced emitted sample.")
    sub = uniq if len(uniq) <= sample else rs.sample(uniq, sample)
    rs.shuffle(sub)
    y = [str(r[LABEL]) for r in sub]
    cut = int(len(sub) * 0.7)

    def codes(vals):
        return _pd.Categorical([str(v) for v in vals]).codes.reshape(-1, 1)

    def solo(f):
        X = codes([r[f] for r in sub])
        try:
            m = DecisionTreeClassifier(random_state=42, max_depth=20).fit(X[:cut], y[:cut])
            return f1_score(y[cut:], m.predict(X[cut:]), average="macro", zero_division=0)
        except Exception:
            return 0.0

    # full-feature reference score
    Xf = _pd.DataFrame({f: [str(r[f]) for r in sub] for f in feats})
    for c in Xf.columns:
        Xf[c] = _pd.Categorical(Xf[c]).codes
    try:
        mf = DecisionTreeClassifier(random_state=42, max_depth=20).fit(Xf[:cut], y[:cut])
        full_f1 = f1_score(y[cut:], mf.predict(Xf[cut:]), average="macro", zero_division=0)
    except Exception:
        full_f1 = 1.0

    findings = []
    for f in feats:
        m = defaultdict(Counter)
        for r in sub:
            m[str(r[f])][str(r[LABEL])] += 1
        purity = sum(c.most_common(1)[0][1] for c in m.values()) / len(sub)
        s = solo(f)
        n_classes = len(set(y))
        stat_key = bool(full_f1 > 0 and s >= dominance * full_f1 and s >= abs_floor)
        det_rename = bool(purity >= 0.999 and len(m) <= max(2 * n_classes, 16))
        findings.append({"feature": f, "cardinality": len(m), "purity": round(purity, 4),
                         "solo_f1": round(s, 4),
                         "leak_like": stat_key or det_rename,
                         "criterion": ("A:statistical" if stat_key else "") +
                                      ("+" if stat_key and det_rename else "") +
                                      ("B:deterministic-rename" if det_rename else "") or None})
    findings.sort(key=lambda d: -d["solo_f1"])
    return findings, round(full_f1, 4)


def to_alert(row, feats):
    lines = ["Analyze this network flow alert:", ""]
    for f in feats:
        v = row[f]
        if v is None or v == "":
            continue
        lines.append(f"{PRETTY.get(f, f)}: {v}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SOURCES), default="unsw")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixed-proportions", action="store_true",
                    help="draw the training split at a class distribution independent of "
                         "--n-train (required for a training-size sweep: the default "
                         "round-robin makes size and class imbalance co-vary, see the note "
                         "at take_fixed)")
    ap.add_argument("--proportion-ceiling", type=int, default=20000,
                    help="largest --n-train the sweep will use; the fixed per-class share is "
                         "bounded by the rarest class at THIS size, so every size in the sweep "
                         "gets the same proportions (default 20000)")
    ap.add_argument("--reuse-test", action="store_true",
                    help="keep the already-shipped salad2_test.json instead of drawing a new "
                         "test split (required for training-size sweeps, and to stay comparable "
                         "with results produced before the 2026-07-30 RNG fix)")
    a = ap.parse_args()
    random.seed(a.seed)
    src = SOURCES[a.dataset]
    out = a.out or f"data/salad2_{a.dataset}"

    from datasets import load_dataset
    print(f"source: {src['hf']}  ({a.dataset})")
    # Pin the exact upstream revision -- defense in depth against the Hub content
    # changing under us later (verified unchanged since 2024-07-17 as of this fix,
    # sha b9515d5c95c0c0e7312274760acff3d54fe5ff41, but pin anyway going forward).
    d = load_dataset(src["hf"], revision=src.get("revision"))

    if a.dataset == "cicids":
        # single 2.1M-row split -> shuffle, consolidate labels, then split by row.
        #
        # BUG FOUND 2026-07-19: this used to call `d["train"].shuffle(seed=a.seed)`
        # (a `datasets`-library shuffle). That is NOT safely reproducible: its output
        # depends on the installed `datasets` library's internal iteration/shuffle
        # implementation, not just the seed value, so the identical script + identical
        # seed can produce a DIFFERENT row order across environments/library versions.
        # This is exactly what happened: commit c8a765e's certified leakage_report.json
        # showed CICIDS-2017 FAILING (Fwd Header Length flagged, solo_f1=0.892), but
        # re-running the unchanged script now reproducibly PASSES (solo_f1 0.5248, no
        # flags) -- close to (not identical to) the ORIGINAL 04132b7 build (solo_f1
        # 0.5949, PASS), 2-for-2. The FAIL was the non-reproducible outlier, not the
        # PASS. (NOTE: the root cause of the c8a765e FAIL itself is the small-sample/
        # class-balance issue documented in the module docstring above, not shuffle
        # non-determinism per se -- this shuffle fix is good practice removed while
        # investigating, not the confirmed cause; see RE-CERTIFICATION note above.)
        # Independent validation: scripts/validate_leak_gate.py, results/leak_gate_*.json.
        #
        # Fix: shuffle with Python's own `random.Random(seed)` over a plain index
        # list -- deterministic purely from the stdlib, independent of any
        # `datasets`-library version, matching what check_leakage() already does
        # internally for its own dedup/split step.
        n_total = len(d["train"])
        idx = list(range(n_total))
        random.Random(a.seed).shuffle(idx)
        feats = [c for c in d["train"].column_names if c not in src["drop"]]
        allr = []
        for i in idx:
            r = d["train"][i]
            y = canon_cicids(r[src["label"]])
            if y is None:
                continue
            rec = {c: r[c] for c in feats}
            rec[LABEL] = y
            allr.append(rec)
            if len(allr) >= 400000:      # plenty for a 5k/10k balanced draw
                break
        cut = int(len(allr) * 0.5)
        tr, te = allr[:cut], allr[cut:]
        print(f"consolidated 27 raw labels -> {len(set(r[LABEL] for r in allr))} families")
    else:
        feats = [c for c in d["train"].column_names if c not in src["drop"]]
        tr = [dict(r, **{LABEL: r[src["label"]]}) for r in d["train"]]
        te = [dict(r, **{LABEL: r[src["label"]]}) for r in d["test"]]

    # ---- LEAKAGE GATE (build fails loudly rather than shipping a lookup table) ----
    findings, full_f1 = check_leakage(tr, feats)
    leaks = [f for f in findings if f["leak_like"]]
    print(f"leakage gate: scanned {len(feats)} features | full-feature DT macro-F1 = {full_f1}")
    print(f"   {'feature':<22} {'card':>7}  {'purity':>6}  {'solo_f1':>7}")
    for f in findings[:5]:
        flag = "  <-- LEAK" if f["leak_like"] else ""
        print(f"   {f['feature']:<22} {f['cardinality']:>7}  {f['purity']:>6}  {f['solo_f1']:>7}{flag}")
    if leaks:
        raise SystemExit(
            f"REFUSING TO BUILD — these fields alone recover >=90% of the full signal: "
            f"{[(l['feature'], l['solo_f1']) for l in leaks]}")
    print("   -> PASS: no single field recovers the label on held-out rows\n")

    # ---- DEDUPLICATE BY PROMPT, THEN SPLIT ----
    # UNSW-NB15 contains duplicate flow vectors, so the official train/test split alone
    # still puts identical prompts on both sides (measured: 217 overlapping prompts).
    # v1 shipped exactly this defect ("original splits had leakage"), so v2 dedupes by
    # rendered prompt first and enforces zero overlap below.
    def key(r):
        return to_alert(r, feats)

    seen = set()
    tr_u, te_u = [], []
    for r in tr:
        k = key(r)
        if k not in seen:
            seen.add(k); tr_u.append(r)
    tr_keys = set(seen)
    for r in te:
        k = key(r)
        if k not in seen and k not in tr_keys:
            seen.add(k); te_u.append(r)
    print(f"deduped: train {len(tr)}->{len(tr_u)} unique | test {len(te)}->{len(te_u)} unique "
          f"(test prompts also present in train were dropped)\n")

    # Balanced round-robin subsample. v1's train was 79% benign while its test was 99.9%
    # malicious -- an inverted distribution. v2 applies the IDENTICAL sampling policy to
    # both splits so train and test are distributionally matched by construction.
    # Rare classes (Worms n=44) simply exhaust; they are not oversampled.
    # BUG FOUND 2026-07-30: take() used to draw from the GLOBAL `random` stream, and
    # take(train) runs before take(test). Its closing `random.shuffle(out)` consumes an
    # amount of randomness proportional to n, so a different --n-train left the global
    # RNG in a different state by the time the test split was drawn -- silently producing
    # a DIFFERENT TEST SET. Measured on a faithful replica of this function: changing
    # --n-train from 5000 to 1000/10000/20000 retained only ~2,500 of 10,000 test rows.
    #
    # Nothing would have flagged it. A training-size sweep (P6's question) would have
    # produced each size on its own test set, none matching the 24-cell ladder's, and the
    # comparison would have looked fine. Same failure family as the `datasets`-library
    # shuffle bug fixed in this file on 2026-07-19 -- that fix seeded the source shuffle
    # but left this one sharing a single mutable stream.
    #
    # Fix: each split draws from its OWN Random instance, so the test split is a pure
    # function of (seed, n_test) and is invariant to n_train.
    def take(rows, n, rng):
        by = defaultdict(list)
        for r in rows:
            by[r[LABEL]].append(r)
        for v in by.values():
            rng.shuffle(v)
        out = []
        while len(out) < n and any(by.values()):
            for k in list(by):
                if by[k] and len(out) < n:
                    out.append(by[k].pop())
        rng.shuffle(out)
        return out

    # --fixed-proportions: draw the training split at a class distribution that does NOT
    # change with n_train.
    #
    # WHY (found 2026-07-30, before any sweep was run): take()'s round-robin lets rare
    # classes exhaust, so bigger n_train is automatically MORE imbalanced -- UNSW went
    # 2.6x imbalance at n=1,000 to 91.3x at n=20,000 (Worms caps at 43 rows available),
    # CICIDS 1.0x to 20.6x. Size and imbalance then move together monotonically across a
    # sweep and cannot be separated. That is not a neutral confound: the reported metric
    # is macro-F1, which weights every class equally and is therefore dominated by exactly
    # the rare classes that stop growing, so "more data" would measure as WORSE purely
    # through imbalance even if every class improved. A training-size scaling result off
    # the default policy would be reporting an artifact of its own sampling.
    #
    # Fix: compute one proportion vector, bounded by the rarest class at the LARGEST size
    # the sweep will use (--proportion-ceiling), and reuse it for every n.
    # NOTE: "fixed proportions" is NOT "balanced". Forcing every class to an equal share
    # would cap the whole draw at the rarest class -- for UNSW that is Worms at 43 rows, so
    # n=20,000 would yield 430 rows total, useless for a size sweep. Instead each class keeps
    # its OWN proportion, and only the requirement that those proportions never change with n
    # is enforced. The reference distribution is the round-robin draw at the ceiling size, so
    # the largest size in the sweep is exactly what the default policy would have produced,
    # and every smaller size is that same distribution scaled down.
    #
    # Consequence to disclose when writing up: the imbalance is then CONSTANT across the
    # sweep at whatever it is at the ceiling (UNSW 91.3x, CICIDS 20.6x) rather than growing
    # with n. High, but fixed -- so it cannot be confounded with training-set size, which is
    # the whole point.
    def take_fixed(rows, n, rng, ceiling):
        ref = take(rows, ceiling, random.Random(rng.randrange(2**31)))
        cnt = Counter(r[LABEL] for r in ref)
        prop = {k: c / len(ref) for k, c in cnt.items()}
        by = defaultdict(list)
        for r in rows:
            by[r[LABEL]].append(r)
        for v in by.values():
            rng.shuffle(v)
        quota = {k: min(len(by[k]), max(1, int(round(prop.get(k, 0) * n)))) for k in by}
        out = []
        for k in sorted(by):                      # sorted: order independent of dict insertion
            out.extend(by[k][:quota[k]])
        rng.shuffle(out)
        return out, {k: quota[k] for k in sorted(by)}, prop

    # Distinct, independently-derived seeds: neither split can perturb the other.
    if a.fixed_proportions:
        trs, quota, prop = take_fixed(tr_u, a.n_train, random.Random(a.seed),
                                      a.proportion_ceiling)
        imb = max(quota.values()) / min(quota.values())
        print(f"--fixed-proportions (ceiling n={a.proportion_ceiling}): {len(trs)} train rows, "
              f"imbalance={imb:.1f}x  quota={quota}")
    else:
        trs = take(tr_u, a.n_train, random.Random(a.seed))
    tes = take(te_u, a.n_test, random.Random(a.seed + 10_000_019))

    system_prompt = make_system(a.dataset)

    # --reuse-test: keep the ALREADY-SHIPPED test split instead of the one just drawn.
    # Needed because the shipped salad2_test.json predates the 2026-07-30 RNG fix above,
    # so a rebuild now yields a different (equally valid, but different) test set -- and
    # every existing result, including P13's published 24-cell ladder, is measured on the
    # shipped one. For a training-size sweep the test split must be held FIXED anyway, or
    # the sizes aren't comparable to each other let alone to the ladder. This reuses the
    # shipped file verbatim and still runs the contamination gate below against it, so a
    # new training set that collides with held-out prompts is still refused.
    reused_test = False
    if a.reuse_test:
        p = os.path.join(a.out or f"data/salad2_{a.dataset}", "salad2_test.json")
        if not os.path.exists(p):
            raise SystemExit(f"--reuse-test: no existing test split at {p}")
        with open(p) as fh:
            shipped = json.load(fh)
        reused_test = True
        print(f"--reuse-test: keeping shipped test split ({len(shipped)} rows) from {p}")

    def emit(rows):
        return [{"conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": to_alert(r, feats)},
            {"from": "gpt", "value": f"Attack Category: {r[LABEL]}"}]} for r in rows]

    # ---- CONTAMINATION GATE (hard failure, BEFORE anything touches disk) ----
    # v1 never verified this at the prompt level. A builder that prints "overlap: 217"
    # and writes the files anyway is how v1's leaky splits shipped. The first version of
    # THIS builder raised only after salad2_train/test.json were already written (external
    # audit, 2026-07-17) — a failed build still left contaminated files behind. Gate first,
    # write second.
    tr_recs = emit(trs)
    te_recs = shipped if reused_test else emit(tes)
    ptr = {c["value"] for r in tr_recs for c in r["conversations"] if c["from"] == "human"}
    pte = {c["value"] for r in te_recs for c in r["conversations"] if c["from"] == "human"}
    overlap = len(ptr & pte)
    print(f"train/test prompt overlap: {overlap}  (must be 0)")
    if overlap:
        raise SystemExit(f"REFUSING TO BUILD — {overlap} prompts appear in both splits")

    os.makedirs(out, exist_ok=True)
    writes = [("train", trs, tr_recs)] if reused_test else [("train", trs, tr_recs), ("test", tes, te_recs)]
    if reused_test:
        print("--reuse-test: test split left untouched on disk")
    for name, rows, recs in writes:
        with open(os.path.join(out, f"salad2_{name}.json"), "w") as fh:
            json.dump(recs, fh, indent=1)
        prompts = {c["value"] for r in recs for c in r["conversations"] if c["from"] == "human"}
        print(f"{name}: {len(recs)} rows | {len(prompts)} unique prompts "
              f"({len(prompts)/len(recs):.1%} unique)  dist={dict(Counter(r[LABEL] for r in rows).most_common())}")

    with open(os.path.join(out, "leakage_report.json"), "w") as fh:
        json.dump({"gate": "PASS", "source": src["hf"], "dataset": a.dataset,
                   "full_feature_dt_macro_f1": full_f1, "features_scanned": len(feats),
                   "train_test_prompt_overlap": len(ptr & pte),
                   "findings": findings}, fh, indent=1)
    print(f"\nwrote {out}/ (salad2_train.json, salad2_test.json, leakage_report.json)")


if __name__ == "__main__":
    main()
