#!/usr/bin/env python3
"""
SALAD-v2 general LoRA/OFT experiment runner for the EdgeXpert (GB10) -- v2 of p22_attrib.py (2026-09-26).

Superset of p22_attrib.py: every recipe factor is a per-cell key, both arms, all four model sizes, LoRA and OFT,
optional micro-batching, and a validation-selection mode. Same data files, same PLAIN-TEXT render (human + "\n" + gpt),
same greedy 16-token eval harness as P14/P22, so a cell with P22's settings reproduces a P22 cell.

Usage: python attrib.py queue.json
Cell keys (defaults):
  name (required, unique), arm=unsw|cicids (unsw), model=0.5b|1.5b|3b|7b (0.5b), method=lora|oft (lora),
  rank (8; OFT r for method=oft), alpha (2*rank; LoRA only), lr (2e-4), steps (0 => 3 epochs), sched=cosine|const (cosine),
  layers=all|top8 (all; LoRA only), rslora=0|1 (0; scale alpha/sqrt(r) when 1), tuned="arm:method" (adopt val-NLL-best lr/epochs from the T_ tuning grid at run time), tuned_use=lr|lr+epochs, epochs, train_dir (dir holding salad2_train.json), save_adapter=0|1, mask_pad=0|1 (0 = P14/P22 behaviour: labels=input_ids incl. padding), eval_bs (1 = original batch-1 harness; >1 batched left-padded generation, validate first), opt=adamw|adam (adamw), seed (42), dtype=fp32|bf16 (fp32),
  micro (0 => whole batch of 4 at once; 1 => 4x accumulation of 1-row micro-batches, like P14 CICIDS),
  [A_CKPT_EVERY env: checkpoint every N steps (default 500; 0 disables) -> $A_OUT/ckpt/<name>, auto-resumed]
  model_id (None; overrides MODELS[model] with an arbitrary HF repo id -- 'model' stays the size LABEL for bookkeeping),
  adapter_dir (None; set to skip training and instead load this saved PEFT adapter for EVAL-ONLY scoring -- cross-arm
    transfer: pair with eval_arm to score on a DIFFERENT arm's test set than the adapter's own, and source_arm to record
    provenance), eval_arm (None => c['arm']; which arm's test set to score, independent of the arm the adapter trained on),
  eval_limit (0 => full 10k test; else fixed random subsample Random(0)),
  val (0 => train on all 5000 rows and score the TEST set; N>0 => hold out N fixed rows (Random(1234)), train on the rest,
       score val rows: strict macro-F1 AND answer-token NLL, never touches the test set)
Results -> $A_OUT (default results/attrib)/{name}.json (+ .preds.jsonl). Resumable: skips cells whose JSON exists.
"""
import gc, glob, json, os, shutil, sys, time, random, re
# every model the queues use is already in $HF_HOME (~/hf_cache), so never ask the hub for it: on 2026-10-20 a DNS outage made
# every cell die in the tokenizer's hub lookup within ~5 min and the lane driver retried that for ~14 h. Set HF_HUB_OFFLINE=0 to
# download a model that is not cached yet. Must be set before transformers is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, OFTConfig, get_peft_model
from sklearn.metrics import f1_score

DEV = "cuda"
MODELS = {"0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
          "3b": "Qwen/Qwen2.5-3B-Instruct", "7b": "Qwen/Qwen2.5-7B-Instruct"}
BATCH = 4
MAX_LEN = 1536
EPOCHS = 3
CHECK = {}
OUT = os.environ.get("A_OUT", "results/attrib")
os.makedirs(OUT, exist_ok=True)
DEFAULTS = dict(arm="unsw", model="0.5b", model_id=None, method="lora", rank=8, alpha=None, lr=2e-4, steps=0, sched="cosine",
                layers="all", opt="adamw", seed=42, dtype="fp32", micro=0, eval_limit=0, val=0, rslora=0, dora=0, mask_pad=0, eval_bs=1,
                tuned=None, tuned_use="lr+epochs", epochs=0, train_dir=None, save_adapter=0,
                adapter_dir=None, eval_arm=None, source_arm=None)


def model_name(c):
    """c['model_id'] overrides c['model'] with an arbitrary HF repo id (for a 2nd model family); c['model'] stays the
    size LABEL used in cell names/cost bookkeeping either way."""
    return c["model_id"] or MODELS[c["model"]]


# ---- crash-tolerance (2026-10-02): the kernel OOM-killer on this shared unified-memory box killed 46 of this user's
# processes (cgroup memory.events oom_kill), each time silently discarding hours of training. Every cell now writes a
# resumable checkpoint (adapter weights + optimizer + scheduler + step + loss history) every A_CKPT_EVERY steps and once
# more when training finishes, so a kill costs at most that many steps (or just the eval pass) instead of the whole cell.
CKPT_EVERY = int(os.environ.get("A_CKPT_EVERY", "500"))
CKPT_ROOT = f"{OUT}/ckpt"
RESUMED = {}


def save_ckpt(m, opt, sched, step, losses, cdir):
    tmp = cdir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    m.save_pretrained(tmp)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict() if sched else None,
                "step": step, "losses": losses}, f"{tmp}/state.pt")
    shutil.rmtree(cdir, ignore_errors=True)
    os.replace(tmp, cdir)


def try_resume(m, opt, sched, cdir):
    if not (os.path.exists(f"{cdir}/state.pt") and os.path.exists(f"{cdir}/adapter_model.safetensors")):
        return 0, []
    try:
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        st = torch.load(f"{cdir}/state.pt", map_location="cpu", weights_only=False)
        set_peft_model_state_dict(m, load_file(f"{cdir}/adapter_model.safetensors"))
        opt.load_state_dict(st["opt"])
        if sched is not None and st.get("sched") is not None:
            sched.load_state_dict(st["sched"])
        log(f"    RESUMED from checkpoint at step {st['step']} ({cdir})")
        return st["step"], st["losses"]
    except Exception as e:
        log(f"    checkpoint unreadable ({e!r}) -- training from scratch")
        return 0, []


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def render_human(conv):
    return {x["from"]: x["value"] for x in conv}.get("human", "")


def render_full(conv):
    c = {x["from"]: x["value"] for x in conv}
    return c.get("human", "") + "\n" + c.get("gpt", "")


def parse_label(text):
    m = re.search(r"Attack Category:\s*([A-Za-z][A-Za-z0-9_\- ]*)", text)
    return m.group(1).strip() if m else text.strip().split("\n")[0].strip()



def resolve_tuned(c):
    """cell key tuned="arm:method": pick the val-NLL-best T_<arm>_<method>_*.json from the tuning grid at RUN time and adopt its lr
    (and epochs when tuned_use contains 'epochs'); fall back to lr 2e-4 / 3 epochs when no tuning result exists. Logged in config."""
    if not c.get("tuned"):
        return
    arm, method = c["tuned"].split(":")
    best, n = None, 0
    for f in sorted(glob.glob(f"{OUT}/T_{arm}_{method}_*.json")):
        d = json.load(open(f))
        if d.get("val_answer_nll") is None:
            continue
        n += 1
        if best is None or d["val_answer_nll"] < best["val_answer_nll"]:
            best = d
    if best is None:
        c["tuned_resolved"] = {"fallback": True, "lr": c["lr"], "epochs": c["epochs"] or EPOCHS}
        log(f"    tuned={c['tuned']}: NO tuning results found -> fallback lr={c['lr']}")
        return
    c["lr"] = best["config"]["lr"]
    if "epochs" in c["tuned_use"]:
        c["epochs"] = round(best["config"]["steps"] / (4500 // BATCH))
    c["tuned_resolved"] = {"from": best["name"], "val_answer_nll": best["val_answer_nll"], "n_candidates": n,
                           "lr": c["lr"], "epochs": c["epochs"] or EPOCHS}
    log(f"    tuned={c['tuned']}: {best['name']} (nll {best['val_answer_nll']}, {n} candidates) -> lr={c['lr']:g} epochs={c['epochs'] or EPOCHS}")


def split_data(c):
    data = json.load(open(f"{c['train_dir'] or 'data/salad2_' + c['arm']}/salad2_train.json"))
    if not c["val"]:
        return data, None
    hold = set(random.Random(1234).sample(range(len(data)), c["val"]))
    return [r for i, r in enumerate(data) if i not in hold], [r for i, r in enumerate(data) if i in hold]


def train_one(c, data):
    seed = c["seed"]
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    name = model_name(c)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = torch.bfloat16 if c["dtype"] == "bf16" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(name, dtype=dt).to(DEV)
    if c["method"] == "oft":
        cfg = OFTConfig(r=c["rank"], oft_block_size=0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    else:
        kw = {}
        if c["layers"] == "top8":
            n = m.config.num_hidden_layers
            kw["layers_to_transform"] = list(range(n - 8, n))
        cfg = LoraConfig(r=c["rank"], lora_alpha=c["alpha"], target_modules=["q_proj", "v_proj"],
                         task_type="CAUSAL_LM", lora_dropout=0.0,
                         use_rslora=bool(c["rslora"]), use_dora=bool(c["dora"]), **kw)
    m = get_peft_model(m, cfg)
    params = [p for p in m.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    opt = (torch.optim.Adam(params, lr=c["lr"]) if c["opt"] == "adam" else torch.optim.AdamW(params, lr=c["lr"]))
    total = c["steps"] if c["steps"] else (len(data) // BATCH) * (c["epochs"] or EPOCHS)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1)) if c["sched"] == "cosine" else None
    micro = c["micro"] or BATCH
    n_micro = BATCH // micro
    log(f"    train: {c['arm']} {c['model']} {c['method']} r={c['rank']} alpha={c['alpha']} lr={c['lr']:g} "
        f"steps={total} sched={c['sched']} opt={c['opt']} layers={c['layers']} {c['dtype']} micro={micro}x{n_micro} "
        f"n_train={len(data)} trainable={trainable:,}")
    m.train()
    cdir = f"{CKPT_ROOT}/{c['name']}"
    step, losses = try_resume(m, opt, sched, cdir)
    RESUMED[c["name"]] = step
    start_step = step
    n_rows = len(data)
    spe = (n_rows - BATCH) // BATCH + 1                    # optimizer steps per epoch (matches the original range() loop)
    idx = list(range(n_rows))
    ep_done, first, t0 = step // spe, True, time.time()
    for ep in range(ep_done):                              # replay the in-place shuffles of finished epochs
        random.Random(seed + ep).shuffle(idx)
    skip = step - ep_done * spe                            # batches already consumed in the current epoch
    epoch = ep_done
    while step < total:
        random.Random(seed + epoch).shuffle(idx)
        start_b = skip if first else 0
        first = False
        for bi in range(start_b, spe):
            i = bi * BATCH
            if step >= total:
                break
            rows = [data[j] for j in idx[i:i + BATCH]]
            step_loss = 0.0
            for k in range(0, BATCH, micro):
                enc = tok([render_full(r["conversations"]) for r in rows[k:k + micro]], return_tensors="pt",
                          padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
                lab = enc["input_ids"].clone()
                if c["mask_pad"]:
                    lab[enc["attention_mask"] == 0] = -100   # do not train on right-padding positions
                out = m(**enc, labels=lab)
                (out.loss / n_micro).backward()
                step_loss += out.loss.item() / n_micro
                del enc, out
            opt.step()
            if sched:
                sched.step()
            opt.zero_grad()
            step += 1
            losses.append(step_loss)
            gc.collect(); torch.cuda.empty_cache()
            if step % 100 == 0:
                log(f"    step {step}/{total} loss(mean last 50)={sum(losses[-50:])/len(losses[-50:]):.4f} "
                    f"elapsed={(time.time()-t0)/60:.1f}m")
            if CKPT_EVERY and step % CKPT_EVERY == 0 and step < total:
                save_ckpt(m, opt, sched, step, losses, cdir)
        epoch += 1
    if CKPT_EVERY and start_step < total:                  # (not when resumed at the end: rewriting it would look like progress)
        save_ckpt(m, opt, sched, total, losses, cdir)      # final weights: an eval-phase kill never retrains
    return m, tok, trainable, losses


def load_for_eval_only(c):
    """adapter_dir set => skip training entirely: load the base model + the saved PEFT adapter from a prior save_adapter=1
    cell, then evaluate() only. c['eval_arm'] (defaults to c['arm']) is the arm whose TEST set to score against -- for a
    cross-arm run this is deliberately the OTHER arm from the one the adapter was trained on; c['source_arm'] (informational
    only, written into the result JSON) should record which arm/cell the adapter came from."""
    from peft import PeftModel
    seed = c["seed"]
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    name = model_name(c)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = torch.bfloat16 if c["dtype"] == "bf16" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(name, dtype=dt).to(DEV)
    m = PeftModel.from_pretrained(base, c["adapter_dir"]).to(DEV)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)  # 0 (frozen), kept for schema parity
    log(f"    eval_only: loaded adapter from {c['adapter_dir']} -- scoring on arm={c.get('eval_arm') or c['arm']} "
        f"(source_arm={c.get('source_arm')})")
    return m, tok, trainable, []

@torch.no_grad()
def answer_nll(m, tok, rows):
    tot, n = 0.0, 0
    for r in rows:
        h = tok(render_human(r["conversations"]), truncation=True, max_length=MAX_LEN)["input_ids"]
        f = tok(render_full(r["conversations"]), return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEV)
        lab = f["input_ids"].clone()
        lab[:, :len(h)] = -100
        if (lab != -100).sum() == 0:
            continue
        tot += m(**f, labels=lab).loss.item(); n += 1
    return tot / max(n, 1)


@torch.no_grad()
def generate_raw(m, tok, rows, bs):
    """greedy 16-token generations for each row's human prompt; bs=1 is the original P14/P22 harness, bs>1 batches
    length-sorted prompts with LEFT padding (HF derives position_ids from the attention mask, so positions are unchanged)."""
    raws = [None] * len(rows)
    if bs <= 1:
        for k, r in enumerate(rows):
            enc = tok(render_human(r["conversations"]), return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEV)
            gen = m.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
            raws[k] = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            if (k + 1) % 2000 == 0:
                log(f"    eval {k+1}/{len(rows)}")
        return raws
    tok.padding_side = "left"
    order = sorted(range(len(rows)), key=lambda i: len(render_human(rows[i]["conversations"])))
    for n, s0 in enumerate(range(0, len(order), bs)):
        idxs = order[s0:s0 + bs]
        enc = tok([render_human(rows[i]["conversations"]) for i in idxs], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(DEV)
        gen = m.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idxs):
            raws[i] = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        del enc, gen
        # 2026-10-03 FIX (found via memwatch): prompts are processed sorted short->long (CICIDS 400..1071 tokens). Every batch needs
        # slightly bigger buffers than any cached block, so the caching allocator kept all the too-small blocks and the process's
        # reserved GPU memory ramped to 56-84 GB near the longest batches (batch 400+/625) -> kernel OOM-kill of 19 jobs in 10 h.
        # UNSW (442..466 tokens) never showed it. Releasing the cache every batch keeps eval at ~10-20 GB.
        torch.cuda.empty_cache()
        if (n + 1) % 50 == 0:
            log(f"    eval batches {n+1}/{(len(order)+bs-1)//bs} reserved={torch.cuda.memory_reserved()/1e9:.1f}GB")
    tok.padding_side = "right"
    return raws


@torch.no_grad()
def evaluate(m, tok, c, name, val_rows):
    m.eval()
    if val_rows is not None:
        rows = val_rows
    else:
        score_arm = c.get("eval_arm") or c["arm"]
        test = json.load(open(f"data/salad2_{score_arm}/salad2_test.json"))
        sel = list(range(len(test)))
        if c["eval_limit"]:
            sel = random.Random(0).sample(sel, c["eval_limit"])
        rows = [test[j] for j in sel]
    raws = generate_raw(m, tok, rows, c.get("eval_bs", 1))
    if c.get("eval_bs", 1) > 1:
        chk = rows[:100]
        r1 = generate_raw(m, tok, chk, 1)
        same = sum(a == b for a, b in zip(r1, raws[:100]))
        # label-level agreement on the same rows: a raw-text difference that leaves every parsed label unchanged cannot move F1
        lab_same = sum(parse_label(a) == parse_label(b) for a, b in zip(r1, raws[:100]))
        # raws[:100] came from batches over the whole set; identical strings expected (validated 600/600 on UNSW 0.5B)
        CHECK[name] = {"rows": len(chk), "identical_raw_vs_batch1": same, "identical_label_vs_batch1": lab_same}
        log(f"    eval_bs self-check: {same}/{len(chk)} identical to batch-1 generation ({lab_same}/{len(chk)} identical parsed labels)")
    gold, pred = [], []
    with open(f"{OUT}/{name}.preds.jsonl", "w") as pf:
        for k, (r, raw) in enumerate(zip(rows, raws)):
            gt = parse_label({x["from"]: x["value"] for x in r["conversations"]}.get("gpt", ""))
            p = parse_label(raw)
            gold.append(gt); pred.append(p)
            pf.write(json.dumps({"k": k, "gold": gt, "pred": p, "raw": raw}) + "\n")
    labels = sorted(set(gold))
    strict = f1_score(gold, pred, labels=labels, average="macro", zero_division=0)
    per = f1_score(gold, pred, labels=labels, average=None, zero_division=0)
    off = sum(1 for p in pred if p not in labels) / len(pred)
    nll = answer_nll(m, tok, rows) if val_rows is not None else None
    return strict, dict(zip(labels, [round(float(x), 4) for x in per])), off, len(gold), nll


def main():
    queue = json.load(open(sys.argv[1]))
    for n, c in enumerate(queue, 1):
        c = {**DEFAULTS, **c}
        if c["alpha"] is None:
            c["alpha"] = 2 * c["rank"]
        name = c["name"]
        rp = f"{OUT}/{name}.json"
        if os.path.exists(rp):
            try:
                ok = json.load(open(rp)).get("name") == name
            except Exception:
                ok = False
            if ok:
                log(f"[{n}/{len(queue)}] {name}: SKIP (exists)"); continue
            log(f"[{n}/{len(queue)}] {name}: result file exists but is not valid JSON for this cell -- re-running")
        try:
            resolve_tuned(c)
            log(f"[{n}/{len(queue)}] {name}: START {json.dumps(c)}")
            t0 = time.time()
            if c["adapter_dir"]:
                # eval-only cells score a test set, unless eval_val=1: then the same val hold-out (c["val"] rows, seed 1234)
                # the adapter's own tuning cell was scored on -- used to re-score one adapter at a different eval_bs
                val_rows = split_data(c)[1] if c.get("eval_val") else None
                m, tok, trainable, losses = load_for_eval_only(c)
            else:
                data, val_rows = split_data(c)
                m, tok, trainable, losses = train_one(c, data)
            ttrain = time.time() - t0
            t1 = time.time()
            strict, per, off, ntest, nll = evaluate(m, tok, c, name, val_rows)
            teval = time.time() - t1
            res = {"name": name, "config": c, "model_id": model_name(c), "trainable_params": trainable,
                   "eval_only": bool(c["adapter_dir"]), "source_arm": c.get("source_arm"),
                   "scored_arm": c.get("eval_arm") or c["arm"],
                   "strict_macro_f1": round(strict, 4), "per_class_f1": per, "off_vocab_rate": round(off, 4),
                   "n_scored": ntest, "scored_on": "val" if val_rows is not None else "test",
                   "val_answer_nll": None if nll is None else round(nll, 4),
                   "train_seconds": round(ttrain, 1), "eval_seconds": round(teval, 1),
                   "loss_first50": round(sum(losses[:50]) / max(1, len(losses[:50])), 4),
                   "loss_last50": round(sum(losses[-50:]) / max(1, len(losses[-50:])), 4),
                   "eval_bs": c["eval_bs"], "eval_bs_check": CHECK.get(name), "resumed_from_step": RESUMED.get(name, 0),
                   "device": DEV, "torch": torch.__version__, "host": os.uname().nodename}
            if c["save_adapter"] and not c["adapter_dir"]:
                m.save_pretrained(f"{OUT}/adapters/{name}")
            json.dump(res, open(rp, "w"), indent=1)
            shutil.rmtree(f"{CKPT_ROOT}/{name}", ignore_errors=True)
            log(f"[{n}/{len(queue)}] {name}: DONE train={ttrain/60:.1f}m eval={teval/60:.1f}m "
                f"strict_f1={strict:.4f} off_vocab={off:.4f} nll={nll} loss {res['loss_first50']}->{res['loss_last50']}")
            del m, tok
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            log(f"[{n}/{len(queue)}] {name}: FAILED ({e!r}) -- continuing")
            gc.collect(); torch.cuda.empty_cache()
    log("queue complete")


if __name__ == "__main__":
    main()
