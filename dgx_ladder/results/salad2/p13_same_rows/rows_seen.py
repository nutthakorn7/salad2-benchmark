# Replays mlx_lm 0.29.1's real iterate_batches on the real prepared train.jsonl and records which
# dataset indices a run visits. Run on the Mac Mini's mlx_lm venv via stdin (no files written there).
import json, sys, types, numpy as np
from transformers import AutoTokenizer
from mlx_lm.tuner.datasets import load_dataset, CacheDataset
from mlx_lm.tuner import trainer

ARM = sys.argv[1] if len(sys.argv) > 1 else "cicids"
tok = AutoTokenizer.from_pretrained("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
args = types.SimpleNamespace(data=f"data/salad2_{ARM}/mlx", train=True, test=False, hf_dataset=False,
                             mask_prompt=False, prompt_feature="prompt", text_feature="text",
                             completion_feature="completion", chat_feature="messages")
train, valid, test = load_dataset(args, tok)
print("train/valid sizes", len(train), len(valid), file=sys.stderr)
ds = CacheDataset(train)
seen = []
class Rec:
    def __init__(s, d): s.d = d
    def __len__(s): return len(s.d)
    def itemlen(s, i): return s.d.itemlen(i)
    def __getitem__(s, i):
        seen.append(i); return s.d[i]
rec = Rec(ds)
lens = [ds.itemlen(i) for i in range(len(ds))]
print("itemlen distinct values:", sorted(set(lens))[:10], "n distinct", len(set(lens)), file=sys.stderr)
# sorted-by-length order that the real code would use (CacheDataset -> itemlen)
out = {}
for seed in (42, 77, 123):
    seen.clear()
    np.random.seed(seed)             # lora.run(): np.random.seed(args.seed)
    it = trainer.iterate_batches(dataset=CacheDataset(train) if False else rec, batch_size=4, max_seq_length=2048, loop=True, seed=None)
    # rec is not a CacheDataset instance -> len_fn would differ; so re-do with a CacheDataset subclass
    break
class RecCache(CacheDataset):
    def __getitem__(s, i):
        seen.append(i); return CacheDataset.__getitem__(s, i)
for seed in (42, 77, 123):
    seen.clear()
    np.random.seed(seed)
    rc = RecCache(train)
    it = trainer.iterate_batches(dataset=rc, batch_size=4, max_seq_length=2048, loop=True, seed=None)
    for _ in range(600): next(it)
    out[str(seed)] = list(map(int, seen))   # 4 indices per step, in order
    print("seed", seed, "steps 600 -> rows", len(seen), "distinct", len(set(seen)),
          "first100 distinct", len(set(seen[:400])), file=sys.stderr)
print(json.dumps(out))
