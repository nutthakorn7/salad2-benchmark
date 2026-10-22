import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
arm = sys.argv[1]
rows = json.loads(sys.stdin.read())
data = [json.loads(l) for l in open(f"data/salad2_{arm}/mlx/train.jsonl")]
L = {}
def n(i):
    if i not in L: L[i] = len(tok.apply_chat_template(data[i]["messages"]))
    return L[i]
for seed, seq in rows.items():
    print(arm, "seed", seed, {k: sum(n(i) for i in seq[: k * 4]) for k in (10, 50, 100)})
