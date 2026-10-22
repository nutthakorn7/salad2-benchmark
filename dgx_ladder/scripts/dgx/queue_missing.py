#!/usr/bin/env python3
"""Print the number of cells of a queue file that have NO VALID result JSON yet (results/attrib/<name>.json). A result
counts only if the file parses as JSON and carries the cell's name: an unclean shutdown (2026-10-17) left a 0-byte
result that the old exists()-only check counted as done. Exit 0 always (prints -1 on an unreadable queue file so the
driver can treat it as unknown). Usage: queue_missing.py <queue.json> [--names]"""
import json, os, sys

out = os.environ.get("A_OUT", "results/attrib")


def valid(path, name):
    try:
        return json.load(open(path)).get("name") == name
    except Exception:
        return False


try:
    q = json.load(open(sys.argv[1]))
except Exception:
    print(-1)
    sys.exit(0)
missing = [c["name"] for c in q if not valid(f"{out}/{c['name']}.json", c["name"])]
if "--names" in sys.argv:
    print("\n".join(missing))
else:
    print(len(missing))
