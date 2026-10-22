"""Command-line entry point.

    cd <repo>/scripts && python3 -m dgx_analysis <subcommand> [options]
    python3 <repo>/scripts/dgx_analysis/cli.py <subcommand> [options]

Subcommand modules are discovered, not hard-wired: every module in SUBCOMMAND_MODULES that exists is imported and its
`register(subparsers)` is called (see the module protocol at the top of common.py). A module that is absent is skipped
silently; a module that exists but fails to import is reported on stderr and skipped, so one broken module never takes
the other subcommands down. `all` runs every module's optional `run_default()` and returns the worst exit code.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a script: make `dgx_analysis` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "dgx_analysis"

SUBCOMMAND_MODULES = ["verify", "tuning", "factorial", "paired", "ladder", "p14", "p22", "p6b", "offvocab", "status"]  # status LAST: it reads the others' reports


def _load_modules() -> list:
    mods = []
    for name in SUBCOMMAND_MODULES:
        full = f"dgx_analysis.{name}"
        try:
            mods.append(importlib.import_module(full))
        except ModuleNotFoundError as e:
            if e.name == full:
                continue  # module simply not written yet
            print(f"[dgx_analysis] WARNING: {full} failed to import: {e!r}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[dgx_analysis] WARNING: {full} failed to import: {e!r}", file=sys.stderr)
    return mods


def build_parser(mods: list | None = None) -> argparse.ArgumentParser:
    mods = _load_modules() if mods is None else mods
    ap = argparse.ArgumentParser(prog="dgx_analysis", description="Analyses of the synced DGX SALAD-v2 results.")
    sub = ap.add_subparsers(dest="cmd", metavar="<subcommand>")
    for m in mods:
        if hasattr(m, "register"):
            m.register(sub)
    allp = sub.add_parser("all", help="run every subcommand's default report (worst exit code wins)")
    allp.set_defaults(func=lambda args, _m=mods: _run_all(_m))
    ap.set_defaults(_mods=mods)
    return ap


def _run_all(mods: list) -> int:
    worst = 0
    for m in mods:
        fn = getattr(m, "run_default", None)
        if fn is None:
            continue
        print(f"=== {m.__name__.split('.')[-1]} ===", flush=True)
        try:
            worst = max(worst, int(fn() or 0))
        except Exception as e:  # noqa: BLE001 - keep going, report
            print(f"[dgx_analysis] {m.__name__} run_default crashed: {e!r}", file=sys.stderr)
            worst = max(worst, 2)
    return worst


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 2
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
