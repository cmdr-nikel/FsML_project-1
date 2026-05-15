"""
Phase#2 pipeline runner.

Modes (mutually exclusive):
  --linear   Train LinearSVC + label 1M + stats        (steps 1-2-6)
  --topbfm   Filter unknown + train TopBFM + label 1M + stats  (steps 3-4-5-6)
  --full     Complete end-to-end run                   (steps 1-2-3-4-5-6)  [default]
  --from N   Start from step N to the end
  --only N   Run exactly one step

Steps:
  1  classic.py          — train LinearSVC
  2  predictor.py        — label 1M file with LinearSVC
  3  filter_unknown.py   — extract unknown_article training data
  4  train_topbfm.py     — train TopBFM (4 classes)
  5  label_large_file.py — label 1M file with TopBFM
  6  stats.py            — print diagnostics

Examples:
  python run_pipeline.py --linear
  python run_pipeline.py --topbfm
  python run_pipeline.py --full
  python run_pipeline.py --from 4
  python run_pipeline.py --only 6
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

_PIPELINE = Path(__file__).parent
_MODELS   = Path(__file__).parents[1] / "models"

STEPS = {
    1: ("LinearSVC training",        _PIPELINE / "classic.py"),
    2: ("LinearSVC → label 1M",      _MODELS   / "predictor.py"),
    3: ("Filter unknown_article",    _PIPELINE / "filter_unknown.py"),
    4: ("TopBFM training",           _PIPELINE / "train_topbfm.py"),
    5: ("TopBFM → label 1M",         _PIPELINE / "label_large_file.py"),
    6: ("Stats & diagnostics",       _PIPELINE / "stats.py"),
}

MODES = {
    "linear":  [1, 2, 6],
    "topbfm":  [3, 4, 5, 6],
    "full":    [1, 2, 3, 4, 5, 6],
}


def _hr(title="", width=62):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'═' * pad} {title} {'═' * (width - pad - len(title) - 2)}")
    else:
        print("═" * width)


def run_step(num):
    name, script = STEPS[num]
    _hr(f"Step {num}: {name}")
    print(f"  Script : {script.relative_to(Path(__file__).parents[2])}")
    print(f"  Start  : {time.strftime('%H:%M:%S')}")
    t0 = time.time()

    result = subprocess.run([sys.executable, str(script)], cwd=str(script.parent))

    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  Done   : {time.strftime('%H:%M:%S')}  ({elapsed:.0f}s)  [{status}]")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Phase#2 pipeline runner",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--linear", action="store_true",
                       help="LinearSVC full cycle: train → label 1M → stats  (steps 1-2-6)")
    group.add_argument("--topbfm", action="store_true",
                       help="TopBFM full cycle: filter → train → label 1M → stats  (steps 3-4-5-6)")
    group.add_argument("--full",   action="store_true",
                       help="Full end-to-end run  (steps 1-2-3-4-5-6)  [default]")
    group.add_argument("--from",   dest="from_step", type=int, metavar="N",
                       help="Run from step N to the end")
    group.add_argument("--only",   dest="only_step", type=int, metavar="N",
                       help="Run exactly one step")
    args = parser.parse_args()

    if args.linear:
        mode_name, step_nums = "LinearSVC cycle", MODES["linear"]
    elif args.topbfm:
        mode_name, step_nums = "TopBFM cycle", MODES["topbfm"]
    elif args.from_step:
        mode_name = f"from step {args.from_step}"
        step_nums = [n for n in sorted(STEPS) if n >= args.from_step]
    elif args.only_step:
        mode_name = f"step {args.only_step} only"
        step_nums = [args.only_step]
    else:
        mode_name, step_nums = "Full pipeline", MODES["full"]

    invalid = [n for n in step_nums if n not in STEPS]
    if invalid:
        print(f"Unknown step(s): {invalid}")
        sys.exit(1)

    _hr(f"Phase#2 — {mode_name}")
    for i, n in enumerate(step_nums, 1):
        print(f"  [{i}/{len(step_nums)}] {STEPS[n][0]}")

    t_total = time.time()

    for num in step_nums:
        ok = run_step(num)
        if not ok:
            _hr("Summary")
            print(f"  Aborted at step {num}: {STEPS[num][0]}")
            _hr()
            sys.exit(1)

    _hr("Summary")
    print(f"  Done — {len(step_nums)} step(s) in {time.time() - t_total:.0f}s.")
    _hr()


if __name__ == "__main__":
    main()
