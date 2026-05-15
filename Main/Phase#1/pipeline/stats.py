import sys
from pathlib import Path
if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).parents[1]))

import joblib
import numpy as np
import pandas as pd

_ROOT         = Path(__file__).parents[1]
_MODEL_PATH   = _ROOT / "models" / "mercedes_model.pkl"
_LABELED_PATH = _ROOT / "output" / "1M_labeled_report.csv"


def _hr(title=""):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (w - pad - len(title) - 2)}")
    else:
        print("─" * w)


def model_stats():
    if not _MODEL_PATH.exists():
        print(f"Model not found at {_MODEL_PATH}")
        print("Run scripts/#1.5 [polygon].py first.")
        return

    m = joblib.load(_MODEL_PATH)

    _hr("MODEL")
    print(f"  Type     : {type(m).__name__}")
    print(f"  Classes  : {m.classes_}  (0=not_mercedes, 1=mercedes)")
    print(f"  C        : {m.C}")
    print(f"  Solver   : {m.solver}")
    print(f"  Features : {m.n_features_in_}")
    print(f"  Fit iters: {m.n_iter_[0]}")

    coefs = m.coef_[0]
    pairs = sorted(zip(m.feature_names_in_, coefs), key=lambda x: abs(x[1]), reverse=True)
    _hr("FEATURE WEIGHTS  (sorted by |coef|, + favours mercedes)")
    for name, coef in pairs:
        bar_len = int(abs(coef) / max(abs(c) for _, c in pairs) * 30)
        bar = ("+" if coef > 0 else "-") * bar_len
        print(f"  {name:35s}  {coef:+.4f}  {bar}")


def inference_stats():
    _hr("1M FILE RESULTS")

    if not _LABELED_PATH.exists():
        print(f"  Labeled file not found: {_LABELED_PATH}")
        print("  Run pipeline/exam.py first.")
        return

    df = pd.read_csv(_LABELED_PATH, dtype=str)
    df["prob_mercedes"] = pd.to_numeric(df["prob_mercedes"], errors="coerce")
    total = len(df)

    print(f"  File    : {_LABELED_PATH.name}")
    print(f"  Total   : {total:,}")

    _hr("LABEL DISTRIBUTION")
    counts = df["decision"].value_counts()
    for label, cnt in counts.items():
        bar = "█" * int(cnt / total * 40)
        print(f"  {label:20s}  {cnt:>10,}  {cnt/total*100:6.2f}%  {bar}")

    _hr("CONFIDENCE (prob_mercedes) BY DECISION")
    print(f"  {'decision':20s}  {'mean':>6}  {'p50':>6}  {'p10':>6}  {'p90':>6}  {'min':>6}  {'max':>6}")
    for label in counts.index:
        sub = df[df["decision"] == label]["prob_mercedes"].dropna()
        if len(sub) == 0:
            continue
        print(
            f"  {label:20s}"
            f"  {sub.mean():6.3f}"
            f"  {sub.median():6.3f}"
            f"  {sub.quantile(0.10):6.3f}"
            f"  {sub.quantile(0.90):6.3f}"
            f"  {sub.min():6.3f}"
            f"  {sub.max():6.3f}"
        )


if __name__ == "__main__":
    model_stats()
    inference_stats()
    _hr()
