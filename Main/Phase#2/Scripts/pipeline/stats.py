import sys
from pathlib import Path
if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).parents[2]))

import json
import joblib
import numpy as np
import pandas as pd

_PHASE2        = Path(__file__).parents[2]
_LINEARSVC_DIR = _PHASE2 / "Models" / "linearcvs"
_TOPBFM_DIR    = _PHASE2 / "Models" / "topbfm"
_LABELED_SVC   = _PHASE2 / "Data" / "processed" / "1M_parts_numbers_labeled.csv"
_LABELED_TOPBFM = _PHASE2 / "Data" / "processed" / "1M_parts_numbers_labeled_topbfm.csv"
_CLUSTER_JSON  = _PHASE2 / "Reports" / "cluster_distribution.json"
_REPORTS_DIR   = _PHASE2 / "Reports"


def _hr(title=""):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (w - pad - len(title) - 2)}")
    else:
        print("─" * w)


def _label_distribution(df, title):
    total = len(df)
    _hr(title)
    print(f"  Total articles : {total:,}")
    _hr("LABEL DISTRIBUTION")
    counts = df["label"].value_counts()
    for label, cnt in counts.items():
        bar = "█" * int(cnt / total * 40)
        print(f"  {label:20s}  {cnt:>10,}  {cnt/total*100:6.2f}%  {bar}")
    return df, total, counts


def _confidence_table(df, prob_cols, counts):
    _hr("CONFIDENCE (max brand prob) BY LABEL")
    print(f"  {'label':20s}  {'mean':>6}  {'p50':>6}  {'p10':>6}  {'p90':>6}  {'min':>6}  {'max':>6}")
    for label in counts.index:
        sub = df[df["label"] == label][prob_cols].apply(pd.to_numeric, errors="coerce")
        if sub.empty:
            continue
        max_prob = sub.max(axis=1).dropna()
        if len(max_prob) == 0:
            continue
        print(
            f"  {label:20s}"
            f"  {max_prob.mean():6.3f}"
            f"  {max_prob.median():6.3f}"
            f"  {max_prob.quantile(0.10):6.3f}"
            f"  {max_prob.quantile(0.90):6.3f}"
            f"  {max_prob.min():6.3f}"
            f"  {max_prob.max():6.3f}"
        )


# ─────────────────────────────────────────────
# LinearSVC
# ─────────────────────────────────────────────

def linearsvc_stats():
    _hr("LinearSVC  MODEL")
    models = sorted(_LINEARSVC_DIR.glob("*.pkl"))
    if not models:
        print(f"  No model found in {_LINEARSVC_DIR}")
        print("  Run pipeline/classic.py first.")
        return

    path = models[-1]
    bundle = joblib.load(path)
    model  = bundle["model"]
    inner  = model.estimator if hasattr(model, "estimator") else model

    print(f"  File     : {path.name}")
    print(f"  Type     : CalibratedClassifierCV(LinearSVC)")
    print(f"  C        : {inner.C}")
    print(f"  Classes  : {list(model.classes_)}")
    print(f"  Features : {len(bundle['feature_order'])}")
    print(f"  Top-10 features: {bundle['feature_order'][:10]}")

    if not _LABELED_SVC.exists():
        _hr("1M FILE RESULTS")
        print(f"  Labeled file not found: {_LABELED_SVC.name}")
        print("  Run pipeline/predictor.py first.")
        return

    df = pd.read_csv(_LABELED_SVC, dtype=str)
    df, total, counts = _label_distribution(df, f"LinearSVC  ·  {_LABELED_SVC.name}")

    prob_cols = [c for c in df.columns if c.endswith("_prob")]
    if prob_cols:
        _confidence_table(df, prob_cols, counts)

    # manual_check breakdown by comment prefix
    manual = df[df["label"] == "manual_check"]
    if len(manual) > 0:
        _hr("MANUAL_CHECK COMMENT BREAKDOWN")
        prefixes = manual["comment"].str.split("_").str[:2].str.join("_").value_counts().head(10)
        for pfx, cnt in prefixes.items():
            print(f"  {pfx:30s}  {cnt:>8,}")


# ─────────────────────────────────────────────
# TopBFM
# ─────────────────────────────────────────────

def topbfm_stats():
    _hr("TopBFM  TRAINING REPORT")

    reports = sorted(_REPORTS_DIR.glob("topbfm_*.txt"))
    if not reports:
        print("  No training reports found.")
        print("  Run pipeline/train_topbfm.py first.")
    else:
        latest = reports[-1]
        print(f"  File: {latest.name}  (latest of {len(reports)} runs)")
        print()
        for line in latest.read_text(encoding="utf-8").splitlines():
            print(f"  {line}")

    _hr("TopBFM  CLUSTER STATS")
    if not _CLUSTER_JSON.exists():
        print(f"  {_CLUSTER_JSON.name} not found. Run pipeline/train_topbfm.py first.")
    else:
        with open(_CLUSTER_JSON, encoding="utf-8") as f:
            data = json.load(f)

        total_clusters = len(data)
        purities = [
            max(v["counts"].values()) / v["size"]
            for v in data.values() if v["size"] > 0
        ]
        clean = sum(1 for p in purities if p >= 0.92)
        sizes = [v["size"] for v in data.values()]

        print(f"  Total clusters     : {total_clusters}")
        print(f"  Clean (≥0.92)      : {clean}  ({clean/total_clusters*100:.1f}%)")
        print(f"  Dirty (<0.92)      : {total_clusters - clean}  ({(total_clusters-clean)/total_clusters*100:.1f}%)")
        print(f"  Avg purity         : {np.mean(purities):.3f}")
        print(f"  Avg cluster size   : {np.mean(sizes):.0f}  (min {min(sizes)}, max {max(sizes)})")

        from collections import Counter
        label_counts = Counter(v["label"] for v in data.values())
        print(f"\n  Cluster label distribution:")
        for lbl, cnt in label_counts.most_common():
            print(f"    {lbl:20s}  {cnt:>5} clusters")

    _hr("TopBFM  1M FILE RESULTS")
    if not _LABELED_TOPBFM.exists():
        print(f"  Labeled file not found: {_LABELED_TOPBFM.name}")
        print("  Run pipeline/label_large_file.py first.")
        return

    df = pd.read_csv(_LABELED_TOPBFM, dtype=str)
    df, total, counts = _label_distribution(df, f"TopBFM  ·  {_LABELED_TOPBFM.name}")

    prob_cols = [c for c in df.columns if c.endswith("_prob")]
    if prob_cols:
        _confidence_table(df, prob_cols, counts)

    manual = df[df["label"] == "manual_check"]
    if len(manual) > 0:
        _hr("MANUAL_CHECK COMMENT BREAKDOWN")
        prefixes = manual["comment"].str.split("_").str[:2].str.join("_").value_counts().head(10)
        for pfx, cnt in prefixes.items():
            print(f"  {pfx:30s}  {cnt:>8,}")


# ─────────────────────────────────────────────

if __name__ == "__main__":
    linearsvc_stats()
    print()
    topbfm_stats()
    _hr()
