"""
Diagnostics for the 1M labeled corpus — both models side by side.

  1. Numeric label distribution per model (counts + %).
  2. LinearSVC × TopBFM agreement + per-problem-class confusion.
  3. Sample collection for problem classes (deduped, random, with TopBFM
     prob columns + article length) → Reports/samples_<class>_disagreements.csv
  4. Cluster analysis: for each problem class, which TopBFM training clusters
     its articles fall into, and why those clusters got their label.

Run:  python Scripts/pipeline/diagnose.py
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd

_PHASE2    = Path(__file__).parents[2]
PROCESSED  = _PHASE2 / "Data" / "processed"
REPORTS    = _PHASE2 / "Reports"
SVC_FILE   = PROCESSED / "1M_parts_numbers_labeled.csv"
BFM_FILE   = PROCESSED / "1M_parts_numbers_labeled_topbfm.csv"
CLUSTERS   = REPORTS / "cluster_distribution.json"

PROBLEM_CLASSES = ["honda", "nissan"]
SAMPLES_PER_BUCKET = 40
SEED = 42


def _load(path, keep_probs=False):
    df = pd.read_csv(path, dtype=str)
    df = df[(df["article"] != "article") & (df["label"] != "label")]
    df = df.dropna(subset=["article", "label"]).reset_index(drop=True)
    return df


def _dist(df, name):
    vc = df["label"].value_counts()
    total = len(df)
    print(f"\n=== {name}: label distribution ({total:,} rows) ===")
    print(f"{'label':<18}{'count':>12}{'pct':>9}")
    for lbl, n in vc.items():
        print(f"{lbl:<18}{n:>12,}{n/total*100:>8.2f}%")


def main():
    svc_full = _load(SVC_FILE)
    bfm_full = _load(BFM_FILE)
    svc = svc_full[["article", "label"]].rename(columns={"label": "svc"})
    # keep BFM prob columns for the sample dump
    prob_cols = [c for c in bfm_full.columns if c.endswith("_prob")]
    bfm = bfm_full[["article", "label"] + prob_cols].rename(columns={"label": "bfm"})

    _dist(svc.rename(columns={"svc": "label"}), "LinearSVC (teacher)")
    _dist(bfm.rename(columns={"bfm": "label"}), "TopBFM (student)")

    merged = svc.merge(bfm, on="article", how="inner")
    print(f"\n=== joined on article: {len(merged):,} rows ===")
    print(f"exact label agreement: {(merged['svc'] == merged['bfm']).mean()*100:.2f}%")

    # ---------------- per-class confusion + samples ----------------
    REPORTS.mkdir(parents=True, exist_ok=True)
    for cls in PROBLEM_CLASSES:
        svc_says = merged[merged["svc"] == cls]
        # dedup on article first so confusion/samples aren't inflated by repeats
        uniq = svc_says.drop_duplicates(subset="article")
        print(f"\n=== {cls}: teacher labeled {len(svc_says):,} rows "
              f"({len(uniq):,} unique) → TopBFM verdict (unique) ===")
        bd = uniq["bfm"].value_counts()
        for lbl, n in bd.items():
            flag = "  <-- agree" if lbl == cls else ""
            print(f"  {lbl:<18}{n:>10,}{n/len(uniq)*100:>8.2f}%{flag}")

        leaked = uniq[uniq["bfm"] != cls].copy()
        leaked["len"] = leaked["article"].str.len()
        # Explicit per-bucket sampling (groupby.apply drops the key col on pandas>=2.2)
        parts = [g.sample(min(SAMPLES_PER_BUCKET, len(g)), random_state=SEED)
                 for _, g in leaked.groupby("bfm")]
        sample = pd.concat(parts, ignore_index=True) if parts else leaked
        keep = ["article", "len", "svc", "bfm"] + prob_cols
        out = REPORTS / f"samples_{cls}_disagreements.csv"
        sample[keep].to_csv(out, index=False)
        print(f"  -> {len(sample)} deduped sample rows → {out.relative_to(_PHASE2)}")

    # ---------------- cluster analysis ----------------
    if CLUSTERS.exists():
        clusters = json.load(open(CLUSTERS))
        print(f"\n=== TopBFM training-cluster analysis ({len(clusters)} clusters) ===")
        for cls in PROBLEM_CLASSES:
            # clusters that actually CONTAIN this brand in training
            holding = []
            for cid, c in clusters.items():
                n = c["counts"].get(cls, 0)
                if n:
                    holding.append((cid, n, c["size"], c["label"], n / c["size"]))
            holding.sort(key=lambda x: x[1], reverse=True)
            total_brand = sum(h[1] for h in holding)
            # how much of this brand's training mass sits in clusters NOT labeled as it
            misrouted = sum(h[1] for h in holding if h[3] != cls)
            print(f"\n  {cls}: appears in {len(holding)} clusters, "
                  f"{total_brand:,} training articles")
            print(f"    {misrouted:,} ({misrouted/total_brand*100:.1f}%) sit in clusters "
                  f"NOT labeled '{cls}' (→ leak to that cluster's label)")
            # destination label breakdown of this brand's training mass
            dest = {}
            for _, n, _, lbl, _ in holding:
                dest[lbl] = dest.get(lbl, 0) + n
            print(f"    training mass by host-cluster label:")
            for lbl, n in sorted(dest.items(), key=lambda x: -x[1]):
                print(f"      {lbl:<18}{n:>8,}{n/total_brand*100:>7.1f}%")
            print(f"    top 5 clusters holding {cls}:")
            print(f"      {'cid':>5} {'n_'+cls:>9} {'size':>7} {'purity':>7}  host_label")
            for cid, n, size, lbl, pur in holding[:5]:
                print(f"      {cid:>5} {n:>9,} {size:>7,} {pur:>7.2f}  {lbl}")
    else:
        print(f"\n[skip cluster analysis] {CLUSTERS} not found")


if __name__ == "__main__":
    main()
