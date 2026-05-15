import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd

_PHASE2      = Path(__file__).parents[2]
_LABELED_SVC = _PHASE2 / "Data" / "processed" / "1M_parts_numbers_labeled.csv"
_OUT_PATH    = _PHASE2 / "Data" / "processed" / "unknown_for_training.csv"


def main():
    if not _LABELED_SVC.exists():
        print(f"LinearSVC labeled file not found: {_LABELED_SVC}")
        print("Run pipeline/predictor.py first.")
        return

    print(f"Reading {_LABELED_SVC.name}...")
    df = pd.read_csv(_LABELED_SVC, dtype=str)
    total = len(df)

    label_counts = df["label"].value_counts()
    print(f"\nFull label distribution ({total:,} rows):")
    for label, cnt in label_counts.items():
        print(f"  {label:20s}  {cnt:>10,}  ({cnt/total*100:.2f}%)")

    # Only take confident unknown_article — exclude manual_check (borderline cases)
    unknown_df = df[df["label"] == "unknown_article"][["article"]].reset_index(drop=True)
    print(f"\nFiltered to unknown_article: {len(unknown_df):,} rows")
    print(f"Excluding manual_check ({label_counts.get('manual_check', 0):,} rows) — borderline, keep clean")

    unknown_df.to_csv(_OUT_PATH, index=False)
    print(f"\nSaved → {_OUT_PATH}")
    print(f"Ready for TopBFM training. Sample 300k from this file in train_topbfm.py.")


if __name__ == "__main__":
    main()
