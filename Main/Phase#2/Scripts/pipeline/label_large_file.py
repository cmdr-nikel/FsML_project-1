import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import joblib
import numpy as np
import pandas as pd

import os
from Scripts.models.paths import (
    DATA_DIR, MODEL_DIR,
    get_generic_features, get_brand_flags
)
from Scripts.models.validator import validate_article

_FLAG_WEIGHT = float(os.getenv("TOPBFM_FLAG_WEIGHT", "5.0"))


def load_models():
    bfm      = joblib.load(MODEL_DIR / "topbfm.pkl")
    embedder = joblib.load(MODEL_DIR / "embedder.pkl")
    scaler   = joblib.load(MODEL_DIR / "scaler.pkl")
    return bfm, embedder, scaler


def process_huge_file(input_file, embedder, bfm, scaler, output_file=None, confidence_threshold=0.75):
    if output_file is None:
        output_file = Path(__file__).parents[2] / "Data" / "processed" / "1M_parts_numbers_labeled_topbfm.csv"

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_brands = sorted({
        brand
        for info in bfm.cluster_labels.values()
        for brand in info["distribution"]
    })
    prob_cols = [f"{b}_prob" for b in all_brands]

    chunk_size = 100_000
    reader = pd.read_csv(input_file, chunksize=chunk_size, header=None, names=["article"])

    total_rows = 0
    validator_overrides = 0
    low_conf_overrides = 0
    label_counter = {}

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(",".join(["article", "label"] + prob_cols + ["comment"]) + "\n")

        for chunk_num, chunk in enumerate(reader, start=1):
            arts = chunk["article"].astype(str).tolist()

            emb   = embedder.transform(arts)
            gen   = get_generic_features(arts)
            core  = np.hstack([emb, gen])
            scaled_core = scaler.transform(core)
            flags = get_brand_flags(arts) * _FLAG_WEIGHT
            scaled_core = np.hstack([scaled_core, flags])

            labels = bfm.predict_labels(scaled_core)
            probas = bfm.predict_proba(scaled_core)
            results = []
            manual_check_count = 0
            unknown_count = 0

            for art, model_label, proba in zip(arts, labels, probas):
                max_conf = max(proba.values()) if proba else 0.0
                final_label = model_label
                comment = f"model_{model_label}_{max_conf:.2f}"

                val_label, val_comment = validate_article(art, model_label)
                if val_label == "manual_check":
                    final_label = "manual_check"
                    comment = val_comment
                    validator_overrides += 1
                    manual_check_count += 1
                elif model_label == "unknown_article":
                    final_label = "unknown_article"
                    comment = f"unknown_cluster_{max_conf:.2f}"
                    unknown_count += 1
                elif max_conf < confidence_threshold:
                    final_label = "manual_check"
                    comment = f"low_conf_{max_conf:.2f}"
                    low_conf_overrides += 1
                    manual_check_count += 1

                prob_values = [proba.get(b, 0.0) for b in all_brands]
                results.append([art, final_label] + prob_values + [comment])

                label_counter[final_label] = label_counter.get(final_label, 0) + 1
                total_rows += 1

            pd.DataFrame(results, columns=["article", "label"] + prob_cols + ["comment"]).to_csv(
                f, header=False, index=False
            )

            print(
                f"Chunk {chunk_num} done ({len(arts)} rows, "
                f"manual_check={manual_check_count}, unknown={unknown_count})"
            )

    print("\n--- Distribution of Labels ---")
    for label, count in sorted(label_counter.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_rows) * 100 if total_rows else 0.0
        print(f"{label:15s} {pct:8.6f}")

    manual_check_pct = (label_counter.get("manual_check", 0) / total_rows * 100) if total_rows else 0.0
    print(f"\nManual_check: {manual_check_pct:.2f}% (threshold={confidence_threshold})")

    print("\n--- Statistics ---")
    print(f"Total rows: {total_rows:,}")
    print(f"Validator overrides: {validator_overrides} ({(validator_overrides / total_rows * 100) if total_rows else 0:.2f}%)")
    print(f"Low confidence overrides: {low_conf_overrides} ({(low_conf_overrides / total_rows * 100) if total_rows else 0:.2f}%)")
    print(f"Output written to {output_file}")


def main():
    bfm, embedder, scaler = load_models()
    input_file = DATA_DIR / "1M_parts_numbers.csv"
    #TUNE
    process_huge_file(input_file, embedder, bfm, scaler, confidence_threshold=0.75)


if __name__ == "__main__":
    main()
