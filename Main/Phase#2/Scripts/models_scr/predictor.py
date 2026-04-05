import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import joblib
import pandas as pd
import numpy as np

from Scripts.features_scr.atomar import featurize_column

# --- Paths ---
# predict.py is at: Phase#2/Scripts/models_scr/predict.py
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = os.path.join(_BASE, "Models", "linearcvs", "linearsvc_atom_1117k_4cls.pkl")
PROCESSED_DIR  = os.path.join(_BASE, "Data", "processed")

CONFIDENCE_THRESHOLD = 0.85  # range 0.85-0.8 is ideal


def predict(input_path: str, output_path: str = None) -> pd.DataFrame:
    bundle = joblib.load(MODEL_PATH)
    scaler = bundle["scaler"]
    model  = bundle["model"]
    feature_order = bundle["feature_order"]
    print(f"Model loaded: {MODEL_PATH}")

    print(f"Reading input file: {input_path}")
    df = pd.read_csv(input_path, dtype=str)
    print(f"Loaded {len(df):,} rows")

    print("Extracting features...")
    articles = df['article'].str.strip().str.upper().reset_index(drop=True)

    X_f  = featurize_column(articles)
    print(f"Features shape: {X_f.shape}")

    X_f = X_f.reindex(columns=feature_order, fill_value=0).fillna(0)
    print(f"Features fixed to model schema: {X_f.shape}")

    unknown_mask = ~X_f["any_known_pattern"].astype(bool).values

    X_fs = scaler.transform(X_f)

    print("Running predictions...")
    probs = model.predict_proba(X_fs)
    classes = model.classes_
    max_prob = probs.max(axis=1)
    best_class = classes[probs.argmax(axis=1)]

    pred_label = np.where(
        unknown_mask,  # unknown format
        "unknown_article",
        np.where(
            max_prob >= CONFIDENCE_THRESHOLD,  # confidence brand
            best_class,
            "manual_check"  # low confidence brand
        )
    )

    classes = model.classes_
    idx = {c: i for i, c in enumerate(classes)}
    is_unknown = unknown_mask

    result = pd.DataFrame({
        "article":  articles.values,
        "label":    pred_label,
        "mb_prob": np.where(is_unknown, np.nan, probs[:, idx["mercedes"]].round(2)),
        "bmw_prob": np.where(is_unknown, np.nan, probs[:, idx["bmw"]].round(2)),
        "vag_prob": np.where(is_unknown, np.nan, probs[:, idx["vag"]].round(2)),
    })
    result["comment"] = result["label"].apply(
        lambda l:
        "unknown format: not matching any known brand patterns" if l == "unknown_article"
        else "low confidence, review needed" if l == "manual_check"
        else f"confident: {l}"
    )

    total        = len(result)
    n_manual     = (result['label'] == 'manual_check').sum()
    n_bmw        = (result['label'] == 'bmw').sum()
    n_vag        = (result['label'] == 'vag').sum()
    n_mercedes   = (result['label'] == 'mercedes').sum()
    n_unknown = (result['label'] == 'unknown_article').sum()

    print(f"  vag              {n_vag:>10,}  ({n_vag / total * 100:.1f}%)")
    print(f"  bmw              {n_bmw:>10,}  ({n_bmw / total * 100:.1f}%)")
    print(f"  mercedes         {n_mercedes:>10,}  ({n_mercedes / total * 100:.1f}%)")
    print(f"  manual_check     {n_manual:>10,}  ({n_manual / total * 100:.1f}%)")
    print(f"  unknown_article  {n_unknown:>10,}  ({n_unknown / total * 100:.1f}%)")

    if output_path is None:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(PROCESSED_DIR, f"{base_name}_labeled.csv")

    result.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")
    print(
        f"Total: {len(result):,} | "
        f"manual_check: {(result['label'] == 'manual_check').sum():,} | "
        f"unknown_article: {(result['label'] == 'unknown_article').sum():,} | "
        f"threshold: {CONFIDENCE_THRESHOLD}"
    )



if __name__ == "__main__":
    input_path = os.path.join(_BASE, "Data", "original", "1M_parts_numbers.csv")
    predict(input_path)

