import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import datetime
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from Scripts.features.embedder import ArticleEmbedder
from Scripts.models.unsuprv import TopBFM
from Scripts.models.paths import (
    DATA_DIR, MODEL_DIR, LOG_DIR,
    get_generic_features, get_brand_flags
)
from Scripts.data.loads import _normalize

_PHASE2       = Path(__file__).parents[2]
_UNKNOWN_PATH = _PHASE2 / "Data" / "processed" / "unknown_for_training.csv"
_N_PER_CLASS  = 300_000

BRANDS = ["bmw", "honda", "mercedes", "nissan", "toyota", "vag", "mitsubishi"]
tqdm.pandas()


def load_data():
    if not _UNKNOWN_PATH.exists():
        raise FileNotFoundError(
            f"Unknown-article training set not found: {_UNKNOWN_PATH}\n"
            "Run pipeline/predictor.py → pipeline/filter_unknown.py first."
        )

    mb = pd.read_csv(
        DATA_DIR / "mercedes-benz 300k.txt",
        header=None, names=["article"],
    ).assign(brand="mercedes")
    bmw = pd.read_csv(
        DATA_DIR / "BMW 300k.csv",
        sep="\t", header=None, names=["brand", "article", "alt"], usecols=["article"],
    ).assign(brand="bmw")
    vag = pd.read_csv(
        DATA_DIR / "VAG 300k.csv",
        sep="\t", usecols=["number"],
    ).rename(columns={"number": "article"}).assign(brand="vag")

    # --- Japanese brands (part number = column index 1 in every file) ---
    toyota = pd.read_csv(DATA_DIR / "toyota.csv", dtype=str, on_bad_lines="skip").iloc[:, 1].to_frame(name="article").assign(brand="toyota")
    honda = pd.read_csv(DATA_DIR / "honda.csv", header=None, dtype=str, on_bad_lines="skip").iloc[:, 1].to_frame(name="article").assign(brand="honda")
    mitsu = pd.read_csv(DATA_DIR / "Auvika_MITSUBISHI.csv", sep=";", header=None, dtype=str, on_bad_lines="skip").iloc[:, 1].to_frame(name="article").assign(brand="mitsubishi")
    nissan = pd.read_csv(DATA_DIR / "Price NISSAN_AE.txt", sep="\t", dtype=str, encoding="utf-8-sig", on_bad_lines="skip").iloc[:, 1].to_frame(name="article").assign(brand="nissan")

    unknown = pd.read_csv(
        _UNKNOWN_PATH, dtype=str,
    ).sample(n=_N_PER_CLASS, random_state=42).assign(brand="unknown_article")

    df = pd.concat(
        [mb, bmw, vag, toyota, honda, mitsu, nissan, unknown], ignore_index=True
    ).dropna(subset=["article"])
    df["article"] = _normalize(df["article"])
    df = df[df["article"].str.contains(r"\d", regex=True)]
    df = df.drop_duplicates(subset=["article"])
    # Balance per class; cap at availability so a short class can't crash sample().
    # Explicit loop (not groupby.apply): pandas>=2.2 drops the grouping column
    # inside apply, which would silently delete 'brand'.
    parts = [
        g.sample(n=min(_N_PER_CLASS, len(g)), random_state=42)
        for _, g in df.groupby("brand")
    ]
    df = pd.concat(parts, ignore_index=True)
    print(f"Training set: {len(df):,} articles  |  {df['brand'].value_counts().to_dict()}")
    return df


# Number of trailing get_generic_features columns that are positional brand
# discriminators (num_letters, first_letter_pos, ends_z_letter, ends_two_letters).
# The leading 2 (len, digit_ratio) are NOT discriminators and stay unweighted.
N_DISCRIMINATORS = 4


def build_features(articles, embedder, scaler, fit=False, flag_weight=5.0, disc_weight=3.0):
    print("  Building embeddings...")
    emb   = embedder.fit_transform(articles) if fit else embedder.transform(articles)
    print("  Building generic features...")
    gen   = get_generic_features(articles)

    core        = np.hstack([emb, gen])
    print("  Scaling core features...")
    scaled_core = scaler.fit_transform(core) if fit else scaler.transform(core)

    # Up-weight the 4 positional discriminators (trailing gen cols) AFTER scaling so
    # they aren't drowned by the 100-dim char embedding in the clustering distance.
    # These already exist in atomar.py (teacher) but reach the student weight-1 → at
    # weight 1 honda's strong binary markers registered but nissan's subtle single-
    # letter signal did not. disc_weight gives them clustering pull. len/digit_ratio
    # (the leading 2 gen cols) are left untouched.
    if disc_weight != 1.0:
        scaled_core[:, -N_DISCRIMINATORS:] *= disc_weight

    # Brand flags added AFTER scaling so StandardScaler doesn't neutralise them.
    # flag_weight controls how strongly brand identity dominates over char similarity.
    print("  Adding brand flags...")
    flags = get_brand_flags(articles) * flag_weight
    return np.hstack([scaled_core, flags])


def save_cluster_distribution(bfm, X_emb, y_true):
    from collections import Counter
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print("  Predicting cluster IDs for distribution report...")
    cluster_ids = bfm.model.predict(X_emb)
    cluster_data = {}
    print("  Aggregating cluster data...")
    for cid in tqdm(sorted(set(cluster_ids))):
        indices = [i for i, c in enumerate(cluster_ids) if c == cid]
        brands_in = [y_true[i] for i in indices]
        cluster_data[str(cid)] = {
            "label": bfm.cluster_labels.get(cid, {}).get("label", "unknown_article"),
            "counts": dict(Counter(brands_in)),
            "size": len(indices),
        }
    with open(LOG_DIR / "cluster_distribution.json", "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, ensure_ascii=False, indent=2)


def main():
    import os
    purity_threshold = float(os.getenv("TOPBFM_PURITY_THRESHOLD", "0.92"))
    flag_weight      = float(os.getenv("TOPBFM_FLAG_WEIGHT",      "5.0"))
    clusters_per_class = int(os.getenv("TOPBFM_CLUSTERS_PER_CLASS", "200"))
    disc_weight      = float(os.getenv("TOPBFM_DISC_WEIGHT",      "3.0"))

    print("--- Block 1: Load and split data ---")
    t0 = time.time()
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["article"].tolist(), df["brand"].tolist(),
        test_size=0.4, stratify=df["brand"], random_state=42,
    )
    print(f"Block 1 done in {time.time() - t0:.2f}s.")

    n_clusters = 150 + len(set(y_train)) * clusters_per_class
    print(f"Hyperparams: n_clusters={n_clusters}  purity={purity_threshold}  flag_weight={flag_weight}  disc_weight={disc_weight}")

    embedder = ArticleEmbedder()
    scaler   = StandardScaler()

    print("\n--- Block 2: Feature extraction ---")
    t0 = time.time()
    print("Building features for TRAIN set...")
    X_train_core = build_features(X_train, embedder, scaler, fit=True,  flag_weight=flag_weight, disc_weight=disc_weight)
    print("Building features for TEST set...")
    X_test_core  = build_features(X_test,  embedder, scaler, fit=False, flag_weight=flag_weight, disc_weight=disc_weight)
    print(f"Block 2 done in {time.time() - t0:.2f}s.")

    print("\n--- Block 3: TopBFM Training ---")
    t0 = time.time()
    bfm = TopBFM(n_clusters=n_clusters, purity_threshold=purity_threshold, verbose=True)

    print("Fitting TopBFM (MiniBatchKMeans)...")
    bfm.fit(X_train_core)
    print("Labeling clusters...")
    bfm.label_clusters(X_train_core, y_train)
    print(f"Block 3 done in {time.time() - t0:.2f}s.")

    print("\n--- Block 4: Evaluation ---")
    t0 = time.time()
    preds = bfm.predict_labels(X_test_core)
    report = classification_report(y_test, preds, zero_division=0)
    print(report)
    print(f"Block 4 done in {time.time() - t0:.2f}s.")

    print("\n--- Block 5: Save models and reports ---")
    t0 = time.time()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"topbfm_{timestamp}.txt"
    log_path.write_text(
        f"n_clusters: {n_clusters}\n"
        f"purity_threshold: {purity_threshold}\n"
        f"report:\n{report}",
        encoding="utf-8"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(embedder, MODEL_DIR / "embedder.pkl")
    joblib.dump(scaler,   MODEL_DIR / "scaler.pkl")
    joblib.dump(bfm,      MODEL_DIR / "topbfm.pkl")
    print(f"Models saved to {MODEL_DIR}")

    save_cluster_distribution(bfm, X_train_core, y_train)
    print(f"Cluster distribution saved to {LOG_DIR}/cluster_distribution.json")
    print(f"Block 5 done in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()