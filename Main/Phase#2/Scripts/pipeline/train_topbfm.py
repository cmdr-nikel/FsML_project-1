import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import datetime
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Scripts.features.embedder import ArticleEmbedder
from Scripts.models.unsuprv import TopBFM
from Scripts.models.paths import (
    DATA_DIR, MODEL_DIR, LOG_DIR,
    get_generic_features, get_brand_flags
)

_PHASE2       = Path(__file__).parents[2]
_UNKNOWN_PATH = _PHASE2 / "Data" / "processed" / "unknown_for_training.csv"
_N_PER_CLASS  = 300_000


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
    unknown = pd.read_csv(
        _UNKNOWN_PATH, dtype=str,
    ).sample(n=_N_PER_CLASS, random_state=42).assign(brand="unknown_article")

    df = pd.concat([mb, bmw, vag, unknown], ignore_index=True).dropna(subset=["article"])
    df["article"] = df["article"].astype(str).str.strip()
    df = df.groupby("brand").sample(n=_N_PER_CLASS, random_state=42)
    print(f"Training set: {len(df):,} articles  |  {df['brand'].value_counts().to_dict()}")
    return df


def build_features(articles, embedder, scaler, fit=False, flag_weight=5.0):
    emb   = embedder.fit_transform(articles) if fit else embedder.transform(articles)
    gen   = get_generic_features(articles)

    core        = np.hstack([emb, gen])
    scaled_core = scaler.fit_transform(core) if fit else scaler.transform(core)

    # Brand flags added AFTER scaling so StandardScaler doesn't neutralise them.
    # flag_weight controls how strongly brand identity dominates over char similarity.
    flags = get_brand_flags(articles) * flag_weight
    return np.hstack([scaled_core, flags])


def save_cluster_distribution(bfm, X_emb, y_true):
    from collections import Counter
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cluster_ids = bfm.model.predict(X_emb)
    cluster_data = {}
    for cid in sorted(set(cluster_ids)):
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

    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["article"].tolist(), df["brand"].tolist(),
        test_size=0.4, stratify=df["brand"], random_state=42,
    )

    n_clusters = 150 + len(set(y_train)) * 100  # 4 classes → 550 clusters
    print(f"Hyperparams: n_clusters={n_clusters}  purity={purity_threshold}  flag_weight={flag_weight}")

    embedder = ArticleEmbedder()
    scaler   = StandardScaler()

    X_train_core = build_features(X_train, embedder, scaler, fit=True,  flag_weight=flag_weight)
    X_test_core  = build_features(X_test,  embedder, scaler, fit=False, flag_weight=flag_weight)

    bfm = TopBFM(n_clusters=n_clusters, purity_threshold=purity_threshold)

    bfm.fit(X_train_core)
    bfm.label_clusters(X_train_core, y_train)

    preds = bfm.predict_labels(X_test_core)
    report = classification_report(y_test, preds, zero_division=0)
    print(report)

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


if __name__ == "__main__":
    main()
