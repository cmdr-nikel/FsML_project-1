import sys
import pandas as pd
import joblib
import datetime
import os
import json
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sklearn.preprocessing import StandardScaler
from Scripts.features_scr.embedder import ArticleEmbedder
from Scripts.models_scr.unsuprv import TopBFM
from Scripts.features_scr.atomar import BMW_RE, VAG_RE, MB_PREFIXES


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data", "original"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Models", "topbfm"))
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Reports"))


def print_cluster_distribution(bfm, X_emb, y_true):
    cluster_ids = bfm.model.predict(X_emb)
    for cluster_id in sorted(set(cluster_ids)):
        mask = cluster_ids == cluster_id
        brands_in_cluster = [y_true[i] for i, m in enumerate(mask) if m]
        counter = Counter(brands_in_cluster)
        total = sum(counter.values())
        label = bfm.cluster_labels.get(cluster_id, "unknown")
        print(f"Cluster {cluster_id:3d} -> [{str(label):15s}] | total: {total:6d} | {counter}")


def save_cluster_distribution(bfm, X_emb, y_true):
    os.makedirs(LOG_DIR, exist_ok=True)
    cluster_ids = bfm.model.predict(X_emb)
    cluster_data = {}
    for cid in sorted(set(cluster_ids)):
        mask = [i for i, c in enumerate(cluster_ids) if c == cid]
        brands_in = [y_true[i] for i in mask]
        cluster_data[str(cid)] = {
            "label": bfm.cluster_labels.get(cid, "unknown_article"),
            "counts": dict(Counter(brands_in)),
            "size": len(mask)
        }
    with open(os.path.join(LOG_DIR, "cluster_distribution.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, ensure_ascii=False, indent=2)


def main():
    mb = pd.read_csv(os.path.join(DATA_DIR, "mercedes-benz 300k.txt"), header=None, names=["article"]).assign(
        brand="mercedes")
    bmw = pd.read_csv(os.path.join(DATA_DIR, "BMW 300k.csv"), sep="\t", header=None, names=["brand", "article", "alt"],
                      usecols=["article"]).assign(brand="bmw")
    vag = pd.read_csv(os.path.join(DATA_DIR, "VAG 300k.csv"), sep="\t", usecols=["number"]).rename(
        columns={"number": "article"}).assign(brand="vag")

    df = pd.concat([mb, bmw, vag], ignore_index=True).dropna(subset=["article"])
    df["article"] = df["article"].astype(str).str.strip()
    df = df.groupby('brand').sample(n=df['brand'].value_counts().min(), random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        df["article"].tolist(), df["brand"].tolist(), test_size=0.3, stratify=df["brand"], random_state=42
    )

    def get_article_features(article_list):
        df_feat = pd.DataFrame({'article': article_list})
        df_feat['len'] = df_feat['article'].str.len()
        df_feat['is_bmw'] = df_feat['article'].apply(lambda x: 1 if BMW_RE.match(str(x)) else 0)
        df_feat['is_vag'] = df_feat['article'].apply(lambda x: 1 if VAG_RE.match(str(x)) else 0)
        df_feat['is_mb'] = df_feat['article'].apply(lambda x: 1 if str(x)[0] in MB_PREFIXES else 0)
        return df_feat[['len', 'is_bmw', 'is_vag', 'is_mb']].values

    def apply_feature_weights(X_final, n_flags=3, weight=2.0):
        X_final[:, -n_flags:] *= weight
        return X_final

    X_train_struct, X_test_struct = get_article_features(X_train), get_article_features(X_test)
    embedder = ArticleEmbedder()
    X_train_emb_base = embedder.fit_transform(X_train)
    X_test_emb_base = embedder.transform(X_test)

    X_train_full = np.hstack([X_train_emb_base, X_train_struct])
    X_test_full = np.hstack([X_test_emb_base, X_test_struct])

    n_flags = 3
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full[:, :-n_flags])
    X_test_scaled = scaler.transform(X_test_full[:, :-n_flags])

    X_train_final = np.hstack([X_train_scaled, X_train_full[:, -n_flags:]])
    X_test_final = np.hstack([X_test_scaled, X_test_full[:, -n_flags:]])

    n_brands = len(set(y_train))
    n_clusters = 100 + n_brands * 100
    X_train_final = apply_feature_weights(X_train_final, n_flags=n_flags, weight=2.0)
    X_test_final = apply_feature_weights(X_test_final, n_flags=n_flags, weight=2.0)

    bfm = TopBFM(n_clusters=n_clusters)
    bfm.fit(X_train_final)
    bfm.label_clusters(X_train_final, y_train)

    preds = bfm.predict(X_test_final)
    report = classification_report(y_test, preds, zero_division=0)
    print(report)

    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"topbfm_{timestamp}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"n_clusters: {n_clusters}\nreport:\n{report}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(embedder, os.path.join(MODEL_DIR, "embedder.pkl"))
    joblib.dump(bfm, os.path.join(MODEL_DIR, "topbfm.pkl"))
    save_cluster_distribution(bfm, X_train_final, y_train)

    with open(os.path.join(LOG_DIR, "cluster_distribution.json"), "r") as f:
        data = json.load(f)
    dirtiest = min([item for item in data.items() if item[1]['size'] > 10],
                   key=lambda x: max(x[1]['counts'].values()) / x[1]['size'])

    def debug_cluster(bfm, X_emb, y_true, cluster_id, X_train, log_file_path):
        indices = [i for i, cid in enumerate(bfm.model.predict(X_emb)) if cid == cluster_id]
        output = f"\n--- Auto: worst cluster {cluster_id} ---\n" + "\n".join(
            [f"Art: {X_train[i]}, Brand: {y_true[i]}" for i in indices[:20]])
        print(output)
        with open(log_file_path, "a", encoding="utf-8") as f: f.write(output)

    debug_cluster(bfm, X_train_final, y_train, int(dirtiest[0]), X_train, log_path)

    def process_huge_file(input_file, embedder, bfm, scaler, output_file=None):
        if output_file is None:
            output_file = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "Data", "processed", "1M_parts_numbers_labeled_topbfm.csv")
            )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        chunk_size = 100000
        reader = pd.read_csv(input_file, chunksize=chunk_size, header=None, names=["article"])

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("article,label,mb_prob,bmw_prob,vag_prob,comment\n")

            for chunk in reader:
                arts = chunk["article"].astype(str).tolist()

                struct = get_article_features(arts)
                emb = embedder.transform(arts)

                full = np.hstack([emb, struct])
                to_scale = full[:, :-3]
                flags = full[:, -3:]
                scaled = scaler.transform(to_scale)
                final = apply_feature_weights(np.hstack([scaled, flags]), n_flags=3, weight=2.0)

                preds = bfm.predict(final)

                results = []
                for i, art in enumerate(arts):
                    if struct[i, 1:].sum() == 0:
                        results.append([art, "unknown_article", None, None, None,
                                        "unknown format: not matching any known brand patterns"])
                    else:
                        results.append(
                            [art, preds[i], 0.0, 0.0, 1.0, "confident: " + preds[i]])

                pd.DataFrame(results,
                             columns=["article", "label", "mb_prob", "bmw_prob", "vag_prob", "comment"]).to_csv(f, header=False, index=False)
                print(f"Processed chunk, current count: {len(arts)}(who could have guessed)")

    print("Starting 1M processing...")
    process_huge_file(os.path.join(DATA_DIR, "1M_parts_numbers.csv"), embedder, bfm, scaler)


if __name__ == "__main__":
    main()

