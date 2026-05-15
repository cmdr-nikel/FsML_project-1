import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import json
import joblib
from collections import Counter

from Scripts.models.paths import MODEL_DIR, LOG_DIR


def load_models():
    bfm      = joblib.load(MODEL_DIR / "topbfm.pkl")
    embedder = joblib.load(MODEL_DIR / "embedder.pkl")
    scaler   = joblib.load(MODEL_DIR / "scaler.pkl")
    return bfm, embedder, scaler


def print_cluster_distribution(bfm, X_emb, y_true):
    cluster_ids = bfm.model.predict(X_emb)
    for cluster_id in sorted(set(cluster_ids)):
        mask = cluster_ids == cluster_id
        brands_in_cluster = [y_true[i] for i, m in enumerate(mask) if m]
        counter = Counter(brands_in_cluster)
        total = sum(counter.values())
        label = bfm.cluster_labels.get(cluster_id, {}).get("label", "unknown")
        print(f"Cluster {cluster_id:3d} -> [{label:15s}] | total: {total:6d} | {counter}")


def find_dirty_clusters(bfm, min_size=10):
    dirty = [
        (cid, info)
        for cid, info in bfm.cluster_labels.items()
        if info["size"] >= min_size and info["label"] == "unknown_article"
    ]
    dirty.sort(key=lambda x: x[1]["purity"])
    return dirty


def debug_cluster(bfm, X_emb, y_true, articles, cluster_id, top_n=20):
    indices = [i for i, cid in enumerate(bfm.model.predict(X_emb)) if cid == cluster_id]
    info = bfm.cluster_labels.get(cluster_id, {})
    print(f"\n--- Cluster {cluster_id} | label: {info.get('label')} | purity: {info.get('purity', 0):.3f} ---")
    for i in indices[:top_n]:
        print(f"  Art: {articles[i]:<20s}  Brand: {y_true[i]}")


def main():
    dist_path = LOG_DIR / "cluster_distribution.json"
    if not dist_path.exists():
        print(f"cluster_distribution.json not found at {LOG_DIR}. Run train_topbfm.py first.")
        return

    with open(dist_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bfm, _, _ = load_models()

    print(f"\nTotal clusters: {len(data)}")

    dirty = [
        (cid, item)
        for cid, item in data.items()
        if item["size"] >= 10 and item["label"] == "unknown_article"
    ]
    dirty.sort(key=lambda x: max(x[1]["counts"].values()) / x[1]["size"])

    print(f"Dirty clusters (purity < threshold, size >= 10): {len(dirty)}")
    print("\nTop 10 dirtiest:")
    for cid, item in dirty[:10]:
        purity = max(item["counts"].values()) / item["size"]
        print(f"  Cluster {cid:>4s} | size: {item['size']:5d} | purity: {purity:.3f} | dist: {item['counts']}")

    if dirty:
        worst_cid, worst_item = dirty[0]
        print(f"\nWorst cluster: {worst_cid} (purity={max(worst_item['counts'].values()) / worst_item['size']:.3f})")
        print("Re-run with embeddings to inspect individual articles (use debug_cluster()).")


if __name__ == "__main__":
    main()
