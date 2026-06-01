from sklearn.cluster import MiniBatchKMeans
from collections import Counter
import numpy as np
import joblib
import os


class TopBFM:
    def __init__(self, n_clusters=50, purity_threshold=0.75, verbose=False):
        self.n_clusters = n_clusters
        self.purity_threshold = purity_threshold
        self.verbose = verbose
        self.model = None
        self.cluster_labels = {}

    def fit(self, embeddings: np.ndarray) -> None:
        self.model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            verbose=self.verbose,
            batch_size=2048,
            n_init=10,
        )
        self.model.fit(embeddings)

    def label_clusters(self, embeddings, true_labels) -> None:
        if self.model is None:
            raise ValueError("Model is not fitted yet")

        cluster_ids = self.model.predict(embeddings)
        cluster_to_labels = {}

        for cid, label in zip(cluster_ids, true_labels):
            cluster_to_labels.setdefault(cid, []).append(label)

        self.cluster_labels = {}

        for cid, labels in cluster_to_labels.items():
            counter = Counter(labels)
            total = len(labels)
            distribution = {k: v / total for k, v in counter.items()}
            top_label, top_count = counter.most_common(1)[0]
            purity = top_count / total
            final_label = top_label if purity >= self.purity_threshold else "manual_check"

            self.cluster_labels[cid] = {
                "label": final_label,
                "purity": purity,
                "distribution": distribution,
                "size": total
            }

    def predict_labels(self, embeddings: np.ndarray) -> list[str]:
        return self.predict(embeddings, return_full=False)

    def predict_proba(self, embeddings: np.ndarray) -> list[dict]:
        if self.model is None:
            raise ValueError("Model is not fitted yet")
        cluster_ids = self.model.predict(embeddings)
        return [
            self.cluster_labels[cid]["distribution"] if cid in self.cluster_labels else {}
            for cid in cluster_ids
        ]

    def predict(self, embeddings: np.ndarray, return_full=False) -> list:
        if self.model is None:
            raise ValueError("Model is not fitted yet")

        cluster_ids = self.model.predict(embeddings)
        results = []

        for cid in cluster_ids:
            cluster_info = self.cluster_labels.get(cid)

            if cluster_info is None:
                results.append({"label": "unknown_article", "purity": 0.0, "distribution": {}} if return_full else "unknown_article")
                continue

            results.append(cluster_info if return_full else cluster_info["label"])

        return results

    def save(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "topbfm.pkl")
        joblib.dump({"kmeans": self.model, "cluster_labels": self.cluster_labels, "n_clusters": self.n_clusters}, path)

    def load(self, model_dir: str) -> None:
        path = os.path.join(model_dir, "topbfm.pkl")
        data = joblib.load(path)
        self.model = data["kmeans"]
        self.cluster_labels = data["cluster_labels"]
        self.n_clusters = data["n_clusters"]