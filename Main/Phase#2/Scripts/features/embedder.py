from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import joblib


class ArticleEmbedder:
    def __init__(self, ngram_range=(2, 4), n_components=100):
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.vectorizer = None
        self.svd = None

    def fit_transform(self, articles: list[str]) -> np.ndarray:
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=self.ngram_range)
        tfidf_matrix = self.vectorizer.fit_transform(articles)
        self.svd = TruncatedSVD(n_components=self.n_components)
        return self.svd.fit_transform(tfidf_matrix)

    def transform(self, articles: list[str]) -> np.ndarray:
        tfidf_matrix = self.vectorizer.transform(articles)
        return self.svd.transform(tfidf_matrix)

    def save(self, model_dir: Path | str) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "svd": self.svd}, model_dir / "embedder.pkl")

    def load(self, model_dir: Path | str) -> None:
        data = joblib.load(Path(model_dir) / "embedder.pkl")
        self.vectorizer = data["vectorizer"]
        self.svd = data["svd"]
