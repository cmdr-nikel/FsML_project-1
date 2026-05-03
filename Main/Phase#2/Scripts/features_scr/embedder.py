from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import joblib
import os

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
        embeddings = self.svd.fit_transform(tfidf_matrix)
        return embeddings

    def transform(self, articles: list[str]) -> np.ndarray:
        tfidf_matrix = self.vectorizer.transform(articles)
        embeddings = self.svd.transform(tfidf_matrix)
        return embeddings

    def save(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "embedder.pkl")
        joblib.dump({"vectorizer": self.vectorizer, "svd": self.svd}, path)

    def load(self, model_dir: str) -> None:
        path = os.path.join(model_dir, "embedder.pkl")
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.svd = data["svd"]

