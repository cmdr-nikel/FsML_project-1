import pandas as pd
import numpy as np
import os
import joblib
from features_scr.atomar import extract_features_from_article, build_feature_matrix

CONFIDENCE_THRESHOLD = 0.85


class BrandPredictor:

    def __init__(self, model, scaler, threshold=CONFIDENCE_THRESHOLD):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold

    def predict_one(self, article: str) -> dict:
        s = article.strip().upper()
        features = extract_features_from_article(s)

        if not features.get("any_known_pattern", 0):
            return {
                "article": s,
                "decision": "unknown_article",
                "best_brand": None,
                "confidence": None,
                "all_probs": None,
            }

        df = pd.DataFrame([features])
        df_s = self.scaler.transform(df)
        proba = self.model.predict_proba(df_s)[0]
        classes = self.model.classes_

        best_idx = proba.argmax()
        best_brand = classes[best_idx]
        best_prob = round(float(proba[best_idx]), 3)

        return {
            "article": s,
            "decision": best_brand if best_prob >= self.threshold else "manual_check",
            "best_brand": best_brand,
            "confidence": best_prob,
            "all_probs": dict(zip(classes, (proba * 100).round(2))),
        }

    def predict_batch(self, articles) -> pd.DataFrame:
        if not isinstance(articles, pd.Series):
            articles = pd.Series(articles)

        articles = articles.str.strip().str.upper()

        df = pd.DataFrame({"article": articles})
        feature_matrix = build_feature_matrix(df)

        unknown_mask = ~feature_matrix["any_known_pattern"].astype(bool).values

        X_s = self.scaler.transform(feature_matrix)
        proba_matrix = self.model.predict_proba(X_s)
        classes = self.model.classes_

        best_idxs = proba_matrix.argmax(axis=1)
        best_brands = classes[best_idxs]
        best_probs = proba_matrix[range(len(proba_matrix)), best_idxs]

        decisions = np.where(
            unknown_mask,
            "unknown_article",
            np.where(best_probs >= self.threshold, best_brands, "manual_check")
        )

        return pd.DataFrame({
            "article": articles.values,
            "decision": decisions,
            "best_brand": np.where(unknown_mask, None, best_brands),
            "confidence": np.where(unknown_mask, np.nan, best_probs.round(3)),
        })


if __name__ == "__main__":
    MODEL_PATH = os.path.join(
        os.path.dirname(__file__),
        "..", "..",
        "Models",
        "linearcvs",
        "linearsvc_atom_1117k_4cls.pkl"
    )
    bundle = joblib.load(MODEL_PATH)
    predictor = BrandPredictor(
        model=bundle["model"],
        scaler=bundle["scaler"],
        threshold=0.80
    )

    #test
    print(predictor.predict_one("A0001234567"))
    print(predictor.predict_batch(["A0001234567", "11127805048"]))