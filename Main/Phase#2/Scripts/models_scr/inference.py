import pandas as pd
from features_scr.atomar import extract_features_from_article, build_feature_matrix

class BrandPredictor:

    def __init__(self, model, threshold_high=0.75, threshold_low=0.40):
            self.model = model
            self.threshold_high = threshold_high
            self.threshold_low = threshold_low

    def make_decision(self, best_brand, best_prob):
        if best_prob >= self.threshold_high:
            return best_brand
        elif best_prob <= self.threshold_low:
            return f"possibly_{best_brand}"
        else:
            return "manual review"

    def predict_one(self,article):
        features = extract_features_from_article(article)
        df = pd.DataFrame([features])

        proba = self.model.predict_proba(df)[0]
        classes = self.model.classes

        best_idx = proba.argmax()
        best_brand = classes[best_idx]
        best_prob = round(float(proba[best_idx]), 2)

        return {
            "article": article,
            "decision": self.make_decision(best_brand, proba[best_idx]),
            "best_brand": best_brand,
            "confidence": best_prob,
            "all_probs": dict(zip(classes, (proba * 100).round(2)))
            }

    def predict_batch(self, articles):
        if not isinstance(articles, pd.Series):
            articles = pd.Series(articles)

        df = pd.DataFrame({"article": articles})
        feature_matrix = build_feature_matrix(df)

        proba_matrix = self.model.predict_proba(feature_matrix)
        classes = self.model.classes_

        best_idxs = proba_matrix.argmax(axis=1)
        best_brands = classes[best_idxs]
        best_probs = proba_matrix[range(len(proba_matrix)), best_idxs]

        results = pd.DataFrame({
            "article":    articles.values,
            "best_brand": best_brands,
            "confidence": (best_probs * 100).round(2),
            "decision":   [
                self.make_decision(b, p)
                for b, p in zip(best_brands, best_probs)
            ]
        })

        return results