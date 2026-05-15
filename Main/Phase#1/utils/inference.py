import pandas as pd
from .features import build_feature_matrix


def predict_brand(csv_input_path, model, output_path="report.csv",
                  threshold_high=0.95, threshold_low=0.40):
    df = pd.read_csv(csv_input_path)
    X = build_feature_matrix(df)
    probs = model.predict_proba(X)[:, 1]

    def make_label(article, prob):
        pct = round(prob * 100, 2)
        if prob >= threshold_high:
            return f"Article {article}: {pct}% Mercedes - it is Mercedes"
        elif prob <= threshold_low:
            return f"Article {article}: {pct}% Mercedes - not Mercedes"
        else:
            return f"Article {article}: {pct}% Mercedes - better take look at"

    df["prob_mercedes"] = (probs * 100).round(4)
    df["decision"] = df["prob_mercedes"].apply(
        lambda p: "mercedes" if p >= threshold_high * 100
        else ("not_mercedes" if p <= threshold_low * 100 else "manual_review")
    )
    df["label"] = [make_label(row["article"], probs[i])
                   for i, (_, row) in enumerate(df.iterrows())]

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} lines → {output_path}")
    print(df["decision"].value_counts().to_string())

    return df
