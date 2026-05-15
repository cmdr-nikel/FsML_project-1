import sys
import joblib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from utils.loaders import load_giga_mixed_fixed
from utils.features import build_feature_matrix
from utils.inference import predict_brand

_ROOT = Path(__file__).parents[1]
_MODELS_DIR = _ROOT / "models"
_OUTPUT_DIR = _ROOT / "output"
_DATA_DIR = _ROOT.parent / "Phase#2" / "Data" / "original"

# Train on the full 600k balanced dataset
train_df = load_giga_mixed_fixed()
X_all = build_feature_matrix(train_df)
y_all = train_df["label"]

# 60% train / 20% val / 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.4, random_state=42, stratify=y_all)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, _MODELS_DIR / "mercedes_model.pkl")

pred_val = model.predict(X_val)
print(f"Val accuracy: {accuracy_score(y_val, pred_val):.3f}")
print(classification_report(y_val, pred_val))

threshold_high = 0.95
threshold_low  = 0.40

probs_val = model.predict_proba(X_val)[:, 1]
print(f"\nMercedes      (>= {threshold_high}): {(probs_val >= threshold_high).sum()}")
print(f"Manual review ({threshold_low}–{threshold_high}): {((probs_val > threshold_low) & (probs_val < threshold_high)).sum()}")
print(f"Not Mercedes  (<= {threshold_low}): {(probs_val <= threshold_low).sum()}")

model = joblib.load(_MODELS_DIR / "mercedes_model.pkl")
predict_brand(
    csv_input_path=_DATA_DIR / "giga_mixed_train_600k.csv",
    model=model,
    output_path=_OUTPUT_DIR / "labeled_report.csv"
)


"""
Feature weights from a reference run:
                   feature    weight
10        prefix_is_mb_set  +7.22  ← main signal: letter A/B/N/C
9               has_prefix  +5.42
4  first_char_is_mb_prefix  +4.29
...
16   possible_truncated_mb  -5.39  ← no letter + 10 digits = likely not MB
5           prefix_letters  -3.87  ← many leading letters = not MB

Val accuracy: 1.000 (perfectly separable with atomic features)
Mercedes      (>= 0.95): 30004
Manual review: 2
Not Mercedes  (<= 0.40): 29994
"""
