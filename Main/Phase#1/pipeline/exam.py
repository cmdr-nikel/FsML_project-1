import sys
import pandas as pd
import joblib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.inference import predict_brand

_ROOT = Path(__file__).parents[1]
_DATA_DIR = _ROOT.parent / "Phase#2" / "Data" / "original"
_OUTPUT_DIR = _ROOT / "output"
_MODELS_DIR = _ROOT / "models"

_fixed = _OUTPUT_DIR / "fathers_file_fixed.csv"

# Normalize the source file before processing
# (re-saving through pandas fixes encoding quirks in some CSV exports)
if _fixed.exists():
    _fixed.unlink()

df = pd.read_csv(_DATA_DIR / "1M_parts_numbers.csv")
df.to_csv(_fixed, index=False)
print(f"Lines after format fix: {len(df)}")
print(df.head())

model = joblib.load(_MODELS_DIR / "mercedes_model.pkl")
result = predict_brand(
    csv_input_path=_fixed,
    model=model,
    output_path=_OUTPUT_DIR / "1M_labeled_report.csv"
)

print(f"\nTotal Lines: {len(result)}")
print(f"\nDistribution:")
print(result["decision"].value_counts().to_string())

print(f"\nPercentage Ratio:")
print((result["decision"].value_counts(normalize=True) * 100).round(2).to_string())

print(f"\nExamples Mercedes:")
print(result[result["decision"] == "mercedes"]["article"].head(20).to_string())

print(f"\nExamples manual_review:")
print(result[result["decision"] == "manual_review"][["article", "prob_mercedes"]].head(20).to_string())
