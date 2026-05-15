import sys, os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Scripts.features_scr.atomar import extract_features_from_article, featurize_column



# 1. Checking a single article from each brand
articles = {
    "mercedes": "A1234567890",   # MB-паттерн: буква + 10 цифр
    "bmw":      "12345678901",   # 11 цифр
    "vag":      "8K0853765A",    # платформа + группа + ревизия
    "unknown":  "XYZ999",        # ни под один паттерн
}

for brand, art in articles.items():
    f = extract_features_from_article(art)
    print(f"\n--- {brand}: {art} ---")
    print(f"  matches_mb_core:       {f['matches_mb_core']}")
    print(f"  bmw_is_valid_pattern:  {f['bmw_is_valid_pattern']}")
    print(f"  vag_is_valid_pattern:  {f['vag_is_valid_pattern']}")
    print(f"  any_known_pattern:     {f['any_known_pattern']}")

# 2. Checking featurize_column
series = pd.Series(articles.values())
df = featurize_column(series)
print(f"\nMatrix Shape: {df.shape}")
print(f"Column: {df.columns.tolist()}")
print(f"\nNaN in matrix: {df.isnull().sum().sum()}")  # should be 0 in here

vag_only = df[df['brand'] == 'vag']['article'].reset_index(drop=True)
vag_features = featurize_column(vag_only)


