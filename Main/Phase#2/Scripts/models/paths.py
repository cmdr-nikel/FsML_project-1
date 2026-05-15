import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))
from Scripts.features.brand_rules import BRAND_RULES

_PHASE2   = Path(__file__).parents[2]
DATA_DIR  = _PHASE2 / "Data" / "original"
MODEL_DIR = _PHASE2 / "Models" / "topbfm"
LOG_DIR   = _PHASE2 / "Reports"

def get_generic_features(article_list):
    """Brand-agnostic core features used by TopBFM."""
    df_feat = pd.DataFrame({'article': article_list})
    df_feat['len'] = df_feat['article'].str.len()
    df_feat['digit_ratio'] = df_feat['article'].apply(lambda x: sum(c.isdigit() for c in str(x)) / len(str(x)) if len(str(x)) > 0 else 0)
    return df_feat[['len', 'digit_ratio']].values

def get_brand_flags(article_list):
    """
    Optional rule flags for experiments.
    Column order follows BRAND_RULES insertion order.
    """
    rows = []
    for article in article_list:
        s = str(article)
        rows.append([1 if rule_fn(s) else 0 for rule_fn in BRAND_RULES.values()])
    return np.asarray(rows, dtype=float)

def apply_feature_weights(X_flags, weight=2.0):
    """Optional helper for weighting rule flags."""
    return X_flags * weight
