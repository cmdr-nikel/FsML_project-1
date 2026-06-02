import re
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
    """Brand-agnostic core features used by TopBFM.

    len/digit_ratio + 4 positional discriminators mirrored bit-for-bit from
    atomar.py's _extract_generic_features (the teacher's set). These separate
    honda (Z-suffix 23.8%, first letter @pos5 84%) from nissan (single embedded
    letter 40%, no Z-suffix). The teacher had them; the student did not — which
    is why student honda/nissan recall collapsed (F1 0.40-0.42) while teacher
    held 0.83-0.92. Pure-digit articles get neutral values (pos=-1, letters=0),
    so that unsolvable ~10% is unaffected. See [[teacher-student-feature-split]].
    """
    rows = []
    for article in article_list:
        s = str(article)
        n = len(s)
        num_letters = sum(1 for ch in s if ch.isalpha())
        num_digits  = sum(1 for ch in s if ch.isdigit())
        first_letter_pos = next((i for i, ch in enumerate(s) if ch.isalpha()), -1)
        rows.append([
            n,                                       # len
            num_digits / n if n > 0 else 0.0,        # digit_ratio
            # --- discriminators (trailing block, up-weighted by disc_weight) ---
            num_letters,                             # num_letters
            first_letter_pos,                        # first_letter_pos
            1 if re.search(r'Z[A-Z]$', s) else 0,    # ends_z_letter  (honda marker)
            1 if re.search(r'[A-Z]{2}$', s) else 0,  # ends_two_letters (honda marker)
            # hybrid positional mask (2026-06-02) — mirrors atomar.py; only the
            # data-discriminative positions: pos5 honda-marker, pos8 honda, pos9
            # nissan. See [[french-brands-integration]] / [[teacher-student-feature-split]].
            1 if (n > 5 and s[5].isalpha()) else 0,  # letterpos_5
            1 if (n > 6 and s[6].isalpha()) else 0,  # letterpos_6
            1 if (n > 7 and s[7].isalpha()) else 0,  # letterpos_7
            1 if (n > 8 and s[8].isalpha()) else 0,  # letterpos_8
            1 if (n > 9 and s[9].isalpha()) else 0,  # letterpos_9
            1 if any(ch.isalpha() for ch in s[-3:]) else 0,  # has_alpha_in_last3
        ])
    return np.asarray(rows, dtype=float)

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
