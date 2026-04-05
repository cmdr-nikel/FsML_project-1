import re
import pandas as pd
import numpy as np

"""
'constructor' for the form of atomic features:
length, start/end structure, core, prefix, whether truncated, etc.
#actially, it is by far more complicated rn
"""

def featurize_column(article):
    features = {}

# --list of constants-- #
BMW_RE = re.compile(
    r'^(?P<main_group>\d{2})'
    r'(?P<subgroup>\d{2})'
    r'(?P<core7>\d{7})$'
)

BMW_HEX_RE = re.compile(
    r'^\d{4}5A[0-9A-F]{5}$',
    re.IGNORECASE
)

VAG_RE = re.compile(
    r'^(?P<platform>[A-Z0-9]{3})'   # 3-char platform code (8K0, 02E, 5C5...)
    r'(?P<group>\d{3})'             # 3 digits — group
    r'(?P<item>\d{3})'              # 3 digits — item
    r'(?P<revision>[A-Z0-9]*)$'     # variable alphanumeric suffix (was [A-Z]{0,2})
)

MB_PREFIXES = set("ABNC")

CORE_RE = re.compile(
    r'^(?P<prefix>[A-Z])?'      # optional 1st letter
    r'(?P<core>\d{10})'         # 10 digits: 3+3+2+2
    r'(?P<suffix>[A-Z0-9]*)$'   # any suffix
)

NON_MB_PREFIXES = set("XZMLDJESTKVW")

def _extract_generic_features(s):
    f = {}

    f["article_len"] = len(s)

    num_letters = sum(1 for ch in s if ch.isalpha())
    num_digits  = sum(1 for ch in s if ch.isdigit())

    f["num_letters"] = num_letters
    f["num_digits"]  = num_digits

    f["first_char_is_letter"] = 1 if (s and s[0].isalpha()) else 0

    prefix_letters = 0
    for ch in s:
        if ch.isalpha():
            prefix_letters += 1
        else:
            break
    f["prefix_letters"] = prefix_letters

    suffix_letters = 0
    suffix_digits  = 0
    for ch in reversed(s):
        if ch.isalpha():
            suffix_letters += 1
        elif ch.isdigit():
            suffix_digits += 1
        else:
            break
    f["suffix_letters"] = suffix_letters
    f["suffix_digits"]  = suffix_digits

    f["all_digits"]     = 1 if s.isdigit() else 0
    f["digit_ratio"]    = num_digits / f["article_len"] if f["article_len"] > 0 else 0.0
    f["has_only_alnum"] = 1 if s.isalnum() else 0
    f["num_blocks"]     = len(re.split(r'[ \-/]', s)) if s else 0

    return f


def _extract_mb_features(s):
    f = {}

    f["first_char_is_mb_prefix"] = 1 if (s and s[0] in MB_PREFIXES) else 0

    # Check CORE_RE only if article starts with a letter — otherwise it can't be MB
    m = CORE_RE.match(s) if (s and s[0].isalpha()) else None
    if m:
        f["matches_mb_core"]      = 1
        f["has_mb_letter_prefix"] = 1 if m.group("prefix") else 0
        f["core_len"]             = len(m.group("core"))   # always 10
        f["mb_suffix_len"]        = len(m.group("suffix"))
    else:
        f["matches_mb_core"]      = 0
        f["has_mb_letter_prefix"] = 0
        f["core_len"]             = 0
        f["mb_suffix_len"]        = 0

    f["mb_is_valid_pattern"] = f["matches_mb_core"]

    return f

def _extract_bmw_features(s: str) -> dict:
    s = str(s).strip()
    f = {}

    f["bmw_all_digits"] = 1 if s.isdigit() else 0
    f["bmw_len"] = len(s)
    f["bmw_is_11_digits"] = 1 if (s.isdigit() and len(s) == 11) else 0

    hex_match = BMW_HEX_RE.match(s)
    m = BMW_RE.match(s)

    f["bmw_is_hex_format"] = 1 if hex_match else 0

    if m:
        f["bmw_main_group_int"] = int(m.group("main_group"))
        f["bmw_subgroup_int"] = int(m.group("subgroup"))
        f["bmw_core7_int"] = int(m.group("core7"))
        f["bmw_is_valid_pattern"] = 1
    else:
        f["bmw_main_group_int"] = -1
        f["bmw_subgroup_int"] = -1
        f["bmw_core7_int"] = -1
        f["bmw_is_valid_pattern"] = 1 if hex_match else 0

    return f


def _extract_vag_features(s):
    f = {}
    f["vag_len"]     = len(s)
    f["vag_is_alnum"] = 1 if s.isalnum() else 0

    m = VAG_RE.match(s)
    if m:
        group    = m.group("group")
        revision = m.group("revision")

        f["vag_three_blocks_match"] = 1
        f["vag_main_group_digit"]   = int(group[0])
        f["vag_subgroup_digits"]    = int(group[1:])
        f["vag_item_number"]        = int(m.group("item"))
        f["vag_has_revision"]       = 1 if revision else 0
        f["vag_revision_len"]       = len(revision)
        f["vag_is_valid_pattern"]   = 1
    else:
        f["vag_three_blocks_match"] = 0
        f["vag_main_group_digit"]   = -1
        f["vag_subgroup_digits"]    = -1
        f["vag_item_number"]        = -1
        f["vag_has_revision"]       = 0
        f["vag_revision_len"]       = 0
        f["vag_is_valid_pattern"]   = 0

    return f

#NEW BLOCK TO INCREASE ACCURACY
def _extract_pk_features(s: str) -> dict:
    f = {}
    f["contains_pk"] = 1 if "PK" in s else 0
    f["starts_with_pk_number"] = 1 if re.match(r"^\d+PK\d+$", s) else 0

    m = re.match(r"^(?P<n>\d+)PK(?P<rest>\d+)$", s)
    if m:
        f["pk_prefix_len"] = len(m.group("n"))
        f["pk_suffix_len"] = len(m.group("rest"))
        f["pk_num"] = int(m.group("n"))
        f["pk_rest_num"] = int(m.group("rest"))
    else:
        f["pk_prefix_len"] = 0
        f["pk_suffix_len"] = 0
        f["pk_num"] = -1
        f["pk_rest_num"] = -1

    f["pk_is_valid_pattern"] = f["starts_with_pk_number"]

    return f

def featurize_prefix(article):
    return {
        'prefix_316': int(article.startswith('316')),
        'prefix_210': int(article.startswith('210')),
        'prefix_236': int(article.startswith('236')),  # MB
        'prefix_len3_unique': len(set(article[:3])) == 1
    }

def featurize_complexity(article):
    length = len(article)
    digit_ratio = np.mean([c.isdigit() for c in article])
    return {
        'len_bucket': np.digitize(length, [8, 10, 12, 15, 20]),
        'digit_ratio': digit_ratio,
        'entropy': -np.sum([p*np.log2(p+1e-10) for p in np.unique(list(article), return_counts=True)[1]/len(article)])
    }

def _extract_gates_features(s: str) -> dict:
    """15-значные числовые — Gates/Dayco/ContiTech"""
    f = {}
    f["gates_is_15_digits"] = 1 if re.match(r'^\d{15}$', s) else 0
    f["gates_is_valid_pattern"] = f["gates_is_15_digits"]
    return f


def _extract_045_features(s: str) -> dict:
    """045*-серия — Hella/Febi"""
    f = {}
    f["hella_is_045"] = 1 if re.match(r'^045\d{9,12}$', s) else 0
    f["hella_is_valid_pattern"] = f["hella_is_045"]
    return f


def _extract_316_features(s: str) -> dict:
    """316*-серия — Lemförder/Febi"""
    f = {}
    f["lemforder_is_316"] = 1 if re.match(r'^316\d{9,12}$', s) else 0
    f["lemforder_is_valid_pattern"] = f["lemforder_is_316"]
    return f


def _extract_0000100_features(s: str) -> dict:
    """0000100*-серия — Bosch OEM"""
    f = {}
    f["bosch_oem"] = 1 if re.match(r'^0000100\d{6,8}$', s) else 0
    f["bosch_is_valid_pattern"] = f["bosch_oem"]
    return f

def extract_features(s: str) -> dict:
    """
    Orchestrator - main entry point.
    Runs all feature extractions for a single article string,
    applies conflict resolution, then computes cross-brand summary.
    Returns a flat dictionary ready for a DataFrame row.
    """
    f = {}
    f.update(_extract_generic_features(s))
    f.update(_extract_mb_features(s))
    f.update(_extract_bmw_features(s))
    f.update(_extract_vag_features(s))
    f.update(_extract_pk_features(s))
    f.update(_extract_gates_features(s))
    f.update(_extract_045_features(s))
    f.update(_extract_316_features(s))
    f.update(_extract_0000100_features(s))



    # --- conflict resolution FIRST ---
    # MB articles (single letter prefix + 10 digits) must not be misclassified as VAG.
    # The expanded VAG_RE ([A-Z0-9]{3} + digits + suffix) now matches MB articles too,
    # so we explicitly zero out VAG flags when MB pattern is certain.
    if f.get("has_mb_letter_prefix", 0) == 1 and f.get("first_char_is_mb_prefix", 0) == 1:
        f["vag_is_valid_pattern"]   = 0
        f["vag_three_blocks_match"] = 0

    if f.get("gates_is_valid_pattern", 0):
        f["bmw_is_valid_pattern"] = 0

    if f.get("gates_is_15_digits", 0):
        f["bmw_is_valid_pattern"] = 0

    # --- cross-brand summary AFTER conflict resolution ---
    valid_flags = [
        int(val)
        for key, val in f.items()
        if key.endswith("_is_valid_pattern")
    ]
    f["any_known_pattern"] = 1 if any(valid_flags) else 0
    f["num_matched_patterns"] = sum(valid_flags)

    # Drop string/categorical fields — model expects only numeric features
    f.pop("vag_platform_code", None)
    f.pop("vag_revision_suffix", None)

    return f

def features_to_series(s: str) -> pd.Series:
    """Converts a single article string to a pd.Series of features."""
    return pd.Series(extract_features(s))


def featurize_column(series: pd.Series) -> pd.DataFrame:
    """
    Applies extract_features to an entire pandas Series of article strings.
    Returns a DataFrame where each row is one article's feature vector.
    """
    return series.apply(extract_features).apply(pd.Series)


# --- Public API aliases (used by inference.py) ---
def extract_features_from_article(article: str) -> dict:
    """Backward-compatible alias for extract_features."""
    return extract_features(article)


def build_feature_matrix(df: pd.DataFrame, article_col: str = "article") -> pd.DataFrame:
    """
    Accepts a DataFrame with an article column,
    returns a numeric feature matrix.
    """
    return featurize_column(df[article_col])