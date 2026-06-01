import re
import pandas as pd
import numpy as np
from tqdm import tqdm

"""
'constructor' for the form of atomic features:
length, start/end structure, core, prefix, whether truncated, etc.
#actially, it is by far more complicated rn
"""

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

# --- Japanese brands (Toyota / Honda / Nissan / Mitsubishi) --- #
# Regex cores derived from the OEM part-number research (see /TMDH report).
# NOTE: all expect a dash-free, uppercased article — normalization strips '-'
# before featurization (honda/nissan are heavily dashed at source, the 1M
# inference corpus is dash-free).
TOYOTA_GENERAL_RE = re.compile(
    r'^(?P<pnc>\d{5})'
    r'(?P<base>\d{5})'
    r'(?P<suffix>[A-Z0-9]{2})?$'
)
TOYOTA_SUBARU_RE = re.compile(r'^SU003(?P<base>\d{5})$')

HONDA_GENERAL_RE = re.compile(
    r'^(?P<function>\d{5})'
    r'(?=[A-Z0-9]*[A-Z])'             # tightened 2026-05-31: require >=1 LETTER after the
                                      # function code. Real honda has letters in the middle
                                      # (87% letters-in-core); without this, pure-digit
                                      # toyota/bmw matched honda → flag was pure noise.
    r'(?P<model>[A-Z0-9]{3})'
    r'(?P<revision>[A-Z0-9]{2,6})$'   # TMDH said {3,4}; real catalog revisions run to 5-6 (e.g. A02RM, 030ZC)
)
HONDA_HARDWARE_RE = re.compile(
    r'^(?P<function>9\d{4})'
    r'(?P<dimension>\d{5})'
    r'(?P<iso>[A-Z0-9]{2,3})?$'
)

NISSAN_GENERAL_RE = re.compile(
    r'^(?=[A-Z0-9]*[A-Z])'           # tightened 2026-05-31: require >=1 LETTER. The old
                                      # rule `[A-Z0-9]{10,11}` matched pure-digit toyota and
                                      # 11-digit bmw → noise. Nissan is 87% letters-in-core;
                                      # the ~2.9% pure-digit nissan is sacrificed for signal.
    r'(?P<section>[A-Z0-9]{5})'
    r'(?P<identifier>[A-Z0-9]{5,6})$'
)
NISSAN_HARDWARE_RE = re.compile(
    r'^08'
    r'(?P<part_type>\d)'
    r'(?P<bolt_type>\d)'
    r'(?P<mod>\d)'
    r'(?P<diameter>\d{2})'
    r'(?P<length>\d{2})'
    r'(?P<finish>\d{2,3})?$'
)

MITSUBISHI_CLASSIC_RE = re.compile(r'^(?P<prefix>[A-Z]{2})(?P<core>\d{6})$')
MITSUBISHI_MODERN_RE = re.compile(r'^(?P<prefix_num>\d{4})(?P<alpha_rev>[A-Z])(?P<core_num>\d{3})$')

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

    # --- positional discriminators (added 2026-05-31 for honda/nissan separability) ---
    # Diagnosis: honda first letter sits at pos 5 (75%), mitsubishi pos 0 (82%),
    # toyota has no letters (58%). Position of the first letter is a strong brand signal.
    first_letter_pos = -1
    for i, ch in enumerate(s):
        if ch.isalpha():
            first_letter_pos = i
            break
    f["first_letter_pos"] = first_letter_pos

    # honda Z-suffix (ZA/ZZ/ZP...): 21.5% of honda vs ~0% of every other brand.
    f["ends_z_letter"] = 1 if re.search(r'Z[A-Z]$', s) else 0
    f["ends_two_letters"] = 1 if re.search(r'[A-Z]{2}$', s) else 0

    # letters strictly inside the core (excl. first/last 2 chars): honda/nissan ~87%
    # vs toyota 29% / bmw 2% / mitsubishi 10% — separates "mixed-core JDM" from digit brands.
    core = s[2:-2]
    f["has_letters_in_core"] = 1 if any(ch.isalpha() for ch in core) else 0

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

# --- Japanese brand extractors (numeric-only, mirror bmw/vag style) --- #
def _extract_toyota_features(s: str) -> dict:
    f = {}
    f["toyota_all_digits"] = 1 if s.isdigit() else 0
    f["toyota_is_10_digits"] = 1 if (s.isdigit() and len(s) == 10) else 0
    f["toyota_is_12_digits"] = 1 if (s.isdigit() and len(s) == 12) else 0
    m_gen = TOYOTA_GENERAL_RE.match(s)
    m_col = TOYOTA_SUBARU_RE.match(s)
    if m_col:
        f["toyota_is_valid_pattern"] = 1
        f["toyota_is_collab_subaru"] = 1
        f["toyota_pnc_int"] = -1
        f["toyota_base_int"] = int(m_col.group("base"))
        f["toyota_has_suffix"] = 0
        f["toyota_is_hardware"] = 0
        f["toyota_is_remanufactured"] = 0
    elif m_gen:
        pnc_val = int(m_gen.group("pnc"))
        f["toyota_is_valid_pattern"] = 1
        f["toyota_is_collab_subaru"] = 0
        f["toyota_pnc_int"] = pnc_val
        f["toyota_base_int"] = int(m_gen.group("base"))
        f["toyota_is_hardware"] = 1 if 90000 <= pnc_val <= 99999 else 0
        suffix = m_gen.group("suffix")
        f["toyota_has_suffix"] = 1 if suffix else 0
        f["toyota_is_remanufactured"] = 1 if suffix == "84" else 0
    else:
        f["toyota_is_valid_pattern"] = 0
        f["toyota_is_collab_subaru"] = 0
        f["toyota_pnc_int"] = -1
        f["toyota_base_int"] = -1
        f["toyota_has_suffix"] = 0
        f["toyota_is_hardware"] = 0
        f["toyota_is_remanufactured"] = 0
    return f


def _extract_honda_features(s: str) -> dict:
    f = {}
    f["honda_all_digits"] = 1 if s.isdigit() else 0
    m_hw = HONDA_HARDWARE_RE.match(s)
    m_gen = HONDA_GENERAL_RE.match(s)
    if m_hw:
        f["honda_is_valid_pattern"] = 1
        f["honda_is_hardware"] = 1
        f["honda_function_int"] = int(m_hw.group("function"))
        f["honda_is_accessory"] = 0
        f["honda_is_right_side"] = -1
        dim = m_hw.group("dimension")
        f["honda_thread_mm"] = int(dim[:2])
        f["honda_length_mm"] = int(dim[2:5])
    elif m_gen:
        func_val = int(m_gen.group("function"))
        f["honda_is_valid_pattern"] = 1
        f["honda_is_hardware"] = 0
        f["honda_function_int"] = func_val
        f["honda_is_accessory"] = 1 if s.startswith("08") else 0
        f["honda_is_right_side"] = func_val % 2
        f["honda_thread_mm"] = -1
        f["honda_length_mm"] = -1
    else:
        f["honda_is_valid_pattern"] = 0
        f["honda_is_hardware"] = 0
        f["honda_function_int"] = -1
        f["honda_is_accessory"] = 0
        f["honda_is_right_side"] = -1
        f["honda_thread_mm"] = -1
        f["honda_length_mm"] = -1
    return f


def _extract_nissan_features(s: str) -> dict:
    f = {}
    f["nissan_all_digits"] = 1 if s.isdigit() else 0
    f["nissan_is_10_digits"] = 1 if len(s) == 10 else 0
    m_hw = NISSAN_HARDWARE_RE.match(s)
    m_gen = NISSAN_GENERAL_RE.match(s)
    if m_hw:
        f["nissan_is_valid_pattern"] = 1
        f["nissan_is_hardware"] = 1
        f["nissan_fastener_type"] = int(m_hw.group("part_type"))
        f["nissan_material_type"] = int(m_hw.group("bolt_type"))
        f["nissan_thread_mm"] = int(m_hw.group("diameter"))
        f["nissan_length_mm"] = int(m_hw.group("length"))
    elif m_gen:
        f["nissan_is_valid_pattern"] = 1
        f["nissan_is_hardware"] = 0
        f["nissan_fastener_type"] = -1
        f["nissan_material_type"] = -1
        f["nissan_thread_mm"] = -1
        f["nissan_length_mm"] = -1
    else:
        f["nissan_is_valid_pattern"] = 0
        f["nissan_is_hardware"] = 0
        f["nissan_fastener_type"] = -1
        f["nissan_material_type"] = -1
        f["nissan_thread_mm"] = -1
        f["nissan_length_mm"] = -1
    return f


def _extract_mitsubishi_features(s: str) -> dict:
    f = {}
    f["mitsubishi_is_8_chars"] = 1 if len(s) == 8 else 0
    m_classic = MITSUBISHI_CLASSIC_RE.match(s)
    m_modern = MITSUBISHI_MODERN_RE.match(s)
    if m_classic:
        prefix = m_classic.group("prefix")
        f["mitsubishi_is_valid_pattern"] = 1
        f["mitsubishi_is_classic"] = 1
        f["mitsubishi_is_modern"] = 0
        f["mitsubishi_core_int"] = int(m_classic.group("core"))
        f["mitsubishi_is_engine_part"] = 1 if prefix == "MD" else 0
        f["mitsubishi_is_general_part"] = 1 if prefix == "MR" else 0
        f["mitsubishi_is_accessory"] = 1 if prefix == "MZ" else 0
    elif m_modern:
        f["mitsubishi_is_valid_pattern"] = 1
        f["mitsubishi_is_classic"] = 0
        f["mitsubishi_is_modern"] = 1
        f["mitsubishi_core_int"] = int(m_modern.group("core_num"))
        f["mitsubishi_is_engine_part"] = 0
        f["mitsubishi_is_general_part"] = 0
        f["mitsubishi_is_accessory"] = 0
    else:
        f["mitsubishi_is_valid_pattern"] = 0
        f["mitsubishi_is_classic"] = 0
        f["mitsubishi_is_modern"] = 0
        f["mitsubishi_core_int"] = -1
        f["mitsubishi_is_engine_part"] = 0
        f["mitsubishi_is_general_part"] = 0
        f["mitsubishi_is_accessory"] = 0
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
    f.update(_extract_toyota_features(s))
    f.update(_extract_honda_features(s))
    f.update(_extract_nissan_features(s))
    f.update(_extract_mitsubishi_features(s))



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
    Optimized for memory.
    """
    records = [extract_features(s) for s in tqdm(series, desc=f"Featurizing {len(series)} articles")]
    return pd.DataFrame.from_records(records, index=series.index)


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