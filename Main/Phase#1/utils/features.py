import re
import pandas as pd

mb_prefixes = set("ABNC")
non_mb_prefixes = set("XZMLDJESTKVW")

core_re = re.compile(
    r'^(?P<prefix>[A-Z])?'
    r'(?P<core>\d{10})'
    r'(?P<suffix>[A-Z0-9]*)$'
)


def extract_features_from_article(article: str) -> dict:
    s = str(article).strip().upper()
    f = {}

    f["article_len"] = len(s)
    f["num_letters"] = sum(1 for ch in s if ch.isalpha())
    f["num_digits"] = sum(1 for ch in s if ch.isdigit())

    f["first_char_is_letter"] = int(bool(s) and s[0].isalpha())
    f["first_char_is_mb_prefix"] = int(bool(s) and s[0] in mb_prefixes)
    f["first_char_is_non_mb"] = int(bool(s) and s[0] in non_mb_prefixes)

    prefix_letters = 0
    for ch in s:
        if ch.isalpha():
            prefix_letters += 1
        else:
            break
    f["prefix_letters"] = prefix_letters

    suffix_letters = suffix_digits = 0
    for ch in reversed(s):
        if ch.isalpha():
            suffix_letters += 1
        elif ch.isdigit():
            suffix_digits += 1
        else:
            break
    f["suffix_letters"] = suffix_letters
    f["suffix_digits"] = suffix_digits

    m = core_re.match(s)
    f["matches_core_pattern"] = int(bool(m))

    if m:
        prefix = m.group("prefix")
        core = m.group("core")
        f["has_prefix"] = int(bool(prefix))
        f["prefix_is_mb_set"] = int(bool(prefix) and prefix in mb_prefixes)
        f["core_len"] = len(core)
        f["model_int"] = int(core[0:3])
        f["group_int"] = int(core[3:6])
        f["version_int"] = int(core[6:8])
        f["revision_int"] = int(core[8:10])
        f["possible_truncated_mb"] = int(not prefix and len(core) == 10)
    else:
        f["has_prefix"] = 0
        f["prefix_is_mb_set"] = 0
        f["core_len"] = 0
        f["model_int"] = -1
        f["group_int"] = -1
        f["version_int"] = -1
        f["revision_int"] = -1
        f["possible_truncated_mb"] = 0

    return f


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(df["article"].apply(extract_features_from_article).tolist())
