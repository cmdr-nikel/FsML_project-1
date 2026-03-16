import pandas as pd

def load_files():
    mb = pd.read_csv('mercedes-benz 300k.txt', header=None, names=['article'], encoding='utf-8')
    not_mb = pd.read_csv('not mercedes-benz 300k.txt', header=None, names=['article'], encoding='utf-8')
    return mb, not_mb
def load_mixed_fixed():
    train_csv = pd.read_csv('mixed_train_300k.csv')
    return train_csv
def load_giga_mixed_fixed():
    giga_train_csv = pd.read_csv('giga_mixed_train_600k.csv')
    return giga_train_csv

"""
Feature Engineering Section (Separate module)
"""
import re
import pandas as pd

mb_prefixes = set("ABNC")

core_re = re.compile(
    r'^(?P<prefix>[A-Z])?'      # 1st letter
    r'(?P<core>\d{10})'         # 10 digit: 3+3+2+2
    r'(?P<suffix>[A-Z0-9]*)$'   # any suffix
)

"""
'constructor' for the form of atomic features: length, start/end structure, core, prefix, whether truncated, etc.
"""

#ideally it should be in a separated file, but let it be here for now
def extract_features_from_article(article: str):        #dict
    s = str(article).strip().upper() #primary filtration of data
    f = {}

    #base features
    f["article_len"] = len(s)
    num_letters = 0
    for ch in s:
        if ch.isalpha():  # ch - is it a letter
            num_letters += 1
    f["num_letters"] = num_letters

    num_digits = 0
    for ch in s:
        if ch.isdigit():  # ch - is it a number
            num_digits += 1
    f["num_digits"] = num_digits

    #prefix
    if s:  # is the string not empty?
        if s[0].isalpha():  # is the first character a letter?
            f["first_char_is_letter"] = 1
        else:
            f["first_char_is_letter"] = 0
    else:
        f["first_char_is_letter"] = 0  # empty string → 0

    if s:  # is the string not empty?
        if s[0] in mb_prefixes:  # is the first character A, B, N or C?
            f["first_char_is_mb_prefix"] = 1
        else:
            f["first_char_is_mb_prefix"] = 0
    else:
        f["first_char_is_mb_prefix"] = 0  # empty string → 0

    prefix_letters = 0
    for ch in s:
        if ch.isalpha():
            prefix_letters += 1
        else:
            break
    f["prefix_letters"] = prefix_letters

    #syffix
    suffix_letters = 0
    suffix_digits = 0
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
    f["matches_core_pattern"] = 1 if m else 0

    if m:
        prefix = m.group("prefix")
        core = m.group("core")
        suffix = m.group("suffix")

        f["has_prefix"] = 1 if prefix else 0
        f["prefix_is_mb_set"] = 1 if prefix and prefix in mb_prefixes else 0

        # deconstruction of base 10 digits core
        model = core[0:3]
        group = core[3:6]
        version = core[6:8]
        revision = core[8:10]

        f["core_len"] = len(core)
        f["model_int"] = int(model)
        f["group_int"] = int(group)
        f["version_int"] = int(version)
        f["revision_int"] = int(revision)

        #1st digit might be cut
        f["possible_truncated_mb"] = 1 if not prefix and len(core) == 10 else 0
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

def build_feature_matrix(df):
    return pd.DataFrame(
        df["article"].apply(extract_features_from_article).tolist()
    )

#in a different script file
"""
from Utils import build_feature_matrix, load_mixed_fixed

train_df = load_mixed_fixed()
X_all = build_feature_matrix(train_df) 
y_all = train_df["label"] 
"""






