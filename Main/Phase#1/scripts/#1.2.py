#Rule Based Solution
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.loaders import load_files

mb, not_mb = load_files()

"""
A    164    720126    28    K56
[ABNC] \d{3}  \d{6}  \d{2}  [A-Z]\d{2}  (more-or-less)
"""

patterns = {
    "org_pattern": r"^[ABNC]\d{10}$",                    # 11 sym - base pattern
    "es1_pattern": r"^[ABNC]\d{10}\d{2}$",               # 13 sym - base + es1
    "es2_pattern": r"^[ABNC]\d{10}[A-Z0-9]{4}$",         # 15 sym + es2 (suffix)
    "color_code":  r"^[ABNC]\d{10}\d{2}[A-Z0-9]{4}$",    # 17 sym + es1 + es2 (colour)
}

def is_mb(article):
    for pattern in patterns.values():
        if re.match(pattern, article):
            return True
    return False

mb["regex_match"] = mb["article"].apply(is_mb)
not_mb["regex_match"] = not_mb["article"].apply(is_mb)

print(f"MB:     {mb['regex_match'].mean():%}")
print(f"Non-MB: {not_mb['regex_match'].mean():%}")

false_positives = not_mb[not_mb["regex_match"] == True]
print(false_positives["article"].values)

"""
Some false positives are indeed MB (or side-manufacturer ones).
[MIGHT BE IMPORTANT FOR LATER]
"""
