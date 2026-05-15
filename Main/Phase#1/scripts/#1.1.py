import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
from utils.loaders import load_files

mb, not_mb = load_files()

print("cleaning module")

mb["article"] = mb["article"].str.strip()
not_mb["article"] = not_mb["article"].str.strip()

mb = mb[mb["article"].notna() & (mb["article"] != "")]
not_mb = not_mb[not_mb["article"].notna() & (not_mb["article"] != "")]

print(f"Mercedes: {len(mb)} articles")
print(f"Non-Mercedes: {len(not_mb)} articles")

print(mb.head(10))
print(not_mb.head(10))

print(mb.info())
print(not_mb.info())

print("Length of MB ", mb["article"].str.len().value_counts().sort_index())
print("Length of non-MB", not_mb["article"].str.len().value_counts().sort_index())
