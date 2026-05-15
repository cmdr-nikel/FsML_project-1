import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_OUTPUT_DIR = _ROOT / "output"

df_new = pd.read_csv(_OUTPUT_DIR / "1M_labeled_report.csv")

df_new.index = range(1, len(df_new) + 1)
df_new.index.name = "index"

with pd.ExcelWriter(_OUTPUT_DIR / "1M_labeled_report_fixed.xlsx") as writer:
    df_new.to_excel(writer, index=True)
