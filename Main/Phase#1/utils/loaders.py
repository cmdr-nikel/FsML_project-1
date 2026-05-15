from pathlib import Path
import pandas as pd

_PHASE1_DIR = Path(__file__).parents[1]
_DATA_DIR = Path(__file__).parents[2] / "Phase#2" / "Data" / "original"


def load_files():
    mb = pd.read_csv(
        _DATA_DIR / "mercedes-benz 300k.txt",
        header=None, names=["article"], encoding="utf-8"
    )
    not_mb = pd.read_csv(
        _DATA_DIR / "not mercedes-benz 300k.txt",
        header=None, names=["article"], encoding="utf-8"
    )
    return mb, not_mb


def load_mixed_fixed():
    return pd.read_csv(_PHASE1_DIR / "data" / "mixed_train_300k.csv")


def load_giga_mixed_fixed():
    return pd.read_csv(_DATA_DIR / "giga_mixed_train_600k.csv")
