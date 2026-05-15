from pathlib import Path
import pandas as pd

_PHASE2 = Path(__file__).parents[2]
DATA_DIR    = _PHASE2 / "Data" / "original"
BMW_PATH    = DATA_DIR / "BMW 300k.csv"
VAG_PATH    = DATA_DIR / "VAG 300k.csv"
MB_PATH     = DATA_DIR / "mercedes-benz 300k.txt"
NOT_MB_PATH = DATA_DIR / "not mercedes-benz 300k.txt"


def load_all() -> pd.DataFrame:
    """
    Loads all brand datasets, returns DataFrame with ['article', 'brand'].

    Sources:
      BMW 300k.csv / VAG 300k.csv — TSV: brand | number | alt_number
      mercedes-benz 300k.txt — plain text, one article per line
    """
    bmw_raw = pd.read_csv(BMW_PATH, sep='\t', dtype=str)
    bmw = bmw_raw.iloc[:, 1].to_frame(name='article')
    bmw['brand'] = 'bmw'

    vag_raw = pd.read_csv(VAG_PATH, sep='\t', dtype=str)
    vag = vag_raw.iloc[:, 1].to_frame(name='article')
    vag['brand'] = 'vag'

    mb = pd.read_csv(MB_PATH, header=None, names=['article'], dtype=str)
    mb['brand'] = 'mercedes'

    df = pd.concat([bmw, vag, mb], ignore_index=True)
    df['article'] = df['article'].str.strip().str.upper()
    df = df.dropna(subset=['article'])
    df = df[df['article'].str.strip() != '']
    df = df.drop_duplicates(subset=['article', 'brand'])

    cross_brand = df[df.duplicated(subset=['article'], keep=False)]
    if len(cross_brand) > 0:
        print(f"[WARNING] Cross-brand duplicates found: {len(cross_brand)} rows")
        print(cross_brand.groupby('article')['brand'].apply(list).head(5))

    df = df.drop_duplicates(subset=['article'], keep='first')
    df = df.groupby('brand').sample(n=300_000, random_state=42).reset_index(drop=True)
    return df


def load_auto(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in Path(data_dir).iterdir():
        brand = path.stem.split()[0].lower()
        if path.suffix == '.csv':
            df = pd.read_csv(path, sep='\t', dtype=str)
            if 'article' not in df.columns:
                df = df.iloc[:, 1].to_frame(name='article')
        elif path.suffix == '.txt':
            df = pd.read_csv(path, header=None, names=['article'], dtype=str)
        else:
            continue
        df['brand'] = brand
        df['article'] = df['article'].str.upper().str.strip()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    print("=== BMW (first 3 rows) ===")
    print(pd.read_csv(BMW_PATH, sep='\t', dtype=str, nrows=3))
    print("\n=== Mercedes (first 3 rows) ===")
    print(MB_PATH.read_text().splitlines()[:3])
    print("\n=== VAG (first 3 rows) ===")
    print(pd.read_csv(VAG_PATH, sep='\t', dtype=str, nrows=3))
    print("\n=== load_all() ===")
    df = load_all()
    print("Shape:", df.shape)
    print(df['brand'].value_counts())
    print(df.head(3))
