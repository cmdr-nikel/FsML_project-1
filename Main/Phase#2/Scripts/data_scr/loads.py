import os
import pandas as pd

# --- PART 1: Paths ---
# loads.py is at: Phase#2/Scripts/data_scr/loads.py
# Going up two levels (.., ..) lands at Phase#2/
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Data", "original")
)

BMW_PATH    = os.path.join(DATA_DIR, "BMW 300k.csv")
VAG_PATH    = os.path.join(DATA_DIR, "VAG 300k.csv")
MB_PATH     = os.path.join(DATA_DIR, "mercedes-benz 300k.txt")
NOT_MB_PATH = os.path.join(DATA_DIR, "not mercedes-benz 300k.txt")  # reserved, not used in load_all()


# --- PART 2: Load functions ---
def load_all() -> pd.DataFrame:
    """
    Loads all brand datasets and returns a single DataFrame
    with columns: ['article', 'brand']

    Sources:
      - BMW 300k.csv  — TSV, columns: brand | number | alt_number
      - VAG 300k.csv  — TSV, same structure as BMW
      - mercedes-benz 300k.txt — plain text, one article per line, no header
    """

    # dtype=str prevents pandas from guessing column types —
    # article numbers must always be treated as strings, not integers
    bmw_raw = pd.read_csv(BMW_PATH, sep='\t', dtype=str)
    bmw = bmw_raw.iloc[:, 1].to_frame(name='article')
    bmw['brand'] = 'bmw'

    vag_raw = pd.read_csv(VAG_PATH, sep='\t', dtype=str)
    vag = vag_raw.iloc[:, 1].to_frame(name='article')
    vag['brand'] = 'vag'

    # Plain text file — no header, one article per line
    mb = pd.read_csv(MB_PATH, header=None, names=['article'], dtype=str)
    mb['brand'] = 'mercedes'

    # Merge all brands into one DataFrame
    df = pd.concat([bmw, vag, mb], ignore_index=True)

    # Normalize: strip whitespace, uppercase — "8k0853765a" and "8K0853765A" are the same article
    df['article'] = df['article'].str.strip().str.upper()

    # Remove nulls and empty strings
    df = df.dropna(subset=['article'])
    df = df[df['article'].str.strip() != '']

    # Remove exact duplicates within the same brand first
    df = df.drop_duplicates(subset=['article', 'brand'])

    # Detect and warn about cross-brand duplicates before dropping them
    cross_brand = df[df.duplicated(subset=['article'], keep=False)]
    if len(cross_brand) > 0:
        print(f"[WARNING] Cross-brand duplicates found: {len(cross_brand)} rows")
        print(cross_brand.groupby('article')['brand'].apply(list).head(5))

    df = df.drop_duplicates(subset=['article'], keep='first')

    df = df.groupby('brand').sample(n=300_000, random_state=42).reset_index(drop=True)

    return df


# --- PART 3: Manual inspection — runs only when executed directly ---
if __name__ == "__main__":
    print("=== BMW (first 3 rows) ===")
    print(pd.read_csv(BMW_PATH, sep='\t', dtype=str, nrows=3))

    print("\n=== Mercedes (first 3 rows) ===")
    print(open(MB_PATH).readlines()[:3])

    print("\n=== VAG (first 3 rows) ===")
    print(pd.read_csv(VAG_PATH, sep='\t', dtype=str, nrows=3))

    print("\n=== load_all() ===")

    df = load_all()
    print("Shape:", df.shape)
    print(df['brand'].value_counts())
    print(df.head(3))


