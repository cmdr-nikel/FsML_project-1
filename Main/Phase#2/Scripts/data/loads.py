from pathlib import Path
import pandas as pd

_PHASE2 = Path(__file__).parents[2]
DATA_DIR    = _PHASE2 / "Data" / "original"
BMW_PATH    = DATA_DIR / "BMW 300k.csv"
VAG_PATH    = DATA_DIR / "VAG 300k.csv"
MB_PATH     = DATA_DIR / "mercedes-benz 300k.txt"
NOT_MB_PATH = DATA_DIR / "not mercedes-benz 300k.txt"

# Japanese brands (added 2026-05). In ALL four the part number is column index 1;
# only the separator / header presence differs.
TOYOTA_PATH = DATA_DIR / "toyota.csv"               # ',' , header 'Brand,Article,Name'
HONDA_PATH  = DATA_DIR / "honda.csv"                 # ',' , no header (brand,code,name)
MITSU_PATH  = DATA_DIR / "Auvika_MITSUBISHI.csv"     # ';' , no header (brand,code,name,price,qty)
NISSAN_PATH = DATA_DIR / "Price NISSAN_AE.txt"       # tab , header + BOM (col0='Nissan', col1='Code')

# French brands (added 2026-06). Unusual formats vs the rest:
#   PC_PATH      — TAB, no header, 13 cols; brand col0='Peugeot-Citroen' (one
#                  source label for both marques — shared PSA numbering), article
#                  = col1 (duplicated in col11). Treated as a single class.
#   RENAULT_PATH — XLSX (the project's first spreadsheet source), sheet
#                  'UAE_RENAULT_3', header row; article = column 'OEM'.
PC_PATH      = DATA_DIR / "0000b96d56472a4f23fb4f214cc31d4084.txt"
RENAULT_PATH = DATA_DIR / "FpzY7aeLYoCxx2uE.xlsx.xlsx"
RENAULT_SHEET = "UAE_RENAULT_3"

SAMPLE_PER_BRAND = 300_000


def _normalize(series: pd.Series) -> pd.Series:
    """
    Canonical article form shared by training and inference:
    strip, uppercase, remove dashes.

    Dash removal is required: honda (~96%) and nissan (~57%) carry dashes at
    source, while the 1M inference corpus is 100% dash-free. Near no-op for the
    legacy brands (bmw/vag/mb ~0% dashes).
    """
    return series.astype(str).str.strip().str.upper().str.replace('-', '', regex=False)


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

    # --- Japanese brands (part number = column index 1 in every file) ---
    toyota = pd.read_csv(TOYOTA_PATH, dtype=str, on_bad_lines='skip').iloc[:, 1].to_frame(name='article')
    toyota['brand'] = 'toyota'

    honda = pd.read_csv(HONDA_PATH, header=None, dtype=str, on_bad_lines='skip').iloc[:, 1].to_frame(name='article')
    honda['brand'] = 'honda'

    mitsu = pd.read_csv(MITSU_PATH, sep=';', header=None, dtype=str, on_bad_lines='skip').iloc[:, 1].to_frame(name='article')
    mitsu['brand'] = 'mitsubishi'

    nissan = pd.read_csv(
        NISSAN_PATH, sep='\t', dtype=str, encoding='utf-8-sig', on_bad_lines='skip'
    ).iloc[:, 1].to_frame(name='article')
    nissan['brand'] = 'nissan'

    # --- French brands ---
    # Peugeot-Citroen: TAB-separated, no header, article = col 1.
    pc = pd.read_csv(
        PC_PATH, sep='\t', header=None, dtype=str,
        keep_default_na=False, on_bad_lines='skip'
    ).iloc[:, 1].to_frame(name='article')
    pc['brand'] = 'peugeot_citroen'

    # Renault: XLSX, article in the 'OEM' column.
    renault = pd.read_excel(
        RENAULT_PATH, sheet_name=RENAULT_SHEET, header=0, dtype=str
    )['OEM'].to_frame(name='article')
    renault['brand'] = 'renault'

    df = pd.concat([bmw, vag, mb, toyota, honda, mitsu, nissan, pc, renault], ignore_index=True)
    df['article'] = _normalize(df['article'])
    df = df.dropna(subset=['article'])
    df = df[df['article'] != '']
    # Drop non-article junk (headers / brand-name rows): every real OEM number has a digit
    df = df[df['article'].str.contains(r'\d', regex=True)]
    df = df.drop_duplicates(subset=['article', 'brand'])

    cross_brand = df[df.duplicated(subset=['article'], keep=False)]
    if len(cross_brand) > 0:
        print(f"[WARNING] Cross-brand duplicates found: {len(cross_brand)} rows")
        print(cross_brand.groupby('article')['brand'].apply(list).head(5))

    df = df.drop_duplicates(subset=['article'], keep='first')
    # Balance per brand; cap at availability so a short brand can't crash sample().
    # Explicit loop (not groupby.apply): pandas>=2.2 drops the grouping column
    # inside apply, which would silently delete 'brand'.
    parts = [
        g.sample(n=min(SAMPLE_PER_BRAND, len(g)), random_state=42)
        for _, g in df.groupby('brand')
    ]
    df = pd.concat(parts, ignore_index=True)
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
