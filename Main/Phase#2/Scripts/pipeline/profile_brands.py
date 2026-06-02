"""
Per-brand SOURCE profiler — feature-engineering diagnostic.

For each brand's ground-truth source data (Data/original), this answers:
  1. What does the raw article structure look like? (length dist, digit/letter layout)
  2. Which BRAND_RULES fire on it? — the crux for feature design:
       - fires ONLY its own rule        -> clean, separable
       - fires multiple rules           -> COLLISION (ambiguous features)
       - fires NO rule                  -> INVISIBLE (will leak to unknown)
  3. Random raw samples per rule-outcome bucket → Reports/profile_<brand>.csv

Run:  python Scripts/pipeline/profile_brands.py
"""
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd

from Scripts.data.loads import (
    _normalize, BMW_PATH, VAG_PATH, MB_PATH,
    TOYOTA_PATH, HONDA_PATH, MITSU_PATH, NISSAN_PATH,
    PC_PATH, RENAULT_PATH, RENAULT_SHEET,
)
from Scripts.features.brand_rules import BRAND_RULES

_PHASE2  = Path(__file__).parents[2]
REPORTS  = _PHASE2 / "Reports"
N_SOURCE = 30_000   # sample per brand from source (speed)
N_DUMP   = 25       # raw rows dumped per bucket
SEED     = 42


def _load_brand(brand):
    """Read a brand's source file the same way load_all does (article = col 1, except MB)."""
    if brand == "bmw":
        s = pd.read_csv(BMW_PATH, sep="\t", dtype=str).iloc[:, 1]
    elif brand == "vag":
        s = pd.read_csv(VAG_PATH, sep="\t", dtype=str).iloc[:, 1]
    elif brand == "mercedes":
        s = pd.read_csv(MB_PATH, header=None, dtype=str).iloc[:, 0]
    elif brand == "toyota":
        s = pd.read_csv(TOYOTA_PATH, dtype=str, on_bad_lines="skip").iloc[:, 1]
    elif brand == "honda":
        s = pd.read_csv(HONDA_PATH, header=None, dtype=str, on_bad_lines="skip").iloc[:, 1]
    elif brand == "mitsubishi":
        s = pd.read_csv(MITSU_PATH, sep=";", header=None, dtype=str, on_bad_lines="skip").iloc[:, 1]
    elif brand == "nissan":
        s = pd.read_csv(NISSAN_PATH, sep="\t", dtype=str, encoding="utf-8-sig",
                        on_bad_lines="skip").iloc[:, 1]
    elif brand == "peugeot_citroen":
        s = pd.read_csv(PC_PATH, sep="\t", header=None, dtype=str,
                        keep_default_na=False, on_bad_lines="skip").iloc[:, 1]
    elif brand == "renault":
        s = pd.read_excel(RENAULT_PATH, sheet_name=RENAULT_SHEET, header=0,
                          dtype=str)["OEM"]
    s = _normalize(s)
    s = s[s.str.contains(r"\d", regex=True)].drop_duplicates()
    return s


def _struct(a):
    """Coarse structural signature: D=digit run, L=letter run. e.g. 'D5L3D5'."""
    out, i, n = [], 0, len(a)
    while i < n:
        c = a[i]
        kind = "D" if c.isdigit() else ("L" if c.isalpha() else "?")
        j = i
        while j < n and (("D" if a[j].isdigit() else ("L" if a[j].isalpha() else "?")) == kind):
            j += 1
        out.append(f"{kind}{j-i}")
        i = j
    return "".join(out)


def _fired_rules(a):
    return tuple(b for b, fn in BRAND_RULES.items() if fn(a))


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    brands = list(BRAND_RULES.keys())

    print("=" * 78)
    print("PER-BRAND SOURCE PROFILE — rule-firing & structure")
    print("=" * 78)

    for brand in brands:
        s = _load_brand(brand)
        samp = s.sample(min(N_SOURCE, len(s)), random_state=SEED).reset_index(drop=True)
        n = len(samp)

        fired = samp.apply(_fired_rules)
        own       = fired.apply(lambda t: brand in t)
        multi     = fired.apply(lambda t: len(t) > 1)
        none      = fired.apply(lambda t: len(t) == 0)
        own_only  = fired.apply(lambda t: t == (brand,))

        print(f"\n### {brand.upper()}  (unique source ≈ {len(s):,}, sampled {n:,})")
        print(f"  fires OWN rule        : {own.mean()*100:5.1f}%")
        print(f"  fires OWN only (clean): {own_only.mean()*100:5.1f}%")
        print(f"  fires MULTIPLE rules  : {multi.mean()*100:5.1f}%   <- collision risk")
        print(f"  fires NO rule         : {none.mean()*100:5.1f}%   <- invisible → unknown")

        # what does it collide WITH?
        collide = Counter()
        for t in fired[multi]:
            for b in t:
                if b != brand:
                    collide[b] += 1
        if collide:
            top = ", ".join(f"{b}:{c/n*100:.1f}%" for b, c in collide.most_common(4))
            print(f"  collides with         : {top}")

        # length + structure
        lens = samp.str.len()
        print(f"  length  p10/p50/p90   : {lens.quantile(.1):.0f} / {lens.median():.0f} / {lens.quantile(.9):.0f}")
        struct_top = samp.apply(_struct).value_counts().head(4)
        print(f"  top structures        : " +
              ", ".join(f"{k} ({v/n*100:.0f}%)" for k, v in struct_top.items()))

        # dump raw samples per outcome bucket for eyeballing
        df = pd.DataFrame({"article": samp, "struct": samp.apply(_struct),
                           "fired": fired.apply(lambda t: "|".join(t) or "<none>")})
        buckets = {
            "own_only":  df[own_only],
            "multi":     df[multi],
            "none":      df[none],
        }
        rows = []
        for name, b in buckets.items():
            take = b.sample(min(N_DUMP, len(b)), random_state=SEED) if len(b) else b
            take = take.assign(bucket=name)
            rows.append(take)
        out = REPORTS / f"profile_{brand}.csv"
        pd.concat(rows, ignore_index=True).to_csv(out, index=False)

    print(f"\nPer-brand raw samples → Reports/profile_<brand>.csv")


if __name__ == "__main__":
    main()
