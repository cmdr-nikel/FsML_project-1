# Handoff — French-brand integration (Phase#2)

**Date:** 2026-06-02
**Status:** ✅ Integration complete & trained. LinearSVC teacher strong (acc **0.92**).
TopBFM student lifted markedly by hybrid positional masks + an unknown-cluster relabel fix.
**Nothing committed to git yet** (this commit closes that out).

---

## TL;DR

Added **Renault** and **Peugeot-Citroën** to the Phase#2 pipeline (both LinearSVC teacher and
TopBFM student) — 7 → **9 brands**, 10 classes incl. `unknown_article`. Two follow-on wins this
session:
1. **Hybrid positional masks** (`letterpos_5…9`, `has_alpha_in_last3`) added to both feature
   functions — lifted student honda F1 0.59→**0.81**, nissan 0.45→**0.69**.
2. **Unknown-cluster fix** in `unsuprv.py` — low-purity clusters whose plurality is
   `unknown_article` now stay unknown instead of being demoted to `manual_check`. Pulled the 1M
   `manual_check` 37.7%→**22.9%** (into the tolerable 15–25% band) and `unknown_article`
   27.7%→**41.8%** (≈ teacher's 47.9%, realistic given unaccounted US/CN/local-market brands).

---

## Results

### LinearSVC teacher — FINAL, good
`Models/linearcvs/linearsvc_atom_2584k_4cls.pkl` · **106 features** · acc **0.92** / macro F1 **0.92**

| class | P | R | F1 |
|-------|---|---|----|
| mercedes | 1.00 | 1.00 | 1.00 |
| vag | 0.98 | 0.98 | 0.98 |
| bmw | 0.92 | 1.00 | 0.96 |
| mitsubishi | 0.96 | 0.95 | 0.96 |
| renault | 0.92 | 0.97 | 0.94 |
| peugeot_citroen | 0.92 | 0.94 | 0.93 |
| honda | 0.97 | 0.88 | 0.92 |
| nissan | 0.81 | 0.86 | 0.83 |
| toyota | 0.88 | 0.78 | 0.82 |
| unknown_article | 0.80 | 0.85 | 0.83 |

✅ Full retrain (`classic.py`) is now **~10–40 min** (perf fix: single calibrated fit, `dual=False`,
float32 matrices — the old `cv=3` calibration was the 3.8h culprit). Cheap to rerun.

### TopBFM student — masks + unknown-fix
`Models/topbfm/{topbfm,embedder,scaler}.pkl` · best params **purity=0.80, flag_weight=8.0,
clusters_per_class=200 (2150 clusters), disc_weight=3.0** · acc **0.80**

| class | P | R | F1 |
|-------|---|---|----|
| mercedes | 0.99 | 1.00 | 0.99 |
| vag | 0.99 | 0.92 | 0.95 |
| mitsubishi | 1.00 | 0.91 | 0.95 |
| bmw | 0.94 | 0.93 | 0.93 |
| renault | 0.93 | 0.87 | 0.90 |
| toyota | 0.96 | 0.74 | 0.83 |
| unknown_article | 0.78 | 0.85 | 0.82 |
| honda | 0.96 | 0.70 | 0.81 |
| peugeot_citroen | 0.96 | 0.67 | 0.79 |
| nissan | 0.94 | 0.55 | 0.69 |

Progression this session (student acc / 1M manual_check): french baseline 0.71 / 39.2% →
+masks 0.77 / 37.7% → +unknown-fix **0.80 / 22.9%**. Clean clusters 54.8% → 65.3%, purity 0.859 → 0.891.

---

## Key design notes

- **Renault `R`-suffix** (`\d{9}R`, ~46% of renault, ~98% of trailing letters) is a brand-exclusive
  marker — the French analogue of honda's Z-suffix. It survives even into the unsupervised student
  (renault F1 0.90).
- **Peugeot + Citroën = one class** `peugeot_citroen` (shared PSA numbering, one source label).
- **Hybrid masks beat the doc.** A supplied OEM doc claimed nissan letters sit at pos 9–10 vs honda
  6–8 (clean split). Data DISPROVED it: nissan letters are at pos 5–7 (95%), on top of honda. So we
  kept only the data-discriminative positions (pos5 honda-marker, pos8 honda, pos9 nissan) instead
  of the full 11-vector. Real honda↔nissan separators = **length (11 vs 10)** + **P(letter@pos5)**.
- **Alliance trap:** 5794 articles appear in BOTH nissan and renault source (shared platform roots
  without the `R` suffix). `load_all` `keep='first'` sends them to nissan — correct, since the
  `R` suffix is what distinguishes a Renault part from its shared Nissan root.
- **Open frontier:** the pure-digit floor. ~22% manual_check is mostly genuinely-ambiguous bare
  10-digit numbers (french + nissan + toyota collisions) that carry no structural signal — masks
  are empty there. Next lever (not done): a leading-prefix dictionary (renault 77/82/85/86, PSA
  16/96/98, nissan section codes) for the bare-digit pool. Or accept it (prod = teacher 0.92).

---

## 1M label distribution (both models, latest)

| label | LinearSVC | TopBFM |
|-------|----------:|-------:|
| unknown_article | 47.9% | 41.8% |
| manual_check | 13.9% | 22.9% |
| toyota | 8.9% | 7.6% |
| peugeot_citroen | 7.0% | 5.4% |
| nissan | 5.5% | 4.3% |
| vag | 4.9% | 7.0% |
| bmw | 3.7% | 3.4% |
| mitsubishi | 3.1% | 2.8% |
| mercedes | 2.2% | 1.9% |
| renault | 1.5% | 1.5% |
| honda | 1.4% | 1.4% |

---

## Code changes (on disk)

| file | change |
|------|--------|
| `Scripts/features/atomar.py` | French regex cores (RENAULT_ALLIANCE/CLASSIC, PSA_LONG/SHORT) + `_extract_renault_features` (6) + `_extract_psa_features` (5); hybrid mask `letterpos_5…9`+`has_alpha_in_last3` in `_extract_generic_features`. Feature count 89→**106**. |
| `Scripts/features/brand_rules.py` | +`renault`, +`peugeot_citroen` BRAND_RULES (7→9). |
| `Scripts/data/loads.py` | PC_PATH/RENAULT_PATH/RENAULT_SHEET + **xlsx reader** (`read_excel`) in `load_all`. |
| `Scripts/pipeline/train_topbfm.py` | BRANDS 7→9 + french in `load_data`; `N_DISCRIMINATORS` 4→**10**. |
| `Scripts/pipeline/label_large_file.py` | `_N_DISCRIMINATORS` 4→**10** (must match train). |
| `Scripts/models/paths.py` | hybrid mask mirrored into `get_generic_features` (6→**12** cols). |
| `Scripts/models/unsuprv.py` | **unknown-cluster fix**: low-purity unknown-plurality clusters stay `unknown_article`. |
| `Scripts/pipeline/profile_brands.py` | french branches for re-profiling. |

---

## Source file formats (verified)

| file | sep | header | article col |
|------|-----|--------|-------------|
| `0000b96d…txt` (Peugeot-Citroën) | tab | no | 1 (col 0 = "Peugeot-Citroen", dup in col 11) |
| `FpzY7aeLYoCxx2uE.xlsx.xlsx` (Renault) | **XLSX** | yes | `OEM` (sheet `UAE_RENAULT_3`) |

peugeot_citroen 611k unique (capped 300k); renault 122k unique (cap not hit). Needs `openpyxl`.

---

## How to re-run

```bash
cd Main/Phase#2
# Full pipeline (teacher retrain + student + 1M relabel). Cheap now (~10–40 min):
TOPBFM_PURITY_THRESHOLD=0.80 TOPBFM_FLAG_WEIGHT=8.0 TOPBFM_CLUSTERS_PER_CLASS=200 \
  TOPBFM_DISC_WEIGHT=3.0 python Scripts/pipeline/run_pipeline.py --full

# Student only (teacher already trained):
TOPBFM_PURITY_THRESHOLD=0.80 TOPBFM_FLAG_WEIGHT=8.0 TOPBFM_CLUSTERS_PER_CLASS=200 \
  TOPBFM_DISC_WEIGHT=3.0 python Scripts/pipeline/run_pipeline.py --from 4
```

⚠️ In-code defaults differ (`purity=0.92`, `flag_weight=5.0`) — pass the env vars to reproduce.

---

## Housekeeping / TODO next session
- [ ] **float32 student** (approved, not done): cast final matrix in `train_topbfm.build_features`
  `np.hstack([...]).astype(np.float32, copy=False)` + MIRROR in `label_large_file.py`. Student is
  float64 (~2.6GB → ~1.3GB; teacher already float32). Run `--from 4` clean to confirm no metric drift.
- [ ] pure-digit floor — leading-prefix dictionary for the bare-digit pool (see Key design notes).
- [ ] Backups to prune once satisfied: `Models/_backup_pre_french_*`, `_backup_baseline_french_*`,
  `_backup_premaskedfix_*` (contain large local-only CSV copies — keep out of git).
- [ ] Throwaway at repo root: `verify_jp.py`, empty `TMNH.md` — deletable.
- [ ] Note: `linearcvs/*.pkl` are git-tracked (force-added) despite `*.pkl` in .gitignore; the 1M
  labeled CSVs are gitignored (exceed GitHub's 100MB limit, regenerable via the pipeline).
