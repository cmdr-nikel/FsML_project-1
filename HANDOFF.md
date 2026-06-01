# Handoff — Japanese-brand integration (Phase#2)

**Date:** 2026-05-30
**Status:** ✅ Integration complete & trained. LinearSVC teacher strong (acc 0.91).
TopBFM student works but honda/nissan weak — root cause diagnosed (see below).
**Nothing committed to git yet.**

---

## TL;DR

Added Toyota / Honda / Nissan / Mitsubishi to the Phase#2 pipeline (both the
LinearSVC teacher and the TopBFM student). Pipeline runs end-to-end. The
supervised teacher classifies all 7 brands well. The unsupervised student is
solid on 5 brands but routes most honda/nissan to `manual_check` — this is a
feature-space limitation, not a bug, and is fully diagnosed.

---

## Results

### LinearSVC teacher — FINAL, good
`Models/linearcvs/linearsvc_atom_2180k_4cls.pkl` · 85 features · acc **0.91** / macro F1 **0.91**

| class | P | R | F1 |
|-------|---|---|----|
| mercedes | 1.00 | 1.00 | 1.00 |
| vag | 0.97 | 0.98 | 0.98 |
| bmw | 0.92 | 1.00 | 0.96 |
| mitsubishi | 0.98 | 0.91 | 0.94 |
| honda | 0.92 | 0.88 | 0.90 |
| nissan | 0.77 | 0.87 | 0.82 |
| toyota | 0.89 | 0.74 | 0.81 |
| unknown_article | 0.86 | 0.94 | 0.89 |

⚠️ Training step (classic.py) takes ~3.8h. **Do NOT rerun unless source data changes.**

### TopBFM student — works, honda/nissan weak
`Models/topbfm/{topbfm,embedder,scaler}.pkl` · best params: **purity=0.80, flag_weight=8.0, clusters_per_class=200 (1750 clusters)** · acc **0.72**

| class | P | R | F1 |
|-------|---|---|----|
| mercedes | 0.99 | 0.99 | 0.99 |
| vag | 0.97 | 0.91 | 0.94 |
| mitsubishi | 1.00 | 0.89 | 0.94 |
| bmw | 0.94 | 0.92 | 0.93 |
| unknown_article | 0.89 | 0.81 | 0.85 |
| toyota | 0.95 | 0.71 | 0.82 |
| honda | 0.93 | 0.30 | 0.45 |
| nissan | 0.94 | 0.26 | 0.41 |

Tuning progression (manual_check% in 1M / accuracy): default 0.92/5 → 78%/0.49 ·
0.80/8 → 32%/0.69 · 0.80/8/cpc200 → **29.8%/0.72**. Diminishing returns reached.

---

## Root-cause diagnosis (honda / nissan)

Run `python Main/Phase#2/Scripts/pipeline/diagnose.py` to regenerate. Findings:

- **honda:** of unique articles the teacher calls honda, 73% → `manual_check`, only 14% → honda.
- **nissan:** 74% → `manual_check`, 14% → nissan, **9.6% → vag** (real misclassification).
- **Cluster analysis (the why):** each brand trained on 180k articles but spread across
  **1221 (honda) / 1255 (nissan) clusters**. Only **29.5% / 26.4%** of that mass lands in
  clusters actually labeled honda/nissan; **~61–65% lands in low-purity clusters → manual_check.**
- **Interpretation:** honda (`41100ZVL030ZC` — digits+letters mixed) and nissan (10-digit,
  collides with toyota) don't form clean clusters. The char-embedding doesn't separate them
  from other/junk strings fast enough; the brand-flag (even at weight 8) can't overcome it.

**This is a feature-space limit, not a threshold/cluster-count issue (already tuned).**

### Possible next steps (NOT done — for tomorrow)
1. Inject honda/nissan brand-flag features **into the embedding** (before scaling), not just
   the post-scale flag layer — strengthen the signal during clustering.
2. Or accept the design: production model = LinearSVC (0.91, handles honda/nissan well);
   TopBFM student is intentionally conservative (dirty cluster → manual_check).
3. If pushing TopBFM: try cpc=250 or purity=0.75, but expect small gains.

Sample files for manual review:
`Reports/samples_honda_disagreements.csv`, `Reports/samples_nissan_disagreements.csv`
(deduped, random, with article length + all *_prob columns).

---

## 1M label distribution (both models)

| label | LinearSVC | TopBFM |
|-------|----------:|-------:|
| unknown_article | 54.2% | 43.1% |
| manual_check | 14.0% | 29.8% |
| toyota | 9.5% | 8.3% |
| nissan | 5.5% | 1.5% |
| vag | 5.1% | 7.4% |
| bmw | 3.8% | 3.4% |
| mitsubishi | 3.0% | 2.7% |
| honda | 2.8% | 0.9% |
| mercedes | 2.2% | 2.9% |

---

## Code changes (on disk, `git diff --stat`)

| file | change |
|------|--------|
| `Scripts/features/atomar.py` | +4 numeric extractors + regex cores (from /TMDH). **Honda regex widened** `revision {3,4}→{2,6}`. |
| `Scripts/features/brand_rules.py` | 7 BRAND_RULES entries (was 3). |
| `Scripts/data/loads.py` | `_normalize()` (strip+upper+**drop dashes**); JP parsers (article = **col index 1** in all four); balance via explicit loop. |
| `Scripts/pipeline/train_topbfm.py` | BRANDS=7; load_data adds JP; env `TOPBFM_CLUSTERS_PER_CLASS` (default 200). |
| `Scripts/models/inference.py` | dash removal in predict_one/predict_batch. |
| `Scripts/models/predictor.py` | **FIXED**: loads newest `linearsvc_atom_*.pkl` by mtime (was hardcoded 1117k). |
| `Scripts/pipeline/diagnose.py` | NEW — diagnostics + sample collection. |

### 3 bugs found & fixed via real runs
1. **honda regex too narrow** — real revisions are 5-6 chars; all honda was → unknown.
2. **`groupby('brand').apply()` drops the grouping column** on pandas≥2.2/py3.14 — silently
   lost the `brand` column. Fixed with explicit loops (loads.py, train_topbfm.py, diagnose.py).
3. **predictor.py hardcoded the old 3-brand teacher** — broke teacher→student (polluted
   `unknown_for_training.csv` with JP articles). Now picks newest model.

---

## Source file formats (verified)

| file | sep | header | article col |
|------|-----|--------|-------------|
| toyota.csv | `,` | yes (`Brand,Article,Name`) | 1 |
| honda.csv | `,` | no | 1 |
| Auvika_MITSUBISHI.csv | `;` | no | 1 |
| Price NISSAN_AE.txt | tab | yes (+BOM) | 1 (col 0 = "Nissan") |

mitsubishi has only ~206k unique (capped below 300k); others hit 300k.
Cross-brand dupes in load_all: only 22 (toyota↔nissan), negligible.

---

## How to re-run

```bash
cd Main/Phase#2
# Full pipeline (incl. 3.8h LinearSVC retrain) — only if data changed:
python Scripts/pipeline/run_pipeline.py --full

# Re-run student only (teacher already trained) with best params:
TOPBFM_PURITY_THRESHOLD=0.80 TOPBFM_FLAG_WEIGHT=8.0 TOPBFM_CLUSTERS_PER_CLASS=200 \
  python Scripts/pipeline/run_pipeline.py --from 4

# Diagnostics:
python Scripts/pipeline/diagnose.py
```

---

## Housekeeping / TODO for tomorrow
- [ ] **Commit** — nothing is committed yet (`git diff --stat` = 6 modified, +4 new data, +diagnose.py).
- [x] **README** — updated: 7 brands in tagline/known-brands/label list, source-file format table, dash-normalization note, enriched "Adding a New Brand" section (now mentions atomar.py extractor + teacher→student order).
- [ ] Decide honda/nissan direction (see "Possible next steps").
- [ ] Delete throwaway `verify_jp.py` at repo root.
- [ ] 3-brand model backup preserved at `Models/_backup_3brands_20260530_130159/` (5 pkl) — keep until satisfied, then delete.
- [ ] Note: `linearcvs/*.pkl` are git-tracked (force-added earlier) despite `*.pkl` in .gitignore; topbfm/*.pkl are ignored.
```
