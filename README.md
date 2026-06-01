# FsML — Automotive Parts Brand Classifier

---

## Overview

Automotive parts catalogues routinely contain millions of raw article numbers stripped of brand metadata.
The task: reconstruct brand identity from the part number string alone.

The project runs in two phases. Phase#1 was a focused binary experiment. Phase#2 is the full system, built around two models that cooperate in sequence — one supervised and rule-anchored, one unsupervised and geometry-driven — connected through a teacher-student data pipeline inspired by work in Graph Signal Processing.

### Multi-brand Classification · LinearSVC × TopBFM · Teacher-Student Pipeline
> *A machine learning pipeline that reads raw automotive part number strings and assigns each one a manufacturer brand — Mercedes-Benz, BMW, VAG, Toyota, Honda, Nissan, Mitsubishi, or unknown — across a corpus of one million articles, with no catalogue context and no barcode.*

---

## Project Structure

```
FsML_project-1/
└── Main/
    ├── Phase#1/                        ← Archived binary classifier (Mercedes vs. rest)
    │   ├── scripts/                    ← Exploration: #1.1 EDA → #1.5 final model
    │   ├── pipeline/                   ← exam.py, converter.py, stats.py
    │   ├── utils/                      ← loaders, features, inference
    │   ├── models/                     ← mercedes_model.pkl
    │   ├── data/                       ← mixed_train_300k.csv
    │   └── output/                     ← Labeled reports and Excel exports
    │
    └── Phase#2/                        ← Active pipeline
        ├── Data/
        │   ├── original/               ← Canonical source data (all phases read from here)
        │   └── processed/              ← LinearSVC output, TopBFM output, unknown_for_training
        ├── Models/
        │   ├── linearcvs/              ← LinearSVC bundle (scaler + model + feature_order)
        │   └── topbfm/                 ← topbfm.pkl, embedder.pkl, scaler.pkl
        ├── Scripts/
        │   ├── data/                   ← loads.py
        │   ├── features/               ← atomar.py, embedder.py, brand_rules.py
        │   ├── models/                 ← unsuprv.py (TopBFM), predictor.py, validator.py
        │   └── pipeline/               ← classic.py, train_topbfm.py, label_large_file.py,
        │                                  filter_unknown.py, run_pipeline.py, stats.py
        ├── Lab/                        ← Experimental scripts
        ├── Reports/                    ← Per-run classification reports, cluster_distribution.json
        └── output/                     ← Visualizations and diagnostics
```

---

## Dataset

- **Source:** proprietary parts catalogue — ~1 million raw article numbers
- **Known brands:** Mercedes-Benz · BMW · VAG (Volkswagen Group) · Toyota · Honda · Nissan · Mitsubishi
- **Training size per class:** 300k articles, balanced (capped at availability — Mitsubishi ≈206k)
- **Split:** 60 / 20 / 20 train / val / test, stratified
- **Normalization:** articles are uppercased and **dash-stripped** — Honda (~96 %) and Nissan
  (~57 %) carry dashes at source, while the 1M inference corpus is dash-free.

| File                        | Contents                                    |
|-----------------------------|---------------------------------------------|
| `mercedes-benz 300k.txt`    | Mercedes-Benz part numbers                  |
| `BMW 300k.csv`              | BMW part numbers with alternative codes     |
| `VAG 300k.csv`              | VAG part numbers                            |
| `toyota.csv`                | Toyota part numbers (`,` · header · col 1)   |
| `honda.csv`                 | Honda part numbers (`,` · no header · col 1) |
| `Auvika_MITSUBISHI.csv`     | Mitsubishi part numbers (`;` · no header · col 1) |
| `Price NISSAN_AE.txt`       | Nissan price list (tab · header+BOM · col 1) |
| `1M_parts_numbers.csv`      | Full unlabelled corpus for inference        |
| `giga_mixed_train_600k.csv` | Extended mixed training set (legacy)        |

Japanese-brand part-number architecture (PNC / platform / section / mnemonic prefixes) and the
per-brand regex cores are documented in the `TMDH` research report at the repo root.

---

## Phase\#1 — Binary Proof of Concept

Phase#1 answered the simplest version of the question: can a LinearSVC distinguish Mercedes-Benz
part numbers from everything else, using only the string itself?

It established the feature vocabulary — character n-grams, length, digit ratio, brand prefix
pattern flags — and produced the first 1M-row labeling pipeline.  The model (`mercedes_model.pkl`)
reached strong performance on the binary task, which confirmed the approach was worth scaling up.

Phase#1 is archived. Its scripts and outputs are preserved for reference but the active
pipeline is Phase#2.

---

## Phase\#2 — Teacher-Student Pipeline

Phase#2 is a two-model system.  The two models are architecturally independent — no shared
weights, no distillation loss — but they cooperate through the data they pass between each other.

### Model 1 — LinearSVC (the Teacher)

**Script:** `Scripts/pipeline/classic.py`  
**Artifacts:** `Models/linearcvs/linearsvc_atom_<N>k_4cls.pkl`

LinearSVC is trained on **atomic, hand-crafted features** extracted by `Scripts/features/atomar.py`:
character-level pattern flags, brand-specific prefix rules, length and digit-ratio statistics.
These features are deliberately interpretable and rule-like, which makes the model precise and
fast — but rigid, since it can only work with what the rules capture.

A `CalibratedClassifierCV` wrapper adds probability estimates: the SVM is fit **once**, then
calibrated on a held-out slice via `FrozenEstimator` — not a 3-fold refit.  Together with
`dual=False`, `tol=1e-3` and `float32` feature matrices, this keeps a full retrain on ~2.2M
articles at **~10 minutes** end-to-end (down from 10+ hours — the old `cv=3` calibration tripled
the data in memory and thrashed swap).  Articles whose top-class confidence falls below **0.85**
receive `manual_check` instead of a brand label — they are genuine borderline cases and are
excluded from everything downstream.

The teacher's primary job is not its own final output.  It is to **label the 1M corpus** so
that the student has high-quality training data for the hardest class.

### Model 2 — TopBFM (the Student)

**Script:** `Scripts/pipeline/train_topbfm.py`  
**Core class:** `Scripts/models/unsuprv.py`  
**Artifacts:** `Models/topbfm/{topbfm, embedder, scaler}.pkl`

TopBFM is a **MiniBatchKMeans clustering model** with a purity-based label assignment scheme.
It operates on brand-agnostic embeddings rather than hand-crafted rules:

- `ArticleEmbedder` — character-level embedding of the part number string (char TF-IDF 2–4 → SVD 100)
- Generic numeric features: `len`, `digit_ratio`, plus 4 positional discriminators mirrored from
  the teacher's `atomar.py` — `num_letters`, `first_letter_pos`, `ends_z_letter`, `ends_two_letters`.
  These are up-weighted (`TOPBFM_DISC_WEIGHT`, ×3) after scaling so they aren't drowned by the
  100-dim embedding.  They separate honda (Z-suffix, two-letter ending) from nissan (a single
  embedded letter) — structure that char n-grams alone could not, which is why the student
  previously collapsed honda/nissan into `manual_check`.
- Brand-flag features appended *after* scaling (multiplied by `TOPBFM_FLAG_WEIGHT`, ×8) so
  `StandardScaler` doesn't neutralise the brand identity signal

Each cluster is assigned the dominant label only if its **purity ≥ 0.80** — impure clusters
are labelled `manual_check`, not forced into a brand.  At 1750 clusters for 8 classes the model
learns the fine-grained geometric structure of each brand's part number space.

**Latest run — 2026-06-01, n_clusters=1750, purity_threshold=0.80, disc_weight=3.0:**

| Class             | Precision | Recall | F1       | Support |
|-------------------|-----------|--------|----------|---------|
| `mercedes`        | 0.99      | 0.99   | **0.99** | 120 000 |
| `mitsubishi`      | 1.00      | 0.92   | **0.96** | 82 594  |
| `vag`             | 0.97      | 0.90   | **0.94** | 120 000 |
| `bmw`             | 0.94      | 0.93   | **0.93** | 120 000 |
| `unknown_article` | 0.90      | 0.89   | **0.89** | 114 715 |
| `toyota`          | 0.96      | 0.74   | **0.84** | 120 000 |
| `honda`           | 0.94      | 0.42   | **0.59** | 120 000 |
| `nissan`          | 0.91      | 0.30   | **0.45** | 120 000 |
| **accuracy**      | —         | —      | **0.76** | 917 309 |

honda/nissan recall is the open frontier: their part numbers are the least structured of the
Japanese brands.  Injecting + weighting the positional discriminators lifted honda F1 0.42 → 0.59
and nissan 0.40 → 0.45, cut the real `nissan → vag` mislabel leak 10.7% → 6.6%, and dropped overall
`manual_check` on the 1M corpus 28.6% → 24.5% — all **without** lowering the purity threshold, i.e.
no accuracy traded for coverage.  High precision (0.91–0.94) with low recall means the student is
conservative by design: when it commits to honda/nissan it is almost always right.

### The Teacher-Student Link

The central design decision of Phase#2: **where does the `unknown_article` training class come from?**

Early versions of TopBFM trained on the three known brands only.  `unknown_article` was assigned
by exclusion — whatever didn't fit a known cluster ended up there.  This created a structural
bias: any article slightly different from the known-brand prototypes would silently bleed into
unknown, even if it was a legitimate brand article the model simply hadn't seen well enough.

The fix was to give `unknown_article` its own real training examples.  LinearSVC labels the
1M corpus; `filter_unknown.py` extracts its confident `unknown_article` predictions (excluding
`manual_check` borderlines); TopBFM then trains on a balanced 4-class dataset where the unknown
class is grounded in actual data, not inferred by subtraction.

```
  LinearSVC  ──── labels 1M corpus
                        │
               filter_unknown.py
                        │
             unknown_for_training.csv   ← 300k clean unknown examples
                        │
    ┌───────────────────┘
    │   + mercedes 300k
    │   + bmw 300k          ──── TopBFM trains on 4 balanced classes
    │   + vag 300k
```

The inspiration for structuring it this way came from **teacher-student learning in GSPy**, where
a simpler, well-understood model provides supervision signal — not weights — to guide a more
complex learner on territory where ground truth is expensive.

---

## Pipeline

The full pipeline is orchestrated by `run_pipeline.py`.

| Step | Script | What it does |
|------|-------------------------|---------------------------------------|
| 1    | `classic.py`            | Train LinearSVC                       |
| 2    | `predictor.py`          | Label 1M file with LinearSVC          |
| 3    | `filter_unknown.py`     | Extract unknown_article training data |
| 4    | `train_topbfm.py`       | Train TopBFM on 4 classes             |
| 5    | `label_large_file.py`   | Label 1M file with TopBFM             |
| 6    | `stats.py`              | Print diagnostics                     |

```bash
# Full end-to-end run
python Main/Phase#2/Scripts/pipeline/run_pipeline.py --full

# LinearSVC cycle only (steps 1-2-6)
python run_pipeline.py --linear

# TopBFM cycle only — reuses existing LinearSVC output (steps 3-4-5-6)
python run_pipeline.py --topbfm

# Resume from a specific step
python run_pipeline.py --from 4

# Single step
python run_pipeline.py --only 6
```

**Tuning knobs:**

```bash
# Cluster purity gate. Higher → more manual_check, fewer borderline brand assignments;
# lower → fewer manual_check at the cost of accuracy on the "rescued" articles.
TOPBFM_PURITY_THRESHOLD=0.80 python .../train_topbfm.py

# Stronger brand-flag signal in the TopBFM feature space
TOPBFM_FLAG_WEIGHT=8.0 python .../train_topbfm.py

# Weight on the 4 positional discriminators (num_letters, first_letter_pos, ends_z_letter,
# ends_two_letters) so they aren't drowned by the 100-dim char embedding.
TOPBFM_CLUSTERS_PER_CLASS=200 TOPBFM_DISC_WEIGHT=3.0 python .../train_topbfm.py
```

> **Note:** the proven configuration is `PURITY=0.80 FLAG_WEIGHT=8.0 CLUSTERS_PER_CLASS=200
> DISC_WEIGHT=3.0`.  The in-code defaults differ (`purity=0.92`, `flag_weight=5.0`) — pass the
> env vars to reproduce the model in the table above.

---

## Output Schema

Both models produce a labeled CSV with the same columns:

| Column    | Values                                                          |
|-----------|-----------------------------------------------------------------|
| `article` | Normalised, uppercased part number                             |
| `label`   | `mercedes` · `bmw` · `vag` · `toyota` · `honda` · `nissan` · `mitsubishi` · `unknown_article` · `manual_check` |
| `*_prob`  | Per-brand confidence score                                      |
| `comment` | Human-readable confidence note                                  |

---

## Adding a New Brand

Both models need to know about the new brand — they load data independently.

1. Drop the brand dataset into `Data/original/`.
2. Register it in `Scripts/data/loads.py` → `load_all()` — this feeds **LinearSVC**.
   Note the source format (separator, header, article column index) — they vary per file.
3. Register it in `Scripts/pipeline/train_topbfm.py` → `load_data()` — this feeds **TopBFM**.
4. Add a rule in `Scripts/features/brand_rules.py` **and** a numeric-only
   `_extract_<brand>_features` extractor in `Scripts/features/atomar.py` (wired into
   `extract_features`).
5. Re-run `--full`. The teacher (LinearSVC) relabels the 1M corpus first, regenerating a
   clean `unknown_for_training.csv`; only then does the student (TopBFM) train.

The core embedding path is brand-agnostic by design — brand identity lives in the flag layer,
not baked into the feature extractor. All article text is normalised via `loads._normalize`
(uppercase + dash removal); inference mirrors this in `models/inference.py`.

---

## Dependencies

```bash
pip install pandas scikit-learn joblib openpyxl plotly
```
