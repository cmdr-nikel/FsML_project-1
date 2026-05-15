# FsML — Automotive Parts Brand Classifier

---

## Overview

Automotive parts catalogues routinely contain millions of raw article numbers stripped of brand metadata.
The task: reconstruct brand identity from the part number string alone.

The project runs in two phases. Phase#1 was a focused binary experiment. Phase#2 is the full system, built around two models that cooperate in sequence — one supervised and rule-anchored, one unsupervised and geometry-driven — connected through a teacher-student data pipeline inspired by work in Graph Signal Processing.

### Multi-brand Classification · LinearSVC × TopBFM · Teacher-Student Pipeline
> *A machine learning pipeline that reads raw automotive part number strings and assigns each one a manufacturer brand — Mercedes-Benz, BMW, VAG, or unknown — across a corpus of one million articles, with no catalogue context and no barcode.*

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
- **Known brands:** Mercedes-Benz · BMW · VAG (Volkswagen Group)
- **Training size per class:** 300k articles, balanced
- **Split:** 60 / 20 / 20 train / val / test, stratified

| File                        | Contents                                    |
|-----------------------------|---------------------------------------------|
| `mercedes-benz 300k.txt`    | Mercedes-Benz part numbers                  |
| `BMW 300k.csv`              | BMW part numbers with alternative codes     |
| `VAG 300k.csv`              | VAG part numbers                            |
| `1M_parts_numbers.csv`      | Full unlabelled corpus for inference        |
| `giga_mixed_train_600k.csv` | Extended mixed training set (legacy)        |

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

A `CalibratedClassifierCV` wrapper adds probability estimates.  Articles whose top-class
confidence falls below **0.85** receive `manual_check` instead of a brand label — they are
genuine borderline cases and are excluded from everything downstream.

The teacher's primary job is not its own final output.  It is to **label the 1M corpus** so
that the student has high-quality training data for the hardest class.

### Model 2 — TopBFM (the Student)

**Script:** `Scripts/pipeline/train_topbfm.py`  
**Core class:** `Scripts/models/unsuprv.py`  
**Artifacts:** `Models/topbfm/{topbfm, embedder, scaler}.pkl`

TopBFM is a **MiniBatchKMeans clustering model** with a purity-based label assignment scheme.
It operates on brand-agnostic embeddings rather than hand-crafted rules:

- `ArticleEmbedder` — character-level embedding of the part number string
- Generic numeric features: `len`, `digit_ratio`
- Brand-flag features appended *after* scaling (multiplied by ×5) so `StandardScaler` doesn't
  neutralise the brand identity signal

Each cluster is assigned the dominant label only if its **purity ≥ 0.92** — impure clusters
are labelled `manual_check`, not forced into a brand.  At 550 clusters for 4 classes the model
learns the fine-grained geometric structure of each brand's part number space.

**Latest run — 2026-05-15, n_clusters=550, purity_threshold=0.92:**

| Class             | Precision | Recall | F1       | Support |
|-------------------|-----------|--------|----------|---------|
| `bmw`             | 1.00      | 0.95   | **0.97** | 120 000 |
| `mercedes`        | 0.99      | 0.98   | **0.99** | 120 000 |
| `unknown_article` | 0.98      | 0.93   | **0.96** | 120 000 |
| `vag`             | 0.99      | 0.93   | **0.96** | 120 000 |
| **accuracy**      | —         | —      | **0.95** | 480 000 |

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
# Stricter cluster purity → more manual_check, fewer borderline brand assignments
TOPBFM_PURITY_THRESHOLD=0.95 python .../train_topbfm.py

# Stronger brand-flag signal in the TopBFM feature space
TOPBFM_FLAG_WEIGHT=8.0 python .../train_topbfm.py
```

---

## Output Schema

Both models produce a labeled CSV with the same columns:

| Column    | Values                                                          |
|-----------|-----------------------------------------------------------------|
| `article` | Normalised, uppercased part number                             |
| `label`   | `mercedes` · `bmw` · `vag` · `unknown_article` · `manual_check` |
| `*_prob`  | Per-brand confidence score                                      |
| `comment` | Human-readable confidence note                                  |

---

## Adding a New Brand

Both models need to know about the new brand — they load data independently.

1. Drop the brand dataset into `Data/original/`.
2. Register it in `Scripts/data/loads.py` → `load_all()` — this feeds **LinearSVC**.
3. Register it in `Scripts/pipeline/train_topbfm.py` → `load_data()` — this feeds **TopBFM**.
4. Re-run `--full`.
5. Optionally extend `Scripts/features/brand_rules.py` with brand-specific pattern rules.

The core embedding path is brand-agnostic by design — brand identity lives in the flag layer,
not baked into the feature extractor.

---

## Dependencies

```bash
pip install pandas scikit-learn joblib openpyxl plotly
```
