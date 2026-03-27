# FsML_project — Mercedes-Benz part number classification

A machine learning project for **binary classification** of automotive part numbers: **Mercedes-Benz** (1) vs **non-Mercedes** (0). It combines format-based rules, string feature engineering, and logistic regression.

## Repository layout

| Path | Purpose |
|------|---------|
| `Main/Phase#1/` | Main pipeline: data, training, inference |
| `Main/Phase#2/Scripts/` | Modular layout: `features_scr/atomar.py`, `models_scr/inference.py`, `data_scr/loads.py` |

## Dependencies

- Python 3  
- `pandas`, `scikit-learn`, `joblib`  
- For `Converter.py`: Excel export (typically `openpyxl` or `xlsxwriter`)

Use a virtual environment and install packages as needed, for example:

```bash
pip install pandas scikit-learn joblib openpyxl
```

## Data files (expected)

Scripts assume files live in the **current working directory** when you run them (often the `Phase#1` folder itself):

- `mercedes-benz 300k.txt` — positive class (one part number per line, no header).  
- `not mercedes-benz 300k.txt` — negative class.  
- `mixed_train_300k.csv` — mixed dataset with `article` and `label` columns (a sample exists in the repo).  
- `giga_mixed_train_600k.csv` — larger mixed dataset (produced by `#1.3.py` if you uncomment the save step).

For the “exam” flow: `1M_parts_numbers.csv` → normalized CSV and a predictions report.

## Phase 1 — scripts

1. **`#1.1.py`** — load both `.txt` files, clean, exploratory analysis (string lengths, etc.).  
2. **`#1.2.py`** — **rule-based** check: regexes for known MB patterns (`[ABNC]` + 10 digits and suffix variants). Reports match rates for MB vs non-MB.  
3. **`#1.3.py`** — build balanced samples (e.g. 75k+75k and 150k+150k), shuffle, optional CSV export.  
4. **`#1.4.py`** — minimal model: single feature — string length + `LogisticRegression` (draft before the full feature set).  
5. **`#1.5 [polygon].py`** — full loop: feature matrix from `Util.build_feature_matrix`, train on a large sample, save **`mercedes_model.pkl`**, validation, confidence thresholds, **`predict_brand`** to label a CSV.  
6. **`Exam.py`** — load a large part-number CSV, normalize by re-saving, run the saved model, write `1M_labeled_report.csv` and print decision stats.  
7. **`Util.py`** — dataset loaders, **`extract_features_from_article`** / **`build_feature_matrix`** (length, letter/digit counts, MB prefixes, 10-digit “core” parsing, suffixes, etc.), **`predict_brand`** — probabilities, `mercedes` / `not_mercedes` / `manual_review` thresholds, report export.  
8. **`Converter.py`** — move the labeled report from CSV to Excel with a numbered index.

## Features (summary)

Numeric and binary features are derived from each article string: length, letter/digit counts, first-character flags (including `A`/`B`/`N`/`C` prefixes), leading letter run and trailing letters/digits, match against a 10-digit core regex, and when it matches — model / group / version / revision fields, plus a “possibly truncated MB” flag (no letter before digits), and more.

## Inference (Phase 2)

`Main/Phase#2/Scripts/models_scr/inference.py` defines **`BrandPredictor`**: single-row and batch prediction, multiclass-style API (`predict_proba`, `classes_`). Feature import: `from features_scr.atomar import ...` — run from the `Scripts` directory or set `PYTHONPATH` accordingly.

**Note:** `BrandPredictor.predict_one` uses `self.model.classes`; in scikit-learn the attribute is usually `classes_` — change it if you hit attribute errors.

## Typical workflow

1. Place the source `.txt` files in the working folder; generate `mixed_train_300k.csv` / `giga_mixed_train_600k.csv` with `#1.3.py` if needed.  
2. Run `#1.5 [polygon].py` from the data directory to obtain `mercedes_model.pkl` and optionally `labeled_report.csv`.  
3. To label a new file in bulk, follow `Exam.py`: `joblib.load("mercedes_model.pkl")` and `predict_brand(...)`.

## Artifacts

- `mercedes_model.pkl` — trained model.  
- `labeled_report.csv`, `1M_labeled_report.csv` — input articles plus `prob_mercedes`, `decision`, text `label`.  
- `1M_labeled_report_fixed.xlsx` — after `Converter.py`.

---

This is a teaching/research project: file paths are hardcoded relative to the process working directory — confirm your cwd and filenames match the scripts before running.
