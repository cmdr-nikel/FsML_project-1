import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import gc
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from Scripts.data.loads import load_all
from Scripts.features.atomar import featurize_column

_PHASE2 = Path(__file__).parents[2]


def reduce_mem_usage(df, verbose=True):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage of dataframe is {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage after optimization is: {end_mem:.2f} MB')
        print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')

    return df


def main():
    # =============================================================================
    # BLOCK 1 - Load and split data
    # =============================================================================
    print("--- Block 1: Load and split data ---")
    t0 = time.time()
    df = load_all()
    print(f"load_all() done in {time.time() - t0:.2f}s. Full dataset size: {df.shape}")
    print(df['brand'].value_counts())

    unknown_path = _PHASE2 / "Data" / "original" / "not mercedes-benz 300k.txt"
    df_unknown = pd.read_csv(unknown_path, header=None, names=["article"], dtype=str)
    df_unknown["article"] = df_unknown["article"].str.strip().str.upper()
    df_unknown["brand"] = "unknown_article"

    print("Filtering unknown articles by pattern...")
    unk_features = featurize_column(df_unknown["article"].reset_index(drop=True))
    unk_mask = unk_features["any_known_pattern"].astype(bool).values
    df_unknown = df_unknown[~unk_mask].reset_index(drop=True)
    print(f"True unknowns after filter: {len(df_unknown):,}")

    n_per_class = 300_000
    df_unknown = df_unknown.sample(n=min(n_per_class, len(df_unknown)), random_state=42)

    df = pd.concat([df, df_unknown], ignore_index=True).sample(frac=1, random_state=42)
    print(f"Full dataset size after adding unknowns: {df.shape}")
    print(df['brand'].value_counts())

    X = df['article'].reset_index(drop=True)
    y = df['brand'].reset_index(drop=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Block 1 done in {time.time() - t0:.2f}s.")

    # =============================================================================
    # BLOCK 2 - Feature extraction
    # =============================================================================
    print("\n--- Block 2: Feature extraction ---")
    t0 = time.time()
    print("Featurizing train set...")
    X_train_f = featurize_column(X_train)
    print("Featurizing validation set...")
    X_val_f   = featurize_column(X_val)
    print("Featurizing test set...")
    X_test_f  = featurize_column(X_test)

    print(f"Done. Shape: {X_train_f.shape}")

    X_train_f = X_train_f.fillna(0)
    X_val_f   = X_val_f.fillna(0)
    X_test_f  = X_test_f.fillna(0)

    print("\nReducing memory usage...")
    X_train_f = reduce_mem_usage(X_train_f)
    X_val_f = reduce_mem_usage(X_val_f)
    X_test_f = reduce_mem_usage(X_test_f)

    assert X_train_f.isna().sum().sum() == 0, "NaN in train!"
    assert X_val_f.isna().sum().sum() == 0, "NaN in val!"
    print("No NaNs. Ready for scaling.")

    feature_order = X_train_f.columns.tolist()
    X_val_f   = X_val_f.reindex(columns=feature_order, fill_value=0)
    X_test_f  = X_test_f.reindex(columns=feature_order, fill_value=0)

    print(f"Fixed shapes: {X_train_f.shape}, {X_val_f.shape}, {X_test_f.shape}")
    print(f"Block 2 done in {time.time() - t0:.2f}s.")

    # =============================================================================
    # BLOCK 3 - Scaling
    # =============================================================================
    print("\n--- Block 3: Scaling ---")
    t0 = time.time()
    scaler = StandardScaler()
    # StandardScaler returns float64; downcast to float32 to halve resident memory.
    # liblinear upcasts to float64 transiently per-fit, but the stored matrices stay small.
    X_train_fs = scaler.fit_transform(X_train_f).astype(np.float32, copy=False)
    X_val_fs   = scaler.transform(X_val_f).astype(np.float32, copy=False)
    X_test_fs  = scaler.transform(X_test_f).astype(np.float32, copy=False)

    # Drop the float DataFrames — feature_order is already captured and the scaled
    # arrays are all we need downstream. Keeps ~3 feature-matrix copies out of RAM,
    # which is what was pushing the box into swap and turning a 5-min job into 10h.
    del X_train_f, X_val_f, X_test_f
    gc.collect()
    print(f"Scaled arrays: dtype={X_train_fs.dtype}, train shape={X_train_fs.shape}")
    print(f"Block 3 done in {time.time() - t0:.2f}s.")

    # =============================================================================
    # BLOCK 4 - C tuning
    # =============================================================================
    print("\n--- Block 4: Train LinearSVC + calibrate ---")
    t0 = time.time()

    # Carve a small calibration slice out of TRAIN so val/test stay fully unseen
    # (honest eval in Blocks 5/6). 15% is plenty to fit per-class sigmoids.
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train_fs, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    del X_train_fs
    gc.collect()

    # One fast primal fit. dual=False is the correct solver when n_samples >> n_features
    # (~1e6 samples vs ~80 features). This is the change that makes it minutes, not hours.
    print("Training bare LinearSVC (C=0.01, dual=False)...")
    base_svc = LinearSVC(C=0.01, dual=False, tol=1e-3, max_iter=2000, random_state=42, verbose=1)
    base_svc.fit(X_tr, y_tr)
    print(f"  Bare SVC trained in {time.time() - t0:.2f}s.")

    # Calibrate probabilities with a single cheap fit on the frozen model, instead of
    # refitting the SVM 3x via cv=3 (the old memory/time sink).
    tc = time.time()
    print("Calibrating probabilities (prefit, sigmoid)...")
    model = CalibratedClassifierCV(FrozenEstimator(base_svc), method="sigmoid")
    model.fit(X_cal, y_cal)
    print(f"  Calibration done in {time.time() - tc:.2f}s.")
    print(f"Block 4 done in {time.time() - t0:.2f}s.")

    # =============================================================================
    # BLOCK 5 - Full evaluation on val set
    # =============================================================================
    print("\n--- Block 5: Validation set evaluation ---")
    t0 = time.time()
    y_pred = model.predict(X_val_fs)

    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Macro F1:", f1_score(y_val, y_pred, average='macro'))
    print("\nClassification report:")
    print(classification_report(y_val, y_pred))
    print("\nConfusion matrix (MB / BMW / VAG):")
    print(confusion_matrix(y_val, y_pred, labels=['mercedes', 'bmw', 'vag', 'unknown_article']))
    print(f"Block 5 done in {time.time() - t0:.2f}s.")

    # =============================================================================
    # BLOCK 6 - Final evaluation on test set
    # =============================================================================
    print("\n--- Block 6: Test set evaluation ---")
    t0 = time.time()
    y_test_pred = model.predict(X_test_fs)

    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Macro F1:", f1_score(y_test, y_test_pred, average='macro'))
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred))
    print("\nConfusion matrix (MB / BMW / VAG / UNK):")
    print(confusion_matrix(y_test, y_test_pred, labels=['mercedes', 'bmw', 'vag', 'unknown_article']))
    print(f"Block 6 done in {time.time() - t0:.2f}s.")

    # =============================================================================
    # BLOCK 7 - Save model + scaler
    # =============================================================================
    print("\n--- Block 7: Save model ---")
    t0 = time.time()
    n_total = len(df)
    MODEL_NAME = f"linearsvc_atom_{n_total // 1000}k_4cls.pkl"
    MODEL_DIR = _PHASE2 / "Models" / "linearcvs"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    save_path = MODEL_DIR / MODEL_NAME
    joblib.dump({
        "scaler": scaler,
        "model": model,
        "feature_order": feature_order
    }, save_path)
    print(f"\nModel saved → {save_path}")

    feature_order_path = Path(__file__).parents[1] / "features" / "feature_order.npy"
    np.save(feature_order_path, np.array(feature_order, dtype=object))
    print(f"Saved {len(feature_order)} features to {feature_order_path}")

    bmw_test_mask  = (y_test == 'bmw')
    lost_bmw_mask  = bmw_test_mask & (y_test_pred == 'unknown_article')
    lost_articles  = X_test[lost_bmw_mask]

    print("\n=== LOST BMW ARTICLES ===")
    print(f"Total lost: {lost_bmw_mask.sum()}")
    print("\nFirst 2 digits distribution:")
    print(lost_articles.str[:2].value_counts().head(15))
    print("\nLength distribution:")
    print(lost_articles.str.len().value_counts())
    print("\nSample (20):")
    print(lost_articles.sample(min(20, len(lost_articles))).tolist())
    print(f"Block 7 done in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()