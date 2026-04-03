import sys, os
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from Scripts.data_scr.loads import load_all
from Scripts.features_scr.atomar import featurize_column




def main():
    # =============================================================================
    # BLOCK 1 - Load and split data
    # =============================================================================
    df = load_all()
    print("Full dataset size:", df.shape)
    print(df['brand'].value_counts())

    # uploading unknown article
    unknown_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "Data", "original", "not mercedes-benz 300k.txt")
    )
    df_unknown = pd.read_csv(unknown_path, header=None, names=["article"], dtype=str)
    df_unknown["article"] = df_unknown["article"].str.strip().str.upper()
    df_unknown["brand"] = "unknown_article"

    # filtration - leaving the ones, that does not match any known article
    print("Filtering unknown articles by pattern...")
    unk_features = featurize_column(df_unknown["article"].reset_index(drop=True))
    unk_mask = unk_features["any_known_pattern"].astype(bool).values
    df_unknown = df_unknown[~unk_mask].reset_index(drop=True)
    print(f"True unknowns after filter: {len(df_unknown):,}")

    # to the normal 300k format
    n_per_class = 300_000
    df_unknown = df_unknown.sample(n=min(n_per_class, len(df_unknown)), random_state=42)

    df = pd.concat([df, df_unknown], ignore_index=True).sample(frac=1, random_state=42)
    print("Full dataset size after adding unknowns:", df.shape)
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
    print(y_train.value_counts())

    # =============================================================================
    # BLOCK 2 - Feature extraction
    # =============================================================================
    print("\nExtracting features...")
    X_train_f = featurize_column(X_train)
    X_val_f   = featurize_column(X_val)
    X_test_f  = featurize_column(X_test)

    print(f"Done. Shape: {X_train_f.shape}")

    X_train_f = X_train_f.fillna(0)
    X_val_f = X_val_f.fillna(0)
    X_test_f = X_test_f.fillna(0)

    # DoubleCheck
    assert X_train_f.isna().sum().sum() == 0, "NaN in train!"
    assert X_val_f.isna().sum().sum() == 0, "NaN in val!"
    print("No NaNs. Ready for scaling.")

    feature_order = X_train_f.columns.tolist()
    X_val_f = X_val_f.reindex(columns=feature_order, fill_value=0)
    X_test_f = X_test_f.reindex(columns=feature_order, fill_value=0)

    print("Fixed shapes:", X_train_f.shape, X_val_f.shape, X_test_f.shape)

    # =============================================================================
    # BLOCK 3 - Scaling
    # Mandatory for SVM: features on different scales (0/1 flags vs. 0-9999 integers)
    # cause LinearSVC to over-weight large-magnitude features.
    # Scaler is fit ONLY on train set, then applied to val and test.
    # =============================================================================
    scaler = StandardScaler()
    X_train_fs = scaler.fit_transform(X_train_f)
    X_val_fs   = scaler.transform(X_val_f)
    X_test_fs  = scaler.transform(X_test_f)

    # =============================================================================
    # BLOCK 4 - C tuning (val set only, never touch test here)
    # LinearSVC C parameter controls regularization strength.
    # Smaller C = stronger regularization (simpler boundary).
    # Larger C = less regularization (fits training data more tightly).
    # =============================================================================
    print("\n--- C tuning ---")
    print("\nTraining with C=0.01 (pre-validated)...")
    model = CalibratedClassifierCV(
        LinearSVC(C=0.01, max_iter=5000, random_state=42), cv=3
    )
    model.fit(X_train_fs, y_train)
    print("Done.")

    # =============================================================================
    # BLOCK 5 - Full evaluation on val set
    # =============================================================================
    y_pred = model.predict(X_val_fs)

    print("\n--- Val set results ---")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Macro F1:", f1_score(y_val, y_pred, average='macro'))
    print("\nClassification report:")
    print(classification_report(y_val, y_pred))
    print("\nConfusion matrix (MB / BMW / VAG):")
    print(confusion_matrix(y_val, y_pred, labels=['mercedes', 'bmw', 'vag', 'unknown_article']))

    # =============================================================================
    # BLOCK 6 - Final evaluation on test set (touch ONCE, after all tuning is done)
    # =============================================================================
    y_test_pred = model.predict(X_test_fs)

    print("\n--- TEST set results (final) ---")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Macro F1:", f1_score(y_test, y_test_pred, average='macro'))
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred))
    print("\nConfusion matrix (MB / BMW / VAG / UNK):")
    print(confusion_matrix(y_test, y_test_pred, labels=['mercedes', 'bmw', 'vag', 'unknown_article']))

    # =============================================================================
    # BLOCK 7 - Save model + scaler
    # Both must be saved together: scaler is required at inference time.
    # =============================================================================
    n_total = len(df)
    MODEL_NAME = f"linearsvc_atom_{n_total // 1000}k_4cls.pkl"
    MODEL_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "Models", "linearcvs")
    )
    os.makedirs(MODEL_DIR, exist_ok=True)

    save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    joblib.dump({
        "scaler": scaler,
        "model": model,
        "feature_order": feature_order
    }, save_path)
    print(f"\nModel saved → {save_path}")

    feature_order_path = os.path.join(os.path.dirname(__file__), "..", "features_scr", "feature_order.npy")
    np.save(feature_order_path, np.array(feature_order, dtype=object))
    print(f"Saved {len(feature_order)} features to {feature_order_path}")

if __name__ == "__main__":
    main()

