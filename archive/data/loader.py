"""
data/loader.py
Loads CBIS-DDSM-R from HuggingFace, encodes labels, extracts radiomics
and clinical feature columns, and returns clean train/val/test DataFrames.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append("..")
import config


def load_cbis_ddsm_r() -> pd.DataFrame:
    """
    Pull the dataset from HuggingFace and return as a pandas DataFrame.
    Only the CSV is available (no images bundled) — image paths are in
    the 'cropped_image_file_path' column, pointing to TCIA DICOM files.
    """
    print("Loading CBIS-DDSM-R from HuggingFace...")
    ds = load_dataset(config.HF_DATASET_NAME, split="train")
    df = ds.to_pandas()
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary:  MALIGNANT=1, BENIGN / BENIGN_WITHOUT_CALLBACK=0
    Ternary: 0/1/2 for the three classes (optional).
    Adds a 'label' column and drops rows with missing pathology.
    """
    df = df.dropna(subset=["pathology"]).copy()

    if config.LABEL_MODE == "binary":
        df["label"] = (df["pathology"] == "MALIGNANT").astype(int)
    else:
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["pathology"])
        print(f"  Ternary classes: {dict(enumerate(le.classes_))}")

    pos = df["label"].sum()
    neg = len(df) - pos
    print(f"  Labels — Positive (malignant): {pos}, Negative (benign): {neg}")
    return df


def get_radiomics_columns(df: pd.DataFrame) -> list:
    """Return all radiomics feature column names."""
    cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in config.RADIOMICS_PREFIXES)
    ]
    print(f"  Radiomics columns found: {len(cols)}")
    return cols


def get_clinical_columns(df: pd.DataFrame) -> list:
    """Return available clinical metadata columns."""
    available = [c for c in config.CLINICAL_FEATURES if c in df.columns]
    return available


def encode_clinical_features(df: pd.DataFrame,
                              clinical_cols: list) -> pd.DataFrame:
    """One-hot encode categorical clinical features."""
    cat_cols = df[clinical_cols].select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def split_data(df: pd.DataFrame,
               radiomics_cols: list,
               clinical_cols: list):
    """
    Stratified train / val / test split.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test)
    where X includes radiomics + (optionally) clinical features.
    The df_* splits retain all columns (needed for image path lookup).
    """
    # Feature matrix
    feature_cols = radiomics_cols.copy()
    if config.USE_CLINICAL and clinical_cols:
        # Get dummy-encoded clinical col names (after get_dummies)
        enc_clinical = [c for c in df.columns
                        if any(c.startswith(orig) for orig in clinical_cols)]
        feature_cols += enc_clinical

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    # First split: train+val vs test
    X_tv, X_test, y_tv, y_test, idx_tv, idx_test = train_test_split(
        X, y, np.arange(len(df)),
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.SEED
    )
    # Second split: train vs val
    val_size_adj = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_tv, y_tv, idx_tv,
        test_size=val_size_adj,
        stratify=y_tv,
        random_state=config.SEED
    )

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val   = df.iloc[idx_val].reset_index(drop=True)
    df_test  = df.iloc[idx_test].reset_index(drop=True)

    print(f"  Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            df_train, df_val, df_test,
            feature_cols)


def prepare_data():
    """
    Master function: load → encode labels → encode clinical →
    split → return everything needed downstream.
    """
    df = load_cbis_ddsm_r()
    df = encode_labels(df)

    radiomics_cols  = get_radiomics_columns(df)
    clinical_cols   = get_clinical_columns(df)
    df              = encode_clinical_features(df, clinical_cols)

    # Handle NaN in radiomics (fill with column median)
    df[radiomics_cols] = df[radiomics_cols].fillna(
        df[radiomics_cols].median()
    )

    return split_data(df, radiomics_cols, clinical_cols)


if __name__ == "__main__":
    result = prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = result
    print(f"Ready. X_train shape: {X_train.shape}")
