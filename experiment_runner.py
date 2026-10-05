"""
experiment_runner.py
Orchestrates all 5 experiment arms end-to-end.
Run from Kaggle/Colab:
    python experiment_runner.py --arm all
    python experiment_runner.py --arm radiomics_only
    python experiment_runner.py --arm vit_fusion
"""

import argparse
import numpy as np
import torch
import random
import os

import config
from data.loader import prepare_data
from data.image_dataset import get_dataloaders
from features.radiomics import RadiomicsPreprocessor
from features.deep_extractor import DeepFeatureExtractor
from models.classifiers import get_classifier, get_xgboost_tuned
from fusion.concat_fusion import FusionClassifier
from evaluation.metrics import (
    compute_metrics, compare_arms,
    plot_roc_curves, plot_confusion_matrix,
    shap_analysis, plot_feature_space
)


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)


# ─── Step 1: Data & Radiomics ─────────────────────────────────────────────────

def prepare_radiomics(X_train, X_val, X_test, y_train, feature_names):
    """Fit radiomics preprocessor on train, transform all splits."""
    print("\n[1/4] Radiomics preprocessing ...")
    prep = RadiomicsPreprocessor(
        method=config.FEATURE_SELECTION,
        n_select=config.RADIOMICS_N_SELECT
    )
    X_rad_train = prep.fit_transform(X_train, y_train, feature_names)
    X_rad_val   = prep.transform(X_val)
    X_rad_test  = prep.transform(X_test)
    rad_names   = prep.get_feature_names()
    print(f"  Selected {X_rad_train.shape[1]} radiomics features")
    return X_rad_train, X_rad_val, X_rad_test, rad_names, prep


# ─── Step 2: Deep Feature Extraction ─────────────────────────────────────────

def extract_deep_features(df_train, df_val, df_test, model_name: str):
    """Extract frozen deep features for one CNN or ViT model."""
    print(f"\n[2/4] Deep feature extraction — {model_name} ...")
    loaders  = get_dataloaders(df_train, df_val, df_test)
    extractor = DeepFeatureExtractor(model_name)
    splits   = {}
    for split, loader in loaders.items():
        feats, labels = extractor.extract(loader)
        splits[split] = (feats, labels)
        print(f"  {split}: {feats.shape}")
    return splits


# ─── Arm 1: Radiomics Only ────────────────────────────────────────────────────

def run_radiomics_only(X_rad_train, X_rad_val, X_rad_test,
                       y_train, y_val, y_test, rad_names) -> dict:
    print("\n" + "="*60)
    print("ARM 1: Radiomics Only (XGBoost)")
    print("="*60)

    clf = get_xgboost_tuned(X_rad_train, y_train)
    y_proba = clf.predict_proba(X_rad_test)
    y_pred  = clf.predict(X_rad_test)

    # SHAP analysis
    shap_analysis(clf, X_rad_train, X_rad_test, rad_names,
                  arm_name="radiomics_only")
    plot_confusion_matrix(y_test, y_pred, "radiomics_only")

    return {"y_true": y_test, "y_pred": y_pred, "y_proba": y_proba,
            "model": clf}


# ─── Arm 2/3: DL Features Only ───────────────────────────────────────────────

def run_dl_only(deep_splits: dict, model_name: str) -> dict:
    arm_name = f"{model_name}_only"
    print(f"\n{'='*60}")
    print(f"ARM 2/3: DL Features Only — {model_name}")
    print("="*60)

    X_train, y_train = deep_splits["train"]
    X_val,   y_val   = deep_splits["val"]
    X_test,  y_test  = deep_splits["test"]

    # Combine train+val for final fit
    X_tv = np.concatenate([X_train, X_val], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)

    clf     = get_xgboost_tuned(X_tv, y_tv)
    y_proba = clf.predict_proba(X_test)
    y_pred  = clf.predict(X_test)

    plot_confusion_matrix(y_test, y_pred, arm_name)
    return {"y_true": y_test, "y_pred": y_pred, "y_proba": y_proba,
            "model": clf}


# ─── Arms 4/5: Fusion (Radiomics + DL) ───────────────────────────────────────

def run_fusion(X_rad_train, X_rad_val, X_rad_test,
               y_train, y_val, y_test,
               deep_splits: dict,
               model_name: str,
               rad_names: list) -> dict:
    arm_name = f"{model_name}_fusion"
    print(f"\n{'='*60}")
    print(f"ARM 4/5: Fusion — Radiomics + {model_name}")
    print("="*60)

    X_deep_train, _ = deep_splits["train"]
    X_deep_val,   _ = deep_splits["val"]
    X_deep_test,  _ = deep_splits["test"]

    # Concatenate radiomics + deep features
    X_train_fused = np.concatenate([X_rad_train, X_deep_train], axis=1)
    X_val_fused   = np.concatenate([X_rad_val,   X_deep_val],   axis=1)
    X_test_fused  = np.concatenate([X_rad_test,  X_deep_test],  axis=1)

    rad_dim  = X_rad_train.shape[1]
    deep_dim = X_deep_train.shape[1]
    all_names = rad_names + [f"{model_name}_feat_{i}" for i in range(deep_dim)]

    # Fusion MLP classifier
    clf = FusionClassifier(radiomics_dim=rad_dim, deep_dim=deep_dim)
    clf.fit(X_train_fused, y_train,
            X_val=X_val_fused, y_val=y_val)

    y_proba = clf.predict_proba(X_test_fused)
    y_pred  = clf.predict(X_test_fused)

    # SHAP on radiomics portion only (interpretable features)
    shap_analysis(
        clf, X_train_fused, X_test_fused[:100],
        feature_names=all_names, arm_name=arm_name
    )
    plot_confusion_matrix(y_test, y_pred, arm_name)

    return {"y_true": y_test, "y_pred": y_pred, "y_proba": y_proba,
            "model": clf}


# ─── Main Orchestrator ────────────────────────────────────────────────────────

def run_all_experiments(arms_to_run: list = None):
    set_seed()
    arms_to_run = arms_to_run or config.EXPERIMENT_ARMS

    # ── Data Loading ──
    print("\n" + "="*60)
    print("LOADING DATA")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     df_train, df_val, df_test,
     feature_cols) = prepare_data()

    rad_dim   = sum(1 for c in feature_cols
                    if any(c.startswith(p) for p in config.RADIOMICS_PREFIXES))
    rad_cols  = feature_cols[:rad_dim]

    # ── Radiomics Preprocessing ──
    X_rad_train, X_rad_val, X_rad_test, rad_names, prep = prepare_radiomics(
        X_train[:, :rad_dim], X_val[:, :rad_dim], X_test[:, :rad_dim],
        y_train, rad_cols
    )

    results = {}

    # ── Arm 1: Radiomics Only ──
    if "radiomics_only" in arms_to_run:
        results["radiomics_only"] = run_radiomics_only(
            X_rad_train, X_rad_val, X_rad_test,
            y_train, y_val, y_test, rad_names
        )

    # ── Arms 2–5: Require images ──
    image_arms = [a for a in arms_to_run if a != "radiomics_only"]
    if image_arms:
        # Choose best CNN and best ViT (configurable)
        cnn_model = "resnet50"    # or "densenet121", "efficientnet"
        vit_model = "swin_t"      # or "deit_small", "mobilevit"

        if any("cnn" in a for a in image_arms):
            cnn_splits = extract_deep_features(
                df_train, df_val, df_test, cnn_model
            )
            if "cnn_only" in image_arms:
                results["cnn_only"] = run_dl_only(cnn_splits, cnn_model)
            if "cnn_fusion" in image_arms:
                results["cnn_fusion"] = run_fusion(
                    X_rad_train, X_rad_val, X_rad_test,
                    y_train, y_val, y_test,
                    cnn_splits, cnn_model, rad_names
                )

        if any("vit" in a for a in image_arms):
            vit_splits = extract_deep_features(
                df_train, df_val, df_test, vit_model
            )
            if "vit_only" in image_arms:
                results["vit_only"] = run_dl_only(vit_splits, vit_model)
            if "vit_fusion" in image_arms:
                results["vit_fusion"] = run_fusion(
                    X_rad_train, X_rad_val, X_rad_test,
                    y_train, y_val, y_test,
                    vit_splits, vit_model, rad_names
                )

        # ── Feature Space Visualization ──
        feat_dict = {"radiomics": X_rad_test}
        if "cnn_only" in results:
            feat_dict[cnn_model] = cnn_splits["test"][0]
        if "vit_only" in results:
            feat_dict[vit_model] = vit_splits["test"][0]
        if len(feat_dict) > 1:
            plot_feature_space(feat_dict, y_test, method="umap")

    # ── Summary ──
    if results:
        summary_df = compare_arms(results)
        plot_roc_curves(results)
        summary_df.to_csv(f"{config.OUTPUT_DIR}/results_summary.csv")
        print(f"\nResults saved to {config.OUTPUT_DIR}/results_summary.csv")

    return results


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBIS-DDSM-R Experiment Runner")
    parser.add_argument(
        "--arm", type=str, default="all",
        choices=["all"] + config.EXPERIMENT_ARMS,
        help="Which experiment arm(s) to run"
    )
    args = parser.parse_args()

    arms = config.EXPERIMENT_ARMS if args.arm == "all" else [args.arm]
    run_all_experiments(arms)
