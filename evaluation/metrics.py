"""
evaluation/metrics.py + evaluation/interpretability.py

METRICS:
  - AUC-ROC, F1, Sensitivity, Specificity, Accuracy
  - DeLong's test for AUC comparison between arms
  - Confusion matrix + classification report

INTERPRETABILITY:
  - SHAP (for XGBoost / fusion classifiers)
  - Grad-CAM (for CNN models)
  - Attention Rollout (for ViT models)
  - t-SNE / UMAP feature space visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)

import sys
sys.path.append("..")
import config

Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_proba: np.ndarray,
                    arm_name: str = "") -> dict:
    """
    Compute full set of evaluation metrics.
    Returns dict with all metrics (for easy DataFrame aggregation).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-8)   # Recall / TPR
    specificity = tn / (tn + fp + 1e-8)   # TNR
    ppv         = tp / (tp + fp + 1e-8)   # Precision
    npv         = tn / (tn + fn + 1e-8)

    metrics = {
        "arm":          arm_name,
        "auc":          roc_auc_score(y_true, y_proba[:, 1]),
        "f1":           f1_score(y_true, y_pred),
        "accuracy":     accuracy_score(y_true, y_pred),
        "sensitivity":  sensitivity,
        "specificity":  specificity,
        "ppv":          ppv,
        "npv":          npv,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

    print(f"\n{'─'*50}")
    print(f"  Arm: {arm_name}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:15s}: {v:.4f}")
    return metrics


def compare_arms(results: dict) -> pd.DataFrame:
    """
    Collect metrics across all experiment arms into a summary DataFrame.
    Args:
        results: {arm_name: {'y_true': ..., 'y_pred': ..., 'y_proba': ...}}
    """
    rows = []
    for arm_name, res in results.items():
        m = compute_metrics(
            res["y_true"], res["y_pred"], res["y_proba"], arm_name
        )
        rows.append(m)
    df = pd.DataFrame(rows).set_index("arm")
    print("\n\n=== ARM COMPARISON ===")
    print(df[["auc", "f1", "accuracy", "sensitivity", "specificity"]].to_string())
    return df


def plot_roc_curves(results: dict, save_path: str = None):
    """Plot ROC curves for all arms on one figure."""
    plt.figure(figsize=(8, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    for (arm_name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"][:, 1])
        auc = roc_auc_score(res["y_true"], res["y_proba"][:, 1])
        plt.plot(fpr, tpr, color=color, linewidth=2,
                 label=f"{arm_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curves — All Experiment Arms")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = save_path or f"{config.OUTPUT_DIR}/roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, arm_name: str, save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"])
    plt.title(f"Confusion Matrix — {arm_name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = save_path or f"{config.OUTPUT_DIR}/cm_{arm_name}.png"
    plt.savefig(path, dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# INTERPRETABILITY — SHAP
# ══════════════════════════════════════════════════════════════════════════════

def shap_analysis(model,
                  X_train: np.ndarray,
                  X_test: np.ndarray,
                  feature_names: list,
                  arm_name: str = "fusion",
                  n_background: int = config.N_SHAP_SAMPLES):
    """
    SHAP analysis for XGBoost or FusionClassifier.
    Produces beeswarm plot + top-feature bar chart.
    """
    try:
        import shap
    except ImportError:
        print("Install shap: pip install shap"); return

    print(f"  Computing SHAP values for {arm_name} ...")

    # Use TreeExplainer for XGBoost (fast), KernelExplainer otherwise
    if hasattr(model, "get_booster"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]   # Class 1 (malignant)
    else:
        background = shap.kmeans(X_train, n_background)
        explainer  = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], background
        )
        shap_values = explainer.shap_values(
            X_test[:100], nsamples=100
        )

    # Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      max_display=20, show=False)
    plt.title(f"SHAP Summary — {arm_name}")
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/shap_{arm_name}.png", dpi=150,
                bbox_inches="tight")
    plt.show()

    # Top features
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx   = np.argsort(mean_shap)[::-1][:20]
    top_df    = pd.DataFrame({
        "feature":    [feature_names[i] for i in top_idx],
        "mean_|shap|": mean_shap[top_idx]
    })
    print(f"\n  Top 10 SHAP features ({arm_name}):")
    print(top_df.head(10).to_string(index=False))
    return top_df


# ══════════════════════════════════════════════════════════════════════════════
# INTERPRETABILITY — Grad-CAM
# ══════════════════════════════════════════════════════════════════════════════

def gradcam_visualization(model, image_tensor, target_layer_name: str,
                          class_idx: int = 1, save_path: str = None):
    """
    Grad-CAM for CNN models. Highlights discriminative image regions.
    Requires pytorch-grad-cam: pip install grad-cam
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("Install: pip install grad-cam"); return

    # Find target layer by name
    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    if target_layer is None:
        print(f"Layer {target_layer_name} not found."); return

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0),
                        targets=None)[0]

    # Convert tensor to RGB numpy for overlay
    img_np = image_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.imshow(img_np, cmap="gray"); plt.title("Original ROI")
    plt.subplot(1, 3, 2); plt.imshow(grayscale_cam, cmap="jet"); plt.title("Grad-CAM Heatmap")
    plt.subplot(1, 3, 3); plt.imshow(visualization); plt.title("Overlay")
    for ax in plt.gcf().axes: ax.axis("off")
    plt.tight_layout()
    path = save_path or f"{config.OUTPUT_DIR}/gradcam.png"
    plt.savefig(path, dpi=150)
    plt.show()
    return grayscale_cam


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SPACE — t-SNE / UMAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_space(features_dict: dict,
                       labels: np.ndarray,
                       method: str = "umap",
                       save_path: str = None):
    """
    Visualize and compare feature spaces from different modalities.

    Args:
        features_dict: {'radiomics': X_rad, 'resnet50': X_cnn, 'swin_t': X_vit}
        labels: ground truth labels
        method: 'umap' or 'tsne'
    """
    n_models = len(features_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, feats) in zip(axes, features_dict.items()):
        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(
                    n_neighbors=config.UMAP_N_NEIGHBORS,
                    min_dist=config.UMAP_MIN_DIST,
                    random_state=config.SEED
                )
            except ImportError:
                print("Install umap: pip install umap-learn"); return
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=config.SEED,
                           perplexity=30, n_iter=1000)

        embedded = reducer.fit_transform(feats)
        scatter  = ax.scatter(embedded[:, 0], embedded[:, 1],
                              c=labels, cmap="coolwarm",
                              alpha=0.6, s=10, edgecolors="none")
        ax.set_title(f"{name}\n({method.upper()})")
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        plt.colorbar(scatter, ax=ax, label="0=Benign, 1=Malignant")

    plt.suptitle(f"Feature Space Comparison — {method.upper()}", fontsize=13)
    plt.tight_layout()
    path = save_path or f"{config.OUTPUT_DIR}/feature_space_{method}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")
