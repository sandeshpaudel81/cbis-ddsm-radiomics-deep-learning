"""
config.py — Central configuration for all hyperparameters, paths, and experiment settings.
Edit this file to change any setting across the whole project.
"""

# ─── Paths ────────────────────────────────────────────────────────────────────
HF_DATASET_NAME   = "helloerikaaa/cbis-ddsm-r"
IMAGE_BASE_DIR    = "/kaggle/input/cbis-ddsm"   # Where TCIA images are stored
                                                  # Change to "/content/cbis-ddsm" for Colab
OUTPUT_DIR        = "./outputs"
CHECKPOINT_DIR    = "./checkpoints"

# ─── Label Settings ───────────────────────────────────────────────────────────
# Binary: MALIGNANT=1, BENIGN + BENIGN_WITHOUT_CALLBACK=0
# Ternary: all three classes separately (optional extension)
LABEL_MODE        = "binary"   # "binary" or "ternary"

# ─── Radiomics Features ───────────────────────────────────────────────────────
# All 93 PyRadiomics feature column prefixes
RADIOMICS_PREFIXES = [
    "original_firstorder_",
    "original_glcm_",
    "original_gldm_",
    "original_glrlm_",
    "original_glszm_",
    "original_ngtdm_",
]
RADIOMICS_N_SELECT  = 30   # Features to keep after selection (mRMR / LASSO)
FEATURE_SELECTION   = "lasso"   # "lasso", "mrmr", "boruta", or "none"

# ─── Clinical / Metadata Features (optional co-variates) ─────────────────────
CLINICAL_FEATURES = [
    "breast_density", "assessment", "subtlety",
    "breast_laterality", "image_view", "abnormality_type",
]
USE_CLINICAL      = True   # Include clinical metadata in fusion

# ─── Image Settings ───────────────────────────────────────────────────────────
IMG_SIZE          = 224    # Resize all ROI crops to this
IMG_CHANNELS      = 3      # Repeat grayscale to 3ch for pretrained models
NORMALIZE_MEAN    = [0.485, 0.456, 0.406]   # ImageNet stats
NORMALIZE_STD     = [0.229, 0.224, 0.225]

# ─── Model Zoo ────────────────────────────────────────────────────────────────
CNN_MODELS = {
    "resnet50":      "resnet50",
    "densenet121":   "densenet121",
    "efficientnet":  "efficientnet_b0",
}
VIT_MODELS = {
    "swin_t":        "swin_tiny_patch4_window7_224",
    "deit_small":    "deit_small_patch16_224",
    "mobilevit":     "mobilevit_s",
}
FEATURE_DIM = {
    "resnet50":      2048,
    "densenet121":   1024,
    "efficientnet":  1280,
    "swin_t":        768,
    "deit_small":    384,
    "mobilevit":     640,
}

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE        = 32
NUM_WORKERS       = 2
SEED              = 42
TEST_SIZE         = 0.2
VAL_SIZE          = 0.1    # From training set
STRATIFY          = True

# Fine-tuning (optional arm)
FINETUNE          = False   # Set True to fine-tune CNNs/ViTs end-to-end
FINETUNE_LR       = 1e-4
FINETUNE_EPOCHS   = 20
FINETUNE_PATIENCE = 5       # Early stopping

# ─── Fusion Settings ──────────────────────────────────────────────────────────
FUSION_TYPE       = "concat"   # "concat" or "cross_attention"
FUSION_HIDDEN     = 256
FUSION_DROPOUT    = 0.3
FUSION_LR         = 1e-3
FUSION_EPOCHS     = 50
FUSION_PATIENCE   = 10

# ─── Experiment Arms ──────────────────────────────────────────────────────────
EXPERIMENT_ARMS = [
    "radiomics_only",    # Arm 1: XGBoost/SVM on 93 radiomics features
    "cnn_only",          # Arm 2: Best CNN features → classifier
    "vit_only",          # Arm 3: Best ViT features → classifier
    "cnn_fusion",        # Arm 4: CNN features + radiomics → fusion
    "vit_fusion",        # Arm 5: ViT features + radiomics → fusion
]

# ─── Subgroup Analyses ────────────────────────────────────────────────────────
SUBGROUPS = {
    "all":           None,                    # Full dataset
    "mass":          ("abnormality_type", "mass"),
    "calcification": ("abnormality_type", "calcification"),
}

# ─── Interpretability ─────────────────────────────────────────────────────────
N_SHAP_SAMPLES    = 200   # Background samples for SHAP KernelExplainer
GRADCAM_LAYER     = "layer4"   # Target layer for Grad-CAM (ResNet)
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.1
