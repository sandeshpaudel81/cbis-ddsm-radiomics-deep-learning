# CBIS-DDSM-R: Radiomics + Deep Learning Fusion for Breast Cancer Classification

## Project Structure
```
cbis_ddsm_project/
├── README.md
├── requirements.txt
├── config.py                    # All hyperparameters & paths
│
├── data/
│   ├── loader.py               # HuggingFace dataset loading + label encoding
│   ├── image_dataset.py        # PyTorch Dataset for ROI images (DICOM/PNG)
│   └── augmentation.py         # Training augmentations
│
├── features/
│   ├── radiomics.py            # Radiomics feature preprocessing + selection
│   └── deep_extractor.py       # CNN & ViT frozen feature extraction
│
├── models/
│   ├── cnn_models.py           # ResNet50, DenseNet121, EfficientNet-B0
│   ├── vit_models.py           # Swin-T, DeiT-Small, MobileViT
│   └── classifiers.py         # XGBoost, MLP, SVM baselines
│
├── fusion/
│   ├── concat_fusion.py        # Simple concatenation fusion
│   └── attention_fusion.py     # Cross-attention fusion module
│
├── evaluation/
│   ├── metrics.py              # AUC, F1, sensitivity, specificity
│   ├── interpretability.py     # SHAP + Grad-CAM + Attention rollout
│   └── visualization.py        # t-SNE/UMAP feature space plots
│
├── utils/
│   └── helpers.py              # Seeding, checkpointing, logging
│
├── train.py                    # Main training script
└── experiment_runner.py        # Runs all 5 experiment arms
```

## Quick Start (Kaggle/Colab)
```bash
pip install -r requirements.txt
python experiment_runner.py --arm all
```
