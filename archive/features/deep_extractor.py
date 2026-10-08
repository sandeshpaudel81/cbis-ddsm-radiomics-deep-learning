"""
features/deep_extractor.py
Frozen deep feature extraction for CNN and ViT architectures via timm.
Passes images through pretrained networks and returns penultimate-layer
embeddings without any fine-tuning (controlled comparison mode).

Supports:
  CNNs  : ResNet50, DenseNet121, EfficientNet-B0
  ViTs  : Swin-Tiny, DeiT-Small, MobileViT-S
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. Run: pip install timm")

import sys
sys.path.append("..")
import config


# ─── Feature Extractor Wrapper ───────────────────────────────────────────────

class DeepFeatureExtractor:
    """
    Loads a pretrained model from timm, removes the classification head,
    and extracts global average pooled feature vectors from images.

    Usage:
        extractor = DeepFeatureExtractor("resnet50")
        features  = extractor.extract(dataloader)  # (N, feature_dim)
    """

    def __init__(self, model_name: str, device: str = None):
        assert TIMM_AVAILABLE, "Install timm: pip install timm"

        self.model_name = model_name
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Look up timm model string
        timm_name = (config.CNN_MODELS.get(model_name)
                     or config.VIT_MODELS.get(model_name)
                     or model_name)

        print(f"  Loading {model_name} ({timm_name}) on {self.device} ...")
        # num_classes=0 removes classifier → returns feature vector directly
        self.model = timm.create_model(
            timm_name,
            pretrained=True,
            num_classes=0,      # No head: outputs feature vector
            global_pool="avg",  # Global average pool
        )
        self.model.eval()
        self.model.to(self.device)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Infer output dimension with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE,
                                device=self.device)
            out = self.model(dummy)
        self.feature_dim = out.shape[-1]
        print(f"  Feature dim: {self.feature_dim}")

    @torch.no_grad()
    def extract(self, dataloader) -> np.ndarray:
        """
        Run all images through the frozen model.
        Returns numpy array of shape (N, feature_dim).
        """
        all_features = []
        all_labels   = []

        for batch in tqdm(dataloader, desc=f"Extracting [{self.model_name}]"):
            images, labels = batch[0], batch[1]
            images = images.to(self.device)
            feats  = self.model(images)          # (B, feature_dim)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

        features = np.concatenate(all_features, axis=0)
        labels   = np.concatenate(all_labels,   axis=0)
        return features, labels

    def extract_single(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extract features for a single image tensor (C, H, W)."""
        self.model.eval()
        with torch.no_grad():
            img = image_tensor.unsqueeze(0).to(self.device)
            feat = self.model(img)
        return feat.cpu().numpy().squeeze()


# ─── Batch Extraction for All Models ─────────────────────────────────────────

def extract_all_models(dataloaders: dict,
                       model_group: str = "cnn") -> dict:
    """
    Extract features from all CNN or ViT models for train/val/test splits.

    Args:
        dataloaders : {'train': loader, 'val': loader, 'test': loader}
        model_group : 'cnn' or 'vit'

    Returns:
        dict of {model_name: {'train': (X, y), 'val': (X, y), 'test': (X, y)}}
    """
    model_dict = config.CNN_MODELS if model_group == "cnn" else config.VIT_MODELS
    results    = {}

    for model_name in model_dict:
        print(f"\n{'='*50}")
        print(f"Processing model: {model_name}")
        extractor = DeepFeatureExtractor(model_name)
        results[model_name] = {}

        for split, loader in dataloaders.items():
            features, labels = extractor.extract(loader)
            results[model_name][split] = (features, labels)
            print(f"  {split}: features shape = {features.shape}")

        # Free GPU memory after each model
        del extractor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ─── Optional: Fine-tuning Wrapper ───────────────────────────────────────────

class FineTuneClassifier(nn.Module):
    """
    Optional fine-tuning wrapper: pretrained backbone + classification head.
    Only used when config.FINETUNE = True.
    """

    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        timm_name   = (config.CNN_MODELS.get(model_name)
                       or config.VIT_MODELS.get(model_name))
        self.backbone = timm.create_model(
            timm_name, pretrained=True,
            num_classes=0, global_pool="avg"
        )
        feat_dim = config.FEATURE_DIM.get(model_name, 512)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats  = self.backbone(x)
        logits = self.classifier(feats)
        return logits

    def get_features(self, x):
        """Returns intermediate features (before classifier)."""
        return self.backbone(x)


if __name__ == "__main__":
    # Smoke test with random tensors (no real images needed)
    from torch.utils.data import DataLoader, TensorDataset
    dummy_imgs   = torch.randn(8, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(dummy_imgs, dummy_labels), batch_size=4)

    ext = DeepFeatureExtractor("resnet50")
    feats, labels = ext.extract(loader)
    print(f"Extracted features: {feats.shape}, labels: {labels.shape}")
