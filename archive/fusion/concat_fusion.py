"""
fusion/concat_fusion.py  +  fusion/attention_fusion.py
Two fusion strategies for combining radiomics + deep learning features.

Strategy A — Concatenation (simple, strong baseline):
    [radiomics_feat | deep_feat] → MLP → sigmoid

Strategy B — Cross-Attention (novel, publishable):
    Radiomics tokens attend to ViT patch-level embeddings (or CNN spatial maps)
    to find which image regions are most relevant given the radiomics context.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import sys
sys.path.append("..")
import config


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY A: Concatenation Fusion
# ══════════════════════════════════════════════════════════════════════════════

class ConcatFusionMLP(nn.Module):
    """
    Simple but effective fusion:
      1. Concatenate radiomics + deep feature vectors
      2. Pass through MLP with BatchNorm + Dropout
      3. Binary classification output

    This is your Arm 4 / Arm 5 baseline fusion model.
    """

    def __init__(self,
                 radiomics_dim: int,
                 deep_dim: int,
                 hidden_dim: int = config.FUSION_HIDDEN,
                 dropout: float  = config.FUSION_DROPOUT,
                 num_classes: int = 2):
        super().__init__()
        fused_dim = radiomics_dim + deep_dim

        self.network = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, radiomics_feat: torch.Tensor,
                deep_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            radiomics_feat : (B, radiomics_dim)
            deep_feat      : (B, deep_dim)
        Returns:
            logits         : (B, num_classes)
        """
        fused = torch.cat([radiomics_feat, deep_feat], dim=-1)
        return self.network(fused)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY B: Cross-Attention Fusion (Novel Contribution)
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion module where radiomics features act as queries
    and deep image features act as keys/values.

    Intuition: the model learns to weight which parts of the image
    representation are most relevant given the radiomics context.

    Architecture:
        radiomics → project → Q  (query)
        deep_feat → project → K, V  (key, value)
        MultiHeadAttention(Q, K, V) → fused → classifier

    This is the novel contribution in your paper (Arm 4b / 5b).
    """

    def __init__(self,
                 radiomics_dim: int,
                 deep_dim: int,
                 proj_dim: int   = 128,
                 n_heads: int    = 4,
                 hidden_dim: int = config.FUSION_HIDDEN,
                 dropout: float  = config.FUSION_DROPOUT,
                 num_classes: int = 2):
        super().__init__()

        # Project both modalities to same dimension for attention
        self.rad_proj  = nn.Linear(radiomics_dim, proj_dim)
        self.deep_proj = nn.Linear(deep_dim, proj_dim)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(proj_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim + proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, radiomics_feat: torch.Tensor,
                deep_feat: torch.Tensor):
        """
        Args:
            radiomics_feat : (B, radiomics_dim) — flat feature vector
            deep_feat      : (B, deep_dim)      — flat feature vector
        Returns:
            logits         : (B, num_classes)
            attn_weights   : (B, 1, 1) — for visualization
        """
        # Project to shared dimension
        Q = self.rad_proj(radiomics_feat).unsqueeze(1)   # (B, 1, proj_dim)
        K = self.deep_proj(deep_feat).unsqueeze(1)        # (B, 1, proj_dim)
        V = K                                             # (B, 1, proj_dim)

        # Cross-attention: radiomics attends to deep features
        attn_out, attn_weights = self.cross_attn(Q, K, V)  # (B, 1, proj_dim)
        attn_out = self.attn_norm(attn_out + Q)             # Residual
        attn_out = attn_out.squeeze(1)                      # (B, proj_dim)

        # Concatenate attended output with projected radiomics (residual path)
        combined = torch.cat([attn_out, Q.squeeze(1)], dim=-1)
        logits   = self.classifier(combined)

        return logits, attn_weights


# ══════════════════════════════════════════════════════════════════════════════
# Sklearn-compatible wrapper for precomputed features (no images needed)
# ══════════════════════════════════════════════════════════════════════════════

class FusionClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper around ConcatFusionMLP.
    Accepts pre-extracted numpy arrays: X = [radiomics | deep_features]

    Usage:
        clf = FusionClassifier(radiomics_dim=30, deep_dim=2048)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
    """

    def __init__(self,
                 radiomics_dim: int,
                 deep_dim: int,
                 hidden_dim: int    = config.FUSION_HIDDEN,
                 dropout: float     = config.FUSION_DROPOUT,
                 lr: float          = config.FUSION_LR,
                 epochs: int        = config.FUSION_EPOCHS,
                 patience: int      = config.FUSION_PATIENCE,
                 device: str        = None):
        self.radiomics_dim = radiomics_dim
        self.deep_dim      = deep_dim
        self.hidden_dim    = hidden_dim
        self.dropout       = dropout
        self.lr            = lr
        self.epochs        = epochs
        self.patience      = patience
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        model = ConcatFusionMLP(
            radiomics_dim=self.radiomics_dim,
            deep_dim=self.deep_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)
        return model

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X: numpy array of shape (N, radiomics_dim + deep_dim)
        Split internally: first radiomics_dim cols = radiomics, rest = deep.
        """
        self.model_    = self._build_model()
        optimizer      = torch.optim.AdamW(self.model_.parameters(), lr=self.lr)
        criterion      = nn.CrossEntropyLoss()
        scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        X_rad   = torch.FloatTensor(X[:, :self.radiomics_dim]).to(self.device)
        X_deep  = torch.FloatTensor(X[:, self.radiomics_dim:]).to(self.device)
        y_torch = torch.LongTensor(y).to(self.device)

        best_val_loss  = float("inf")
        patience_count = 0
        self.train_losses_ = []

        for epoch in range(self.epochs):
            self.model_.train()
            optimizer.zero_grad()
            logits = self.model_(X_rad, X_deep)
            loss   = criterion(logits, y_torch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.train_losses_.append(loss.item())

            # Early stopping on validation set
            if X_val is not None:
                val_loss = self._eval_loss(X_val, y_val, criterion)
                if val_loss < best_val_loss:
                    best_val_loss  = val_loss
                    patience_count = 0
                    self._save_best()
                else:
                    patience_count += 1
                if patience_count >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    self._load_best()
                    break

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, loss={loss.item():.4f}")

        return self

    def _eval_loss(self, X_val, y_val, criterion):
        self.model_.eval()
        with torch.no_grad():
            X_rad   = torch.FloatTensor(X_val[:, :self.radiomics_dim]).to(self.device)
            X_deep  = torch.FloatTensor(X_val[:, self.radiomics_dim:]).to(self.device)
            y_torch = torch.LongTensor(y_val).to(self.device)
            logits  = self.model_(X_rad, X_deep)
            return criterion(logits, y_torch).item()

    def _save_best(self):
        self._best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}

    def _load_best(self):
        if hasattr(self, "_best_state"):
            self.model_.load_state_dict(self._best_state)

    def predict_proba(self, X) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            X_rad  = torch.FloatTensor(X[:, :self.radiomics_dim]).to(self.device)
            X_deep = torch.FloatTensor(X[:, self.radiomics_dim:]).to(self.device)
            logits = self.model_(X_rad, X_deep)
            proba  = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def classes_(self):
        return np.array([0, 1])
