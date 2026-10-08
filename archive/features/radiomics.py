"""
features/radiomics.py
Radiomics feature preprocessing pipeline:
  1. Scale (z-score normalization)
  2. Variance threshold (remove near-zero variance features)
  3. Correlation filter (remove highly correlated redundant features)
  4. Feature selection: LASSO / mRMR / Boruta
  5. Return selected feature names and fitted transformers
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("..")
import config


# ─── Step 1–3: Preprocessing Pipeline ───────────────────────────────────────

class RadiomicsPreprocessor:
    """
    Fits on training data, transforms train/val/test consistently.
    Attributes after fit():
      selected_features_  : list of selected feature names
      n_features_in_      : original feature count
      n_features_out_     : features after selection
    """

    def __init__(self,
                 variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.95,
                 n_select: int = config.RADIOMICS_N_SELECT,
                 method: str = config.FEATURE_SELECTION):
        self.variance_threshold    = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.n_select              = n_select
        self.method                = method

        self.scaler_         = StandardScaler()
        self.var_selector_   = VarianceThreshold(threshold=variance_threshold)
        self.feature_names_  = None   # Input column names (set in fit)
        self.kept_after_var_ = None   # Feature names after variance filter
        self.kept_after_cor_ = None   # Feature names after correlation filter
        self.lasso_selector_ = None   # Fitted LASSO selector (if used)
        self.selected_idx_   = None   # Final selected indices (into scaled array)

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list = None):
        """
        Fit all preprocessing steps on training data.
        X: (n_samples, n_features) raw radiomics matrix
        y: (n_samples,) labels
        feature_names: list of column names (for interpretability)
        """
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = (feature_names if feature_names is not None
                               else [f"feat_{i}" for i in range(X.shape[1])])

        # 1. Variance filter
        self.var_selector_.fit(X)
        X_var = self.var_selector_.transform(X)
        mask_var = self.var_selector_.get_support()
        self.kept_after_var_ = [self.feature_names_[i]
                                 for i in range(len(mask_var)) if mask_var[i]]
        print(f"  Variance filter: {X.shape[1]} → {X_var.shape[1]} features")

        # 2. Correlation filter
        X_var_df = pd.DataFrame(X_var, columns=self.kept_after_var_)
        corr_matrix = X_var_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns
                   if any(upper[col] > self.correlation_threshold)]
        X_var_df = X_var_df.drop(columns=to_drop)
        self.kept_after_cor_ = list(X_var_df.columns)
        X_cor = X_var_df.values
        print(f"  Correlation filter: {X_var.shape[1]} → {X_cor.shape[1]} features "
              f"(threshold={self.correlation_threshold})")

        # 3. Scale
        X_scaled = self.scaler_.fit_transform(X_cor)

        # 4. Feature selection
        if self.method == "lasso":
            X_selected, self.lasso_selector_ = self._lasso_select(X_scaled, y)
        elif self.method == "none":
            X_selected = X_scaled
            self.selected_idx_ = np.arange(X_scaled.shape[1])
        else:
            # Default: keep top-n by variance (simple fallback)
            variances = np.var(X_scaled, axis=0)
            top_idx = np.argsort(variances)[::-1][:self.n_select]
            self.selected_idx_ = top_idx
            X_selected = X_scaled[:, top_idx]

        self.n_features_out_ = X_selected.shape[1]
        self._build_selected_names()
        print(f"  Final radiomics features: {self.n_features_out_}")
        return self

    def _lasso_select(self, X: np.ndarray, y: np.ndarray):
        """LASSO-based feature selection via cross-validated alpha."""
        lasso = LassoCV(cv=5, max_iter=5000, random_state=config.SEED)
        selector = SelectFromModel(lasso, max_features=self.n_select)
        selector.fit(X, y)
        self.lasso_selector_ = selector
        self.selected_idx_   = np.where(selector.get_support())[0]
        return selector.transform(X), selector

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted preprocessing to new data."""
        # Variance filter
        X_var = self.var_selector_.transform(X)
        # Correlation filter (drop same columns)
        X_var_df = pd.DataFrame(X_var, columns=self.kept_after_var_)
        X_cor = X_var_df[self.kept_after_cor_].values
        # Scale
        X_scaled = self.scaler_.transform(X_cor)
        # Feature selection
        if self.method == "lasso" and self.lasso_selector_ is not None:
            return self.lasso_selector_.transform(X_scaled)
        return X_scaled[:, self.selected_idx_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                      feature_names: list = None) -> np.ndarray:
        self.fit(X, y, feature_names)
        return self.transform(X)

    def _build_selected_names(self):
        """Map selected_idx_ back to human-readable feature names."""
        after_cor_names = self.kept_after_cor_
        if self.selected_idx_ is not None:
            self.selected_features_ = [after_cor_names[i]
                                        for i in self.selected_idx_
                                        if i < len(after_cor_names)]
        else:
            self.selected_features_ = after_cor_names

    def get_feature_names(self) -> list:
        return self.selected_features_


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(42)
    X_dummy = rng.random((200, 93)).astype(np.float32)
    y_dummy = rng.integers(0, 2, size=200)
    names   = [f"feat_{i}" for i in range(93)]

    prep = RadiomicsPreprocessor(method="lasso", n_select=20)
    X_proc = prep.fit_transform(X_dummy, y_dummy, names)
    print(f"Output shape: {X_proc.shape}")
    print(f"Selected features: {prep.get_feature_names()[:5]} ...")
