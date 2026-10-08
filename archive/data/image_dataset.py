"""
data/image_dataset.py
PyTorch Dataset that loads cropped ROI images from TCIA DICOM files
(or PNG fallbacks if already converted). Handles grayscale→RGB conversion.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

import sys
sys.path.append("..")
import config


# ─── Image Loading Utilities ─────────────────────────────────────────────────

def load_dicom(path: str) -> np.ndarray:
    """Load a DICOM file and return as uint8 numpy array."""
    dcm = pydicom.dcmread(path)
    arr = dcm.pixel_array.astype(np.float32)
    # Normalize to 0-255
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    return arr.astype(np.uint8)


def load_image(path: str) -> Image.Image:
    """
    Try to load an image from disk. Supports DICOM and standard formats.
    Returns a PIL Image in RGB mode (3-channel).
    """
    if not os.path.exists(path):
        # Return blank image if file not found (graceful degradation)
        return Image.fromarray(
            np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)
        ).convert("RGB")

    ext = os.path.splitext(path)[-1].lower()
    if ext == ".dcm" and PYDICOM_AVAILABLE:
        arr = load_dicom(path)
        img = Image.fromarray(arr)
    else:
        img = Image.open(path)

    # Grayscale → RGB (repeat channel 3 times for pretrained models)
    if img.mode != "RGB":
        img = img.convert("L")
        img = Image.merge("RGB", [img, img, img])

    return img


# ─── Transforms ──────────────────────────────────────────────────────────────

def get_transforms(split: str = "train") -> T.Compose:
    """
    Training: random flip, rotation, colour jitter for augmentation.
    Val/Test: only resize and normalize.
    """
    base = [
        T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ]
    if split == "train":
        augment = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return T.Compose(augment + base)
    return T.Compose(base)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class CBISDDSMDataset(Dataset):
    """
    Loads cropped ROI images for CNN / ViT feature extraction.

    Args:
        df          : DataFrame with 'cropped_image_file_path' and 'label' cols
        image_dir   : Base directory where TCIA images are stored locally
        split       : 'train', 'val', or 'test' (controls augmentation)
        return_path : If True, also return the image path (for debugging)
    """

    def __init__(self, df, image_dir: str = config.IMAGE_BASE_DIR,
                 split: str = "train", return_path: bool = False):
        self.df          = df.reset_index(drop=True)
        self.image_dir   = image_dir
        self.transform   = get_transforms(split)
        self.return_path = return_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        rel_path = row["cropped_image_file_path"]
        label    = int(row["label"])

        full_path = os.path.join(self.image_dir, rel_path)
        img       = load_image(full_path)
        tensor    = self.transform(img)

        if self.return_path:
            return tensor, label, full_path
        return tensor, label


def get_dataloaders(df_train, df_val, df_test,
                    image_dir: str = config.IMAGE_BASE_DIR):
    """
    Build train / val / test DataLoaders.
    Returns dict: {'train': loader, 'val': loader, 'test': loader}
    """
    loaders = {}
    for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        dataset = CBISDDSMDataset(df, image_dir=image_dir, split=split)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
    return loaders


if __name__ == "__main__":
    # Quick sanity check (no real images needed — returns blank arrays)
    import pandas as pd
    dummy_df = pd.DataFrame({
        "cropped_image_file_path": ["fake/path.dcm"] * 10,
        "label": [0, 1] * 5
    })
    ds = CBISDDSMDataset(dummy_df, image_dir="/nonexistent", split="val")
    img, lbl = ds[0]
    print(f"Image tensor shape: {img.shape}, label: {lbl}")
