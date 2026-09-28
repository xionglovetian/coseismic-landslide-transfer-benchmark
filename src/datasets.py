from pathlib import Path
from typing import Optional
import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, root: str | Path, split: str, input_size: int, train: bool = False, augmentation: str = "current"):
        if augmentation not in {"none", "current", "strong"}:
            raise ValueError(f"Unknown augmentation mode: {augmentation}")
        self.root = Path(root)
        self.split_dir = self.root / split
        self.image_dir = self.split_dir / "images"
        self.mask_dir = self.split_dir / "masks"
        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing images/masks under {self.split_dir}")
        self.images = sorted(self.image_dir.glob("*.png"))
        if not self.images:
            raise RuntimeError(f"No PNG images found in {self.image_dir}")
        transforms = [A.Resize(input_size, input_size)]
        if train and augmentation == "current":
            transforms += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        elif train and augmentation == "strong":
            transforms += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.30, contrast_limit=0.30, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.35),
                A.ShiftScaleRotate(shift_limit=0.0, scale_limit=(-0.25, 0.25), rotate_limit=0.0, p=0.5),
            ]
        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.float32)
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"].astype(np.float32) / 255.0
        mask = transformed["mask"].astype(np.float32)
        return (
            torch.from_numpy(image.transpose(2, 0, 1)),
            torch.from_numpy(mask[None]),
            image_path.stem,
        )


def build_dataset(root: str | Path, split: str, input_size: int, train: bool = False, augmentation: str = "current"):
    return SegmentationDataset(root, split, input_size, train, augmentation)
