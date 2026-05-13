# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import scipy.io
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(split: str, img_size: int = 224):
    """Transforms used for ImageNet-pretrained CNN fine-tuning."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Flower102Dataset(Dataset):
    """
    Oxford 102 Flowers dataset using official MATLAB files.

    setid.mat split names:
      - trnid: train ids
      - valid: validation ids
      - tstid: test ids

    imagelabels.mat stores labels from 1 to 102, so we convert them to 0 to 101
    for PyTorch CrossEntropyLoss.
    """
    SPLIT_KEY = {
        "train": "trnid",
        "val": "valid",
        "valid": "valid",
        "validation": "valid",
        "test": "tstid",
    }

    def __init__(self, root: str | Path, split: str, transform=None):
        self.root = Path(root)
        self.split = split
        if split not in self.SPLIT_KEY:
            raise ValueError(f"Unknown split={split}. Choose from train/val/test.")

        self.img_dir = self.root / "jpg"
        self.labels_path = self.root / "imagelabels.mat"
        self.setid_path = self.root / "setid.mat"

        missing = [p for p in [self.img_dir, self.labels_path, self.setid_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Dataset files are missing. Run: python download_data.py --data_root ./data/flowers102\n"
                f"Missing: {missing}"
            )

        labels = scipy.io.loadmat(self.labels_path)["labels"].squeeze()
        setid = scipy.io.loadmat(self.setid_path)
        ids = setid[self.SPLIT_KEY[split]].squeeze()

        self.samples = []
        for image_id in ids:
            image_id = int(image_id)
            img_path = self.img_dir / f"image_{image_id:05d}.jpg"
            label = int(labels[image_id - 1]) - 1
            self.samples.append((img_path, label))

        self.transform = transform if transform is not None else build_transforms("train" if split == "train" else "val")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
