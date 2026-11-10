# -*- coding: utf-8 -*-
"""数据集读取与图像增强。所有脚本共用，保证训练与推理的预处理一致。"""
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, DATA_ROOT, IMAGE_SIZE, LABELS_CSV  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_rows(split=None, labels_csv=LABELS_CSV):
    with Path(labels_csv).open(encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if split:
        rows = [r for r in rows if r["split"] == split]
    return rows


def build_transform(train: bool, size: int = IMAGE_SIZE):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FrameDataset(Dataset):
    """按 labels.csv 读取抽帧图片。"""

    def __init__(self, rows, transform, class_to_idx, root=DATA_ROOT):
        self.rows = rows
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.root = Path(root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.root / row["file"]).convert("RGB")
        return self.transform(image), self.class_to_idx[row["label"]]


def class_to_idx(classes=None):
    return {name: i for i, name in enumerate(classes or CLASSES)}


def make_loader(rows, train: bool, size: int, batch_size: int, workers: int = 4):
    ds = FrameDataset(rows, build_transform(train, size), class_to_idx())
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=train, num_workers=workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )
