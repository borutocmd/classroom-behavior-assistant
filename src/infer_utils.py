# -*- coding: utf-8 -*-
"""推理工具：加载训练好的模型、对单张图/一批图做预测。训练与推理共用同一套预处理。"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, MODEL_DIR, WEIGHT_DIR  # noqa: E402
from src.dataset import build_transform  # noqa: E402

BACKBONE_DIR = WEIGHT_DIR / "resnet18"


def load_model(tag: str = "hf_frozen", device=None):
    """读取 models/<tag>/config.json 与 model.pt，返回 (model, classes, image_size)。"""
    save_dir = Path(MODEL_DIR) / tag
    cfg_path = save_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"没有找到模型 {save_dir}，先运行 src/train_cnn.py")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    classes = cfg.get("classes", CLASSES)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if cfg["backbone"] == "hf":
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(
            str(BACKBONE_DIR), num_labels=len(classes), ignore_mismatched_sizes=True)
    else:
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

    state = torch.load(save_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, classes, cfg.get("image_size", 224)


@torch.inference_mode()
def predict_batch(model, images, image_size=224, device=None):
    """images: PIL 图片列表；返回 (类别下标列表, 置信度列表, 四类概率矩阵)。"""
    device = device or next(model.parameters()).device
    transform = build_transform(False, image_size)
    batch = torch.stack([transform(im.convert("RGB")) for im in images]).to(device)
    logits = model(pixel_values=batch).logits if hasattr(model, "config") else model(batch)
    probs = torch.softmax(logits.float(), dim=1).cpu()
    conf, idx = probs.max(dim=1)
    return idx.tolist(), conf.tolist(), probs.numpy()


def predict_image(model, path, image_size=224):
    with Image.open(path) as im:
        idx, conf, probs = predict_batch(model, [im], image_size)
    return idx[0], conf[0], probs[0]
