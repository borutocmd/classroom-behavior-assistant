# -*- coding: utf-8 -*-
"""下载 ResNet18 预训练权重到本地（走 hf-mirror，可离线复用）。

本机直连不了 download.pytorch.org，所以用 HuggingFace 的 microsoft/resnet-18，
下载后保存到 <数据目录>/weights/resnet18，训练时完全离线加载。

用法：
    python tools/fetch_backbone.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.pop("HF_HUB_OFFLINE", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import WEIGHT_DIR, ensure_dirs  # noqa: E402

MODEL_ID = "microsoft/resnet-18"


def main():
    ensure_dirs()
    target = WEIGHT_DIR / "resnet18"
    if (target / "config.json").exists():
        print(f"已存在：{target}（如需重新下载请先删除该目录）")
        return 0

    t0 = time.time()
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    print(f"从 {os.environ['HF_ENDPOINT']} 下载 {MODEL_ID} …")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    target.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(target)
    model.save_pretrained(target)
    size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1024 ** 2
    print(f"完成：{target}（{size_mb:.1f} MB，用时 {time.time() - t0:.1f} 秒）")

    # 立刻验证一次离线加载，避免训练时才发现问题
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoModelForImageClassification as AM
    reloaded = AM.from_pretrained(str(target), num_labels=6, ignore_mismatched_sizes=True)
    n_linear = sum(1 for m in reloaded.modules() if type(m).__name__ == "Linear")
    print(f"离线加载验证通过：{type(reloaded).__name__}，Linear 层 {n_linear} 个，参数 "
          f"{sum(p.numel() for p in reloaded.parameters()):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
