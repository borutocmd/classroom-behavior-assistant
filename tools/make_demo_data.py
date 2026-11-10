# -*- coding: utf-8 -*-
"""合成一小份“课堂场景”数据，用于在真实数据到位前验证整条流程。

⚠️ 只用于流程自检（跑通抽帧→划分→训练→评价→演示），**不能**写进论文当实验结果。

用法：
    python tools/make_demo_data.py --out <data>/raw --per-class 8
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, CLASSES  # noqa: E402

W = H = 320


def render(label: str, rng: random.Random) -> Image.Image:
    """按行为类别画一张示意图：背景 + 桌子 + 头 + 类别特征。"""
    img = Image.new("RGB", (W, H), (238, 240, 245))
    d = ImageDraw.Draw(img)
    # 背景（黑板/墙面）
    d.rectangle([0, 0, W, int(H * 0.45)], fill=(205, 215, 225))
    # 课桌
    desk_y = int(H * 0.72)
    d.rectangle([int(W * 0.12), desk_y, int(W * 0.88), desk_y + int(H * 0.08)], fill=(150, 120, 90))

    cx = rng.randint(int(W * 0.40), int(W * 0.60))
    head_r = rng.randint(24, 30)
    dy = {"listening": 0, "raising_hand": 0, "reading": 16, "writing": 24, "phone": 22, "desk": 40}[label]
    head_y = int(H * 0.34) + dy
    d.ellipse([cx - head_r, head_y - head_r, cx + head_r, head_y + head_r], fill=(245, 214, 180))
    d.rectangle([cx - head_r, head_y + head_r, cx + head_r, desk_y], fill=(70, 110, 170))  # 身体

    if label == "raising_hand":
        d.line([cx + head_r - 4, head_y + 10, cx + head_r + 40, head_y - 60], fill=(70, 110, 170), width=7)
    elif label == "reading":
        d.rectangle([cx - 60, desk_y - 34, cx + 60, desk_y - 6], fill=(250, 250, 245), outline=(120, 120, 120), width=3)
    elif label == "writing":
        d.rectangle([cx - 46, desk_y - 20, cx + 10, desk_y - 6], fill=(250, 250, 245), outline=(120, 120, 120), width=2)
        d.line([cx + 10, desk_y - 14, cx + 44, desk_y - 26], fill=(70, 110, 170), width=6)
    elif label == "phone":
        d.rectangle([cx + 14, desk_y - 30, cx + 44, desk_y - 6], fill=(40, 40, 45))
        d.line([cx + 20, head_y + head_r + 6, cx + 26, desk_y - 24], fill=(70, 110, 170), width=6)
    elif label == "desk":
        d.ellipse([cx - head_r - 6, desk_y - head_r - 12, cx + head_r + 6, desk_y + 2], fill=(245, 214, 180))

    arr = np.asarray(img).astype(np.int16)
    arr += rng.randint(-18, 18)                       # 亮度扰动
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if rng.random() < 0.5:                            # 水平翻转增强多样性
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def main():
    parser = argparse.ArgumentParser(description="生成合成课堂场景数据（仅用于流程自检）")
    parser.add_argument("--out", default=str(RAW_DIR))
    parser.add_argument("--per-class", type=int, default=8, help="每类生成多少个来源")
    parser.add_argument("--frames", type=int, default=6, help="每个来源多少帧")
    parser.add_argument("--videos", type=int, default=2, help="每类含几段 mp4（其余为图片）")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_root = Path(args.out)
    total = 0
    for label in CLASSES:
        label_dir = out_root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            stem = f"demo_{label}_{i:02d}"
            if i < args.videos:
                writer = cv2.VideoWriter(str(label_dir / f"{stem}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
                for _ in range(args.frames * 5):      # 10 fps × 3 秒
                    writer.write(cv2.cvtColor(np.asarray(render(label, rng)), cv2.COLOR_RGB2BGR))
                writer.release()
                total += args.frames
            else:
                for k in range(args.frames):
                    render(label, rng).save(label_dir / f"{stem}_{k:02d}.jpg", quality=92)
                    total += 1
        print(f"  {label:14s} 已生成 {args.per_class} 个来源")
    print(f"\n合成数据写入 {out_root}，预计抽帧后约 {total} 帧")
    print("提醒：仅用于流程自检，论文实验必须用真实采集数据。")


if __name__ == "__main__":
    main()
