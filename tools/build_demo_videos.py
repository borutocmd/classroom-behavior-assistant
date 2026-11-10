# -*- coding: utf-8 -*-
"""用图片拼出演示视频（公开数据集只有图片、没有视频时用这个）。

输出到 <数据目录>/demo_videos/，**不参与训练**（训练只读 raw/ 与 frames/）：
  demo_mixed.mp4        一段“行为会变化”的课堂视频（每类各占几秒）
  demo_<类别>.mp4        每个类别一段，方便逐个演示

用法：
    python tools/build_demo_videos.py --src <数据目录>/frames --seconds-per-class 4
"""
import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, CLASS_NAMES_CN, DEMO_DIR, FRAME_DIR, ensure_dirs  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path, label: str):
    folder = root / label
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXT)


def build_video(images, out_path: Path, seconds: float, fps: float, size: int, rng, jitter=0.0):
    """把图片循环拼成视频；jitter 控制随机亮度扰动幅度。"""
    if not images:
        return 0
    frames = max(1, int(seconds * fps))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    written = 0
    for i in range(frames):
        img = cv2.imread(str(images[i % len(images)]))
        if img is None:
            continue
        img = cv2.resize(img, (size, size))
        if jitter:
            img = cv2.convertScaleAbs(img, alpha=1.0, beta=rng.randint(-int(jitter), int(jitter)))
        writer.write(img)
        written += 1
    writer.release()
    return written


def main():
    ap = argparse.ArgumentParser(description="图片 → 演示视频")
    ap.add_argument("--src", default=str(FRAME_DIR), help="图片根目录，其下按类别分文件夹")
    ap.add_argument("--out", default=str(DEMO_DIR))
    ap.add_argument("--seconds-per-class", type=float, default=4.0)
    ap.add_argument("--seconds-per-video", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    ensure_dirs()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    available = {label: list_images(src, label) for label in CLASSES}
    missing = [label for label, imgs in available.items() if not imgs]
    if missing:
        print("以下类别没有图片，跳过：" + "、".join(CLASS_NAMES_CN[m] for m in missing))

    # 每类一段
    for label, images in available.items():
        if not images:
            continue
        path = out / f"demo_{label}.mp4"
        n = build_video(images, path, args.seconds_per_video, args.fps, args.size, rng, jitter=10)
        print(f"  demo_{label}.mp4  {n} 帧（用图 {min(len(images), int(args.seconds_per_video * args.fps))} 张）")

    # 混合视频：按类别顺序各占若干秒，模拟一节课的行为变化
    mixed = out / "demo_mixed.mp4"
    writer = None
    total = 0
    for label in CLASSES:
        images = available.get(label) or []
        if not images:
            continue
        per_class = max(1, int(args.seconds_per_class * args.fps))
        for i in range(per_class):
            img = cv2.imread(str(images[i % len(images)]))
            if img is None:
                continue
            img = cv2.resize(img, (args.size, args.size))
            img = cv2.convertScaleAbs(img, alpha=1.0, beta=rng.randint(-10, 10))
            if writer is None:
                writer = cv2.VideoWriter(str(mixed), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (args.size, args.size))
            writer.write(img)
            total += 1
    if writer is not None:
        writer.release()
    print(f"  demo_mixed.mp4  {total} 帧（{total / args.fps:.1f} 秒，"
          f"顺序：{' → '.join(CLASS_NAMES_CN[c] for c in CLASSES if available.get(c))}）")
    print(f"\n演示视频目录：{out}（不参与训练）")
    print("启动网页后可在下拉框里直接选到这些视频。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
