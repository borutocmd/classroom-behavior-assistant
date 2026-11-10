# -*- coding: utf-8 -*-
"""把一批图片拼成一段演示视频（不想录制视频时的替代方案）。

典型用途：从课堂行为数据集里挑若干张不同行为的图片，拼成 20–30 秒的“模拟课堂”视频，
用来自检时间线、统计与报告整条链路，演示也能用。

用法：
    python tools/images_to_video.py --src <数据目录>/frames --out /tmp/demo_class.mp4 \
        --order listening,raising_hand,reading,writing,phone,desk --seconds-per-class 4
"""
import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, CLASS_NAMES_CN  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description="图片序列 → 演示视频")
    ap.add_argument("--src", required=True, help="图片根目录，其下按类别分文件夹")
    ap.add_argument("--out", default="/tmp/demo_class.mp4")
    ap.add_argument("--order", default=",".join(CLASSES), help="行为出现顺序，逗号分隔")
    ap.add_argument("--seconds-per-class", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.src)
    order = [c.strip() for c in args.order.split(",") if c.strip()]
    frames_per_class = int(args.seconds_per_class * args.fps)
    writer = None
    total, used = 0, {}
    for label in order:
        folder = src / label
        if not folder.exists():
            print(f"跳过不存在的类别目录：{folder}")
            continue
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXT)
        if not images:
            print(f"跳过没有图片的类别：{folder}")
            continue
        used[label] = len(images)
        for i in range(frames_per_class):
            path = images[i % len(images)]          # 循环取图，凑够时长
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            frame = cv2.resize(frame, (args.size, args.size))
            if rng.random() < 0.5:                   # 轻微亮度扰动，模拟真实视频
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=rng.randint(-10, 10))
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (args.size, args.size))
            writer.write(frame)
            total += 1
    if writer is not None:
        writer.release()
    print(f"生成 {total} 帧（{total / args.fps:.1f} 秒）→ {args.out}")
    print("各类别用图数量：" + "、".join(f"{CLASS_NAMES_CN.get(k, k)} {v} 张" for k, v in used.items()))
    print("行为顺序：" + " → ".join(CLASS_NAMES_CN.get(c, c) for c in used))


if __name__ == "__main__":
    main()
