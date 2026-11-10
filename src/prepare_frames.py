# -*- coding: utf-8 -*-
"""抽帧与自动标注。

输入目录结构（标签来自文件夹名，无需手工标注）：
    raw/<类别>/xxx.mp4          视频
    raw/<类别>/xxx.jpg          图片
类别可用中文或常见英文别名（见 config.LABEL_ALIASES）。

输出：
    frames/<类别>/<来源>_0001.jpg
    labels.csv                  列为 file,label,source,frame_idx,split

用法：
    python src/prepare_frames.py --src raw --fps 1 --size 256
    python src/prepare_frames.py --clean       # 先清空 frames 再抽
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (CLASSES, DATA_ROOT, FRAME_DIR, LABELS_CSV, RAW_DIR,  # noqa: E402
                    ensure_dirs, norm_label)

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIFF_THRESHOLD = 3.0  # 相邻帧平均绝对差低于该值视为重复（画面几乎没动）


def frame_diff(a, b):
    return float(np.mean(cv2.absdiff(a, b)))


def extract_video(path: Path, out_dir: Path, label: str, fps: float, size: int):
    """按目标 fps 抽帧，顺带跳过几乎相同的帧。返回写入的行。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  !! 打不开视频：{path}")
        return []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / max(fps, 0.01))))
    rows, saved, idx, last = [], 0, 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            if last is None or frame_diff(small, last) >= DIFF_THRESHOLD:
                out = out_dir / f"{path.stem}_{saved:04d}.jpg"
                cv2.imwrite(str(out), small, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                rows.append({"file": str(out.relative_to(DATA_ROOT)), "label": label,
                             "source": path.stem, "frame_idx": saved, "split": ""})
                last, saved = small, saved + 1
        idx += 1
    cap.release()
    return rows


def collect_images(path: Path, out_dir: Path, label: str, size: int):
    rows = []
    for i, img_path in enumerate(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT)):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  !! 读不到图片：{img_path}")
            continue
        small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        out = out_dir / f"{img_path.stem}_{i:04d}.jpg"
        cv2.imwrite(str(out), small, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        # 同一批连拍（xxx_01、xxx_02…）算同一个来源，避免划分时相邻图片被拆到不同集合
        source = re.sub(r"_\d+$", "", img_path.stem) or img_path.stem
        rows.append({"file": str(out.relative_to(DATA_ROOT)), "label": label,
                     "source": source, "frame_idx": i, "split": ""})
    return rows


def main():
    parser = argparse.ArgumentParser(description="抽帧与自动标注")
    parser.add_argument("--src", default=str(RAW_DIR), help="原始视频/图片目录（默认 data/raw）")
    parser.add_argument("--fps", type=float, default=1.0, help="每秒抽多少帧（默认 1）")
    parser.add_argument("--size", type=int, default=256, help="保存的方形边长（默认 256）")
    parser.add_argument("--clean", action="store_true", help="抽帧前清空 frames 目录")
    args = parser.parse_args()

    ensure_dirs()
    src = Path(args.src)
    if not src.exists():
        print(f"源目录不存在：{src}\n先按 docs/数据采集指南.md 放好视频，例如 raw/writing/zhangsan_01.mp4")
        return 1

    if args.clean and FRAME_DIR.exists():
        for old in FRAME_DIR.rglob("*.jpg"):
            old.unlink()
        print(f"已清空 {FRAME_DIR}")

    rows, per_label = [], {}
    for label_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        label = norm_label(label_dir.name)
        if label is None:
            print(f"跳过无法识别的类别目录：{label_dir.name}")
            continue
        if label not in CLASSES:
            print(f"跳过未启用的类别目录：{label_dir.name}（识别为 {label}，当前 CLASSES 里没有它）")
            continue
        out_dir = FRAME_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        got = []
        for f in sorted(label_dir.iterdir()):
            if f.suffix.lower() in VIDEO_EXT:
                got += extract_video(f, out_dir, label, args.fps, args.size)
        got += collect_images(label_dir, out_dir, label, args.size)
        rows += got
        per_label[label] = len(got)
        print(f"  {label:14s} {len(got):5d} 帧")

    with LABELS_CSV.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["file", "label", "source", "frame_idx", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n共 {len(rows)} 帧，写入 {LABELS_CSV}")
    if per_label:
        lo = min(per_label.values())
        hi = max(per_label.values())
        if lo < 100:
            print(f"提示：样本最少的类别只有 {lo} 帧，建议补录到每类 150 帧以上（最多 {hi} 帧）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
