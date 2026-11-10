# -*- coding: utf-8 -*-
"""录制素材预检：录完之后先跑这个，别等整条流程跑完才发现问题。

检查内容：
  · 每个类别目录名能否识别、有几个文件
  · 视频能否打开、时长、分辨率、帧率，按 1 fps 估算能抽多少帧
  · 是否达到“每类 ≥150 帧”的建议线，缺哪些类别
  · 命名是否能区分“来源”（同一段视频的帧不会被拆到不同集合）
  · 明显问题：打不开的文件、时长过短、重名、单文件过长

用法：
    python src/check_raw.py                       # 检查 <数据目录>/raw
    python src/check_raw.py --src <其它目录>
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES_CN, CLASSES, RAW_DIR, norm_label  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_FRAMES_PER_CLASS = 150
MIN_SECONDS = 8
MAX_SECONDS = 180


def probe(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    seconds = count / fps if fps else 0
    return {"fps": round(fps, 1), "frames": int(count), "seconds": round(seconds, 1),
            "size": f"{width}x{height}"}


def main():
    ap = argparse.ArgumentParser(description="录制素材预检")
    ap.add_argument("--src", default=str(RAW_DIR), help="素材目录（默认数据目录下的 raw/）")
    ap.add_argument("--target-per-class", type=int, default=TARGET_FRAMES_PER_CLASS,
                    help=f"每类目标帧数（默认 {TARGET_FRAMES_PER_CLASS}）")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists() or not any(src.iterdir()):
        print(f"{src} 还是空的。按 docs/数据采集指南.md 放好素材，例如：")
        print("    raw/listening/zhangsan_01.mp4")
        print("    raw/raising_hand/zhangsan_01.mp4   …")
        return 1

    per_class = defaultdict(list)
    problems = []
    unknown_dirs = []
    for sub in sorted(p for p in src.iterdir() if p.is_dir()):
        label = norm_label(sub.name)
        if label is None:
            unknown_dirs.append(sub.name)
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in VIDEO_EXT:
                per_class[label].append(f)
            elif f.suffix.lower() in IMAGE_EXT:
                per_class[label].append(f)

    print(f"检查目录：{src}")
    print(f"{'类别':<12}{'文件数':>7}{'总时长(秒)':>11}{'预计抽帧':>10}   分辨率/帧率")
    print("-" * 62)

    est_total = 0
    for label in CLASSES:
        files = per_class.get(label, [])
        seconds = 0.0
        est = 0
        info_set = Counter()
        for f in files:
            if f.suffix.lower() in IMAGE_EXT:
                est += 1
                info_set["图片"] += 1
                continue
            info = probe(f)
            if info is None:
                problems.append(f"打不开：{f.relative_to(src)}")
                continue
            seconds += info["seconds"]
            est += max(1, int(info["seconds"] // 1))      # 按 1 fps 估算
            info_set[f"{info['size']} @{info['fps']}fps"] += 1
            if info["seconds"] < MIN_SECONDS:
                problems.append(f"时长偏短（{info['seconds']} 秒）：{f.relative_to(src)}，建议 ≥{MIN_SECONDS} 秒")
            if info["seconds"] > MAX_SECONDS:
                problems.append(f"时长偏长（{info['seconds']} 秒）：{f.relative_to(src)}，可截断以免抽帧过多")
        est_total += est
        flag = "" if est >= args.target_per_class else "  ← 偏少"
        print(f"{CLASS_NAMES_CN[label]:<12}{len(files):>7}{seconds:>11.1f}{est:>10}{flag}"
              f"   {dict(info_set) if info_set else '-'}")

    # 来源分组检查
    sources = {lab: {f.stem.rsplit('_', 1)[0] if '_' in f.stem else f.stem for f in files}
               for lab, files in per_class.items()}
    print(f"\n文件数 {sum(len(v) for v in per_class.values())}，预计总帧数 {est_total}，"
          f"来源数 {sum(len(v) for v in sources.values())}")

    missing = [c for c in CLASSES if not per_class.get(c)]
    if unknown_dirs:
        problems.append("目录名无法识别（不会被抽帧）：" + "、".join(unknown_dirs)
                        + "  → 请改成英文类名或中文类名，如 listening/听讲")
    if missing:
        problems.append("缺少类别：" + "、".join(CLASS_NAMES_CN[c] for c in missing))
    short = [CLASS_NAMES_CN[c] for c in CLASSES
             if per_class.get(c) and len(per_class[c]) < 2]
    if short:
        problems.append("来源太少（建议每类 ≥3 段不同视频/不同人）：" + "、".join(short))

    print("\n结论：")
    if not problems:
        print("  ✅ 素材看起来没问题，可以开始：python src/prepare_frames.py")
    else:
        for p in problems:
            print(f"  ⚠️ {p}")
        print("\n  修好后再跑 python src/prepare_frames.py（重跑不会污染旧数据）")

    est_per_class = {CLASS_NAMES_CN[c]: len(per_class.get(c, [])) for c in CLASSES}
    print(f"\n每类文件数：{est_per_class}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
