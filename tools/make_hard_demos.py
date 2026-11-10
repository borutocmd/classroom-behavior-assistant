# -*- coding: utf-8 -*-
"""生成两段更长的“困难版”演示视频（行为来回切换，考验时序平滑）。

- demo_long_a.mp4：约 60 秒，12 段、每段 3–6 秒，行为反复切换（快节奏课堂）
- demo_long_b.mp4：约 90 秒，9 段、含 10–15 秒的长段落（接近真实课堂节奏）

两段都用真实抽帧图片拼成，放在 <数据目录>/demo_videos/，**不参与训练**。

用法：
    python tools/make_hard_demos.py
"""
import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES_CN, CLASSES, DEMO_DIR, FRAME_DIR, ensure_dirs  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SCENARIOS = {
    "demo_long_a.mp4": [
        ("listening", 5), ("reading", 6), ("writing", 4), ("raising_hand", 3),
        ("desk", 4), ("listening", 5), ("reading", 6), ("writing", 5),
        ("raising_hand", 3), ("desk", 4), ("reading", 6), ("writing", 6),
    ],
    "demo_long_b.mp4": [
        ("listening", 15), ("reading", 12), ("writing", 10), ("raising_hand", 4),
        ("desk", 8), ("listening", 12), ("reading", 14), ("writing", 8), ("desk", 7),
    ],
}


def load_images(root: Path, label: str):
    folder = root / label
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXT)


def main():
    ap = argparse.ArgumentParser(description="生成困难版长演示视频")
    ap.add_argument("--src", default=str(FRAME_DIR), help="图片根目录（按类别分文件夹）")
    ap.add_argument("--out", default=str(DEMO_DIR))
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    ensure_dirs()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    pool = {label: load_images(src, label) for label in CLASSES}
    missing = [CLASS_NAMES_CN[c] for c, v in pool.items() if not v]
    if missing:
        print("以下类别没有图片，跳过：" + "、".join(missing))
    print(f"可用图片：{ {CLASS_NAMES_CN[k]: len(v) for k, v in pool.items()} }")

    for name, segments in SCENARIOS.items():
        path = out / name
        writer = None
        total, used = 0, {}
        plan = []
        for label, seconds in segments:
            images = pool.get(label) or []
            if not images:
                continue
            frames = max(1, int(seconds * args.fps))
            for i in range(frames):
                img = cv2.imread(str(images[(total + i) % len(images)]))
                if img is None:
                    continue
                img = cv2.resize(img, (args.size, args.size))
                img = cv2.convertScaleAbs(img, alpha=1.0, beta=rng.randint(-12, 12))
                if writer is None:
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.fps, (args.size, args.size))
                writer.write(img)
                total += 1
            used[label] = used.get(label, 0) + frames
            plan.append(f"{CLASS_NAMES_CN[label]}{seconds}s")
        if writer is not None:
            writer.release()
        print(f"\n{name}：{total} 帧 / {total / args.fps:.0f} 秒，{len(plan)} 段")
        print("  行为序列：" + " → ".join(plan))

    print(f"\n输出目录：{out}（不参与训练，网页下拉框里直接可选）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
