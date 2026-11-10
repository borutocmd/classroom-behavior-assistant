# -*- coding: utf-8 -*-
"""合成一段“行为会变化”的课堂视频，用来验证时序平滑与时间线。

一段视频里依次出现：听讲 → 举手 → 低头写字 → 使用手机 → 趴桌，
并且逐帧加入随机扰动，让逐帧预测出现抖动，这样才能看出平滑的作用。

⚠️ 仅用于流程自检，论文实验必须用真实采集数据。

用法：
    python tools/make_demo_video.py --out /tmp/mixed.mp4 --seconds-per-class 6 --fps 10
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.make_demo_data import render  # noqa: E402

SEQUENCE = ["listening", "raising_hand", "writing", "phone", "desk"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/mixed_behavior.mp4")
    ap.add_argument("--seconds-per-class", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    frames_per_class = int(args.seconds_per_class * args.fps)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (320, 320))
    total = 0
    for label in SEQUENCE:
        # 每个类别段内每 1 秒换一次随机姿态，段边界附近制造少量“错帧”模拟抖动
        for i in range(frames_per_class):
            current = label
            if rng.random() < 0.12:                      # 12% 的帧随机串到相邻类别
                idx = SEQUENCE.index(label)
                current = SEQUENCE[min(max(idx + rng.choice([-1, 1]), 0), len(SEQUENCE) - 1)]
            frame = np.asarray(render(current, rng))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            total += 1
    writer.release()
    print(f"生成 {total} 帧（{total / args.fps:.1f} 秒）→ {args.out}")
    print(f"行为顺序：{' → '.join(SEQUENCE)}，每段 {args.seconds_per_class} 秒，含约 12% 的随机抖动帧")


if __name__ == "__main__":
    main()
