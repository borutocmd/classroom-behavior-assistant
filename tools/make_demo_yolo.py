# -*- coding: utf-8 -*-
"""生成一份 YOLO 格式的小数据集，用来验证 import_yolo_dataset.py 的导入链路。

⚠️ 只是测试用的小样本，不是真实数据。

用法：
    python tools/make_demo_yolo.py --out /tmp/demo_yolo
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.make_demo_data import render  # noqa: E402

# 模拟公开数据集的中文类别名（含“低头/睡觉”这类与我们的类目不同的名字）
PUBLIC_NAMES = ["举手", "阅读", "写作", "使用手机", "低头", "睡觉"]
INTERNAL = ["raising_hand", "reading", "writing", "phone", "desk", "desk"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/demo_yolo")
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.out)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/train\nnc: 6\n"
        f"names: {PUBLIC_NAMES}\n".replace("'", '"'), encoding="utf-8")

    rng = random.Random(args.seed)
    written = 0
    for cls_id, internal in enumerate(INTERNAL):
        for i in range(args.per_class):
            # 一张“教室”图里放两个学生，模拟多目标场景
            canvas = np.full((320, 320, 3), 230, dtype=np.uint8)
            boxes = []
            for k in range(2):
                person = np.asarray(render(internal, rng))
                person = cv2.cvtColor(person, cv2.COLOR_RGB2BGR)
                person = cv2.resize(person, (110, 150))
                x, y = 20 + k * 150 + rng.randint(-8, 8), 140 + rng.randint(-10, 10)
                canvas[y:y + 150, x:x + 110] = person
                cx = (x + 55) / 320
                cy = (y + 75) / 320
                boxes.append((cls_id, cx, cy, 110 / 320, 150 / 320))
            stem = f"scene_{cls_id}_{internal}_{i:02d}"
            cv2.imwrite(str(root / "images" / "train" / f"{stem}.jpg"), canvas)
            (root / "labels" / "train" / f"{stem}.txt").write_text(
                "\n".join(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes),
                encoding="utf-8")
            written += 1
    print(f"生成 {written} 张 YOLO 格式图片（每张 2 个框）→ {root}")
    print(f"类别名（模拟公开数据集）：{PUBLIC_NAMES}")


if __name__ == "__main__":
    main()
