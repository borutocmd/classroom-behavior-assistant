# -*- coding: utf-8 -*-
"""基线：HOG 特征 + SVM。

先有基线再谈深度学习：基线出的混淆矩阵就能满足大纲“最低验收”的要求，
同时给主模型一个可比较的参照。

用法：
    python src/baseline_hog_svm.py
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, DATA_ROOT, OUTPUT_DIR, SEED  # noqa: E402
from src.dataset import load_rows  # noqa: E402
from src.evaluate import evaluate_predictions  # noqa: E402

HOG_WIN = (64, 128)   # cv2.HOGDescriptor 默认窗口


def hog_features(rel_path: str) -> np.ndarray:
    img = cv2.imread(str(DATA_ROOT / rel_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(rel_path)
    img = cv2.resize(img, HOG_WIN)
    hog = cv2.HOGDescriptor()
    return hog.compute(img).ravel()


def build_xy(rows):
    x = np.stack([hog_features(r["file"]) for r in rows])
    y = np.array([CLASSES.index(r["label"]) for r in rows])
    return x, y


def main():
    import argparse
    parser = argparse.ArgumentParser(description="基线：HOG 特征 + SVM，输出混淆矩阵与指标")
    parser.add_argument("--c", type=float, default=10.0, help="SVM 正则参数 C（默认 10）")
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "linear"], help="核函数")
    parser.add_argument("--max-train", type=int, default=6000, help="训练集最多用多少张（HOG+SVM 在大数据上很慢）")
    parser.add_argument("--max-test", type=int, default=3000, help="测试集最多用多少张")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        if not load_rows(split):
            print(f"缺少 {split} 数据，先运行 src/prepare_frames.py 与 src/make_splits.py")
            return 1

    def limited(split, cap):
        rows = load_rows(split)
        if cap and len(rows) > cap:
            rng = random.Random(args.seed)
            rows = rng.sample(rows, cap)
            print(f"  {split}: 从 {len(load_rows(split))} 张里随机抽 {cap} 张（HOG+SVM 的规模上限）")
        return rows

    print("提取 HOG 特征（64x128 窗口，3780 维）…")
    x_train, y_train = build_xy(limited("train", args.max_train))
    x_test, y_test = build_xy(limited("test", args.max_test))
    print(f"训练样本 {len(y_train)}，测试样本 {len(y_test)}，特征维数 {x_train.shape[1]}")

    scaler = StandardScaler().fit(x_train)
    clf = SVC(C=args.c, kernel=args.kernel, gamma="scale", class_weight="balanced",
              random_state=SEED)
    clf.fit(scaler.transform(x_train), y_train)

    y_pred = clf.predict(scaler.transform(x_test))
    metrics = evaluate_predictions(y_test, y_pred, "baseline_hog_svm", OUTPUT_DIR)
    print(f"图表与指标已写入 {OUTPUT_DIR}")
    return 0 if metrics["macro_f1"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
