# -*- coding: utf-8 -*-
"""数据集体检：分布、划分、泄漏、重复、分辨率。

这些数字是后续训练与评估的依据。

用法：
    python src/check_dataset.py
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, CLASS_NAMES_CN, DATA_ROOT, LABELS_CSV, OUTPUT_DIR, SPLITS_JSON  # noqa: E402
from src.evaluate import setup_matplotlib  # noqa: E402


def dhash(path, size=8):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size + 1, size))
    return "".join("1" if small[i, j] < small[i, j + 1] else "0"
                   for i in range(size) for j in range(size))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="数据集体检：分布、划分、泄漏、重复帧、分辨率")
    parser.parse_args()

    if not LABELS_CSV.exists():
        print(f"缺少 {LABELS_CSV}，先运行 src/prepare_frames.py")
        return 1
    rows = list(csv.DictReader(LABELS_CSV.open(encoding="utf-8")))
    splits = json.loads(SPLITS_JSON.read_text(encoding="utf-8")) if SPLITS_JSON.exists() else {}

    counts = Counter((r["label"], r["split"]) for r in rows)
    sources = defaultdict(set)
    for r in rows:
        sources[r["label"]].add(r["source"])

    print(f"{'类别':<14}{'train':>7}{'val':>7}{'test':>7}{'合计':>7}{'来源数':>7}")
    print("-" * 52)
    for label in CLASSES:
        tr, va, te = counts[(label, "train")], counts[(label, "val")], counts[(label, "test")]
        print(f"{CLASS_NAMES_CN[label]:<14}{tr:>7}{va:>7}{te:>7}{tr + va + te:>7}{len(sources[label]):>7}")
    total = {k: sum(v for (lab, s), v in counts.items() if s == k) for k in ("train", "val", "test")}
    print("-" * 52)
    print(f"{'合计':<14}{total['train']:>7}{total['val']:>7}{total['test']:>7}"
          f"{sum(total.values()):>7}{sum(len(v) for v in sources.values()):>7}")

    # 泄漏检查
    seen = defaultdict(set)
    for r in rows:
        seen[r["source"]].add(r["split"])
    leaked = {s: sorted(v) for s, v in seen.items() if len(v) > 1}
    print(f"\n来源泄漏：{'❌ ' + str(len(leaked)) + ' 个来源跨集合' if leaked else '无泄漏 ✅'}")

    # 重复帧检查
    hashes = defaultdict(list)
    for r in rows:
        h = dhash(DATA_ROOT / r["file"])
        if h:
            hashes[h].append(r["file"])
    dup_groups = [v for v in hashes.values() if len(v) > 1]
    dup_frames = sum(len(v) - 1 for v in dup_groups)
    print(f"近似重复帧：{dup_frames} 张（{len(dup_groups)} 组）"
          f"{'，建议提高抽帧去重阈值' if dup_frames > len(rows) * 0.1 else ''}")

    # 分辨率抽查
    sample = rows[:: max(1, len(rows) // 50)][:50]
    shapes = Counter()
    for r in sample:
        img = cv2.imread(str(DATA_ROOT / r["file"]))
        if img is not None:
            shapes[f"{img.shape[1]}x{img.shape[0]}"] += 1
    print(f"分辨率抽查（{len(sample)} 张）：{dict(shapes)}")

    # 最少类别提示
    per_class_total = {lab: sum(v for (name, split), v in counts.items() if name == lab)
                       for lab in CLASSES}
    weakest = min(per_class_total.values())
    print(f"\n最少类别样本数 {weakest}，来源总数 {sum(len(v) for v in sources.values())}")
    if weakest < 150:
        print("提示：每类建议 150 帧以上，否则模型容易过拟合，论文里也要如实说明数据规模限制。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "frames_total": len(rows),
        "splits": total,
        "per_class": {CLASS_NAMES_CN[lab]: {
            "train": counts[(lab, "train")], "val": counts[(lab, "val")],
            "test": counts[(lab, "test")], "sources": len(sources[lab])} for lab in CLASSES},
        "leaked_sources": leaked,
        "duplicate_frames": dup_frames,
        "resolutions": dict(shapes),
        "seed": splits.get("seed"),
    }
    (OUTPUT_DIR / "数据集概况.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plt = setup_matplotlib()
    labels = [CLASS_NAMES_CN[c] for c in CLASSES]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for offset, split in enumerate(["train", "val", "test"]):
        ax.bar(x + (offset - 1) * width, [counts[(c, split)] for c in CLASSES], width, label=split)
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("帧数")
    ax.set_title(f"数据集分布（共 {len(rows)} 帧，来源 {sum(len(v) for v in sources.values())} 个）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "数据分布.png", dpi=160)
    plt.close(fig)
    print(f"\n概况写入 {OUTPUT_DIR / '数据集概况.json'}，分布图 {OUTPUT_DIR / '数据分布.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
