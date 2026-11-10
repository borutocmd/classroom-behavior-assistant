# -*- coding: utf-8 -*-
"""按“来源”分层划分训练/验证/测试集。

关键点：**同一段视频/同一张原图的帧只能进同一个集合**，否则相邻帧同时出现在
训练和测试里会造成数据泄漏、指标虚高。

注意（SCB-Dataset 这类数据的坑）：一张课堂照片里可能同时标了多种行为，
所以同一个“来源”会出现在多个类别里。因此分配必须**全局按来源做一次**，
不能按类别各分一次，否则同一来源会在 A 类进训练、在 B 类进测试。

用法：
    python src/make_splits.py --seed 42
输出：
    labels.csv 增加 split 列；splits.json 记录种子、比例、各集合帧数与来源清单。
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, LABELS_CSV, SPLITS_JSON, TEST_RATIO, VAL_RATIO, ensure_dirs  # noqa: E402


def assign_splits(rows, seed, val_ratio, test_ratio):
    """全局按来源分配，返回 ({source: split}, {source: 信息})。"""
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    info = {}
    for source, items in by_source.items():
        per_label = defaultdict(int)
        for r in items:
            per_label[r["label"]] += 1
        info[source] = {"frames": len(items),
                        "dominant": max(per_label, key=per_label.get)}

    rng = random.Random(seed)
    assignment = {}
    for label in CLASSES:
        sources = [s for s, meta in info.items() if meta["dominant"] == label]
        rng.shuffle(sources)
        total = sum(info[s]["frames"] for s in sources)
        target_test = round(total * test_ratio)
        target_val = round(total * val_ratio)
        filled = {"train": 0, "val": 0, "test": 0}
        for source in sources:
            frames = info[source]["frames"]
            if filled["test"] < target_test:
                bucket = "test"
            elif filled["val"] < target_val:
                bucket = "val"
            else:
                bucket = "train"
            assignment[source] = bucket
            filled[bucket] += frames
        # 某个类别来源很少时，保证 val/test 至少各有一个来源
        if sources:
            for bucket in ("val", "test"):
                if not any(assignment[s] == bucket for s in sources):
                    smallest = min(sources, key=lambda s: info[s]["frames"])
                    assignment[smallest] = bucket
    return assignment, info


def main():
    parser = argparse.ArgumentParser(description="分层划分数据集（全局按来源分组，防止泄漏）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val", type=float, default=VAL_RATIO)
    parser.add_argument("--test", type=float, default=TEST_RATIO)
    args = parser.parse_args()

    ensure_dirs()
    if not LABELS_CSV.exists():
        print(f"找不到 {LABELS_CSV}，先运行 src/prepare_frames.py")
        return 1

    with LABELS_CSV.open(encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        print("labels.csv 是空的，先抽帧")
        return 1

    assignment, info = assign_splits(rows, args.seed, args.val, args.test)
    for r in rows:
        r["split"] = assignment[r["source"]]

    with LABELS_CSV.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["file", "label", "source", "frame_idx", "split"])
        writer.writeheader()
        writer.writerows(rows)

    stats = defaultdict(lambda: defaultdict(int))
    for r in rows:
        stats[r["label"]][r["split"]] += 1

    leaks = defaultdict(set)
    for r in rows:
        leaks[r["source"]].add(r["split"])
    leaked = {s: sorted(v) for s, v in leaks.items() if len(v) > 1}

    summary = {
        "seed": args.seed,
        "val_ratio": args.val,
        "test_ratio": args.test,
        "unit": "source（全局分配，同一来源只进一个集合）",
        "counts": {label: dict(v) for label, v in stats.items()},
        "sources": {label: sorted({r["source"] for r in rows if r["label"] == label})
                    for label in CLASSES},
        "assignment": assignment,
        "leaked_sources": leaked,
    }
    SPLITS_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'类别':<14}{'train':>7}{'val':>7}{'test':>7}")
    for label in CLASSES:
        v = stats.get(label, {})
        print(f"{label:<14}{v.get('train', 0):>7}{v.get('val', 0):>7}{v.get('test', 0):>7}")
    total = {k: sum(v.get(k, 0) for v in stats.values()) for k in ("train", "val", "test")}
    print(f"{'合计':<14}{total['train']:>7}{total['val']:>7}{total['test']:>7}")
    print(f"\n来源总数 {len(info)}")
    print(f"来源泄漏检查：{'❌ ' + str(len(leaked)) + ' 个来源跨集合' if leaked else '无泄漏 ✅'}")
    print(f"划分已写入 {SPLITS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
