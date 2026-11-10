# -*- coding: utf-8 -*-
"""公开数据集预检：先看清楚有没有我们要的类别，再决定要不要自己补录。

支持两种结构：
  1) YOLO 检测集：images/{train,val,test} + labels/{train,val,test} + data.yaml
  2) 分类集：若干类别子目录，每个子目录里是图片

用法：
    python src/inspect_dataset_source.py --src "E:/datasets/课堂行为数据集"
输出：每类数量、类别映射结果、缺少哪些类、是否需要补录（不写入任何数据）。
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES_CN, CLASSES  # noqa: E402
from src.import_yolo_dataset import IMAGE_EXT, find_pairs, map_label, read_names  # noqa: E402


def count_source_dirs(src: Path):
    """分类目录结构：返回 {目录名: 图片数}。"""
    counts = {}
    for sub in sorted(p for p in src.iterdir() if p.is_dir()):
        if sub.name.lower() in {"images", "labels", "train", "val", "test"}:
            continue
        n = sum(1 for f in sub.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        if n:
            counts[sub.name] = n
    return counts


def main():
    ap = argparse.ArgumentParser(description="公开数据集预检")
    ap.add_argument("--src", required=True, help="数据集目录")
    args = ap.parse_args()
    src = Path(args.src)
    if not src.exists():
        print(f"目录不存在：{src}")
        return 1

    print(f"检查目录：{src}")
    yaml_names = read_names(src)
    pairs = find_pairs(src)
    structure = "YOLO 检测集" if pairs else "分类目录"
    if not pairs and not yaml_names:
        dir_counts = count_source_dirs(src)
        if not dir_counts:
            print("既没有 images/labels 配对，也没有类别子目录，请确认解压是否完整。")
            return 1
        print(f"结构：{structure}，共 {sum(dir_counts.values())} 张图片，{len(dir_counts)} 个类别目录")
    else:
        print(f"结构：{structure}，{len(pairs)} 张带标注图片，data.yaml 类别名 {yaml_names or '未找到'}")

    # 统计原始类别 → 本项目类别
    raw_counts = Counter()
    if pairs:
        for _, label_path in pairs:
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                name = yaml_names[cls_id] if yaml_names and cls_id < len(yaml_names) else str(cls_id)
                raw_counts[name] += 1
    else:
        raw_counts.update(count_source_dirs(src))

    mapped = Counter()
    unmapped = Counter()
    print(f"\n{'原始类别':<16}{'数量':>8}   映射到")
    print("-" * 46)
    for name, count in raw_counts.most_common():
        target = map_label(name)
        if target:
            mapped[target] += count
            print(f"{str(name):<16}{count:>8}   {CLASS_NAMES_CN[target]}（{target}）")
        else:
            unmapped[name] += count
            print(f"{str(name):<16}{count:>8}   ❌ 无法映射，需要补进 EXTRA_ALIASES")

    covered = [c for c in CLASSES if mapped.get(c)]
    missing = [c for c in CLASSES if not mapped.get(c)]
    print(f"\n覆盖到的类别（{len(covered)}/{len(CLASSES)}）："
          + "、".join(f"{CLASS_NAMES_CN[c]}({mapped[c]})" for c in covered))
    if missing:
        print("缺少的类别：" + "、".join(f"{CLASS_NAMES_CN[c]}" for c in missing))
    if unmapped:
        print(f"无法映射的类别：{dict(unmapped)}")

    print("\n结论：")
    if "listening" not in missing:
        print("  ✅ 数据集里已经有“听讲”类，**不需要自己录视频**，直接导入即可。")
    else:
        print("  ⚠️ 数据集里没有“听讲”类 → 两个选择：")
        print("     A. 自己录 3–5 分钟听讲视频（最简单，还能拿到授权书，满足大纲“采集授权”要求）：")
        print("        放到 <数据目录>/raw/listening/ 下，再跑 prepare_frames.py")
        print("     B. 不录：把“听讲”从类别里去掉，改用数据集自带的 6 类"
              "（举手/阅读/写作/使用手机/低头/睡觉），抬头率改由“非低头类占比”近似计算")
    print(f"\n下一步命令：\n  python src/import_yolo_dataset.py --src \"{src}\" --out <数据目录>/public_raw")
    return 0


if __name__ == "__main__":
    sys.exit(main())
