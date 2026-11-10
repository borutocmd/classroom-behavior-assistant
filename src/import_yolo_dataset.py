# -*- coding: utf-8 -*-
"""把 YOLO 格式的目标检测数据集导入成本项目的分类数据集。

适用：公开课堂行为数据集（如 SCB-Dataset 及其转载版本），结构为
    <源目录>/images/{train,val,test}/*.jpg
    <源目录>/labels/{train,val,test}/*.txt
    <源目录>/data.yaml         （可选，含 names 类别名）
每张图里有多名学生，每个学生一个框 —— 本脚本把框裁出来，按类别存成单人图，
于是检测数据集就变成了分类数据集，可以直接接 prepare_frames.py 与 make_splits.py。

用法：
    python src/import_yolo_dataset.py --src <解压后的数据集目录> --out <数据目录>/public_raw
    # 然后照常：python src/prepare_frames.py --src <数据目录>/public_raw ...
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES_CN, LABEL_ALIASES, ensure_dirs, norm_label  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 数据集里常见的额外写法；基础别名表统一来自 config，避免两处不一致导致整类被跳过
EXTRA_ALIASES = {
    **LABEL_ALIASES,
    "using phone": "phone", "use_phone": "phone", "playphone": "phone",
    "bow_head": "desk", "sleep": "desk", "lying": "desk",
    "raisehand": "raising_hand", "listening": "listening",
}


def map_label(name: str):
    key = str(name).strip().lower()
    if key in EXTRA_ALIASES:
        return EXTRA_ALIASES[key]
    return norm_label(name)


def read_names(src: Path):
    """从 data.yaml / classes.txt / names.txt 里读类别名。"""
    yaml_path = next(iter(src.glob("*.yaml")), None) or next(iter(src.glob("*.yml")), None)
    if yaml_path:
        text = yaml_path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.strip().startswith("names"):
                raw = line.split(":", 1)[1] if ":" in line else line
                items = [x.strip().strip("'\"[] ") for x in raw.split(",")]
                if any(items):
                    return [i for i in items if i]
        try:
            import yaml
            data = yaml.safe_load(text)
            if isinstance(data, dict) and isinstance(data.get("names"), (list, dict)):
                names = data["names"]
                return list(names.values()) if isinstance(names, dict) else names
        except Exception:
            pass
    for candidate in ["classes.txt", "names.txt", "predefined_classes.txt"]:
        path = src / candidate
        if path.exists():
            return [line.strip() for line in path.read_text(encoding="utf-8",
                                                            errors="ignore").splitlines()
                    if line.strip()]
    return []


def find_pairs(src: Path):
    """返回 (图片路径, 标签路径) 列表，兼容有/无 train|val|test 子目录两种结构。"""
    pairs = []
    for img in src.rglob("*"):
        if img.suffix.lower() not in IMAGE_EXT:
            continue
        parts = list(img.parts)
        if "images" in parts:
            idx = parts.index("images")
            label_path = Path(*parts[:idx], "labels", *parts[idx + 1:]).with_suffix(".txt")
        else:
            label_path = img.with_suffix(".txt")
        if label_path.exists():
            pairs.append((img, label_path))
    # 同一张图可能被重复匹配，去重
    seen, result = set(), []
    for pair in pairs:
        if pair[0] not in seen:
            seen.add(pair[0])
            result.append(pair)
    return result


def main():
    ap = argparse.ArgumentParser(description="YOLO 检测数据集 → 分类数据集")
    ap.add_argument("--src", required=True, help="解压后的数据集目录")
    ap.add_argument("--out", required=True, help="输出目录（建议 <数据目录>/public_raw）")
    ap.add_argument("--pad", type=float, default=0.06, help="框外扩比例，默认 0.06（即上下左右各外扩约 6 个百分点）")
    ap.add_argument("--max-per-class", type=int, default=0, help="每类最多导入多少张，0 表示不限")
    ap.add_argument("--min-size", type=int, default=32, help="裁出的小图最小边长，太小丢弃")
    ap.add_argument("--source-prefix", type=int, default=0,
                    help="用文件名前 N 个字符作为“来源”分组（SCB-Dataset 这类连续帧数据集建议设 4，"
                         "避免同一段视频的相邻帧被拆到训练集和测试集）")
    ap.add_argument("--class-names", default="",
                    help="数据集缺 yaml/classes 文件时手动指定类别名，逗号分隔，例如 bow_head,turn_head")
    args = ap.parse_args()

    src, out_root = Path(args.src), Path(args.out)
    if not src.exists():
        print(f"源目录不存在：{src}")
        return 1
    ensure_dirs()
    out_root.mkdir(parents=True, exist_ok=True)

    names = [n.strip() for n in args.class_names.split(",") if n.strip()] or read_names(src)
    pairs = find_pairs(src)
    print(f"找到 {len(pairs)} 张带标注的图片" + (f"，类别名 {names}" if names else "，未找到类别名文件"))
    if not pairs:
        print("没有配对成功。检查目录结构是否为 images/ 与 labels/ 并列。")
        return 1

    counts, skipped, unmapped = Counter(), 0, Counter()
    for img_path, label_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        for i, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(x) for x in parts[1:5])
            name = names[cls_id] if names and cls_id < len(names) else str(cls_id)
            label = map_label(name)
            if label is None:
                unmapped[name] += 1
                continue
            if args.max_per_class and counts[label] >= args.max_per_class:
                continue
            pad_w, pad_h = bw * (1 + args.pad), bh * (1 + args.pad)
            x1 = int(max(0, (cx - pad_w / 2) * w))
            y1 = int(max(0, (cy - pad_h / 2) * h))
            x2 = int(min(w, (cx + pad_w / 2) * w))
            y2 = int(min(h, (cy + pad_h / 2) * h))
            if x2 - x1 < args.min_size or y2 - y1 < args.min_size:
                skipped += 1
                continue
            label_dir = out_root / label
            label_dir.mkdir(parents=True, exist_ok=True)
            # 文件名决定“来源”分组：prepare_frames 会把结尾的 _数字 去掉当作来源。
            # 连续的课堂视频帧必须按“场景前缀”归组，否则同一段视频的相邻帧会被拆到训练/测试两边。
            group = (f"scene{img_path.stem[:args.source_prefix]}"
                     if args.source_prefix > 0 else img_path.stem)
            serial = counts[label]
            out_path = label_dir / f"{group}_{serial:06d}.jpg"
            cv2.imwrite(str(out_path), img[y1:y2, x1:x2], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            counts[label] += 1

    print(f"\n{'类别':<14}{'导入张数':>10}")
    for label, count in counts.most_common():
        print(f"{CLASS_NAMES_CN.get(label, label):<14}{count:>10}")
    print(f"{'合计':<14}{sum(counts.values()):>10}")
    if skipped:
        print(f"跳过（读不到图 / 框太小）：{skipped}")
    if unmapped:
        print(f"未能映射的原始类别名：{dict(unmapped)}  —— 需要时把它们补进 EXTRA_ALIASES")
    print(f"\n输出目录 {out_root}，接着运行：\n"
          f"  python src/prepare_frames.py --src {out_root} --fps 1 --size 256\n"
          f"  python src/make_splits.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
