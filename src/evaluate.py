# -*- coding: utf-8 -*-
"""评价指标与混淆矩阵绘图。混淆矩阵是大纲要求的最低验收项之一。"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, CLASS_NAMES_CN, OUTPUT_DIR, PROJECT_ROOT  # noqa: E402

FONT_CANDIDATES = [
    PROJECT_ROOT / "assets" / "fonts" / "simhei.ttf",
    Path("/mnt/c/Windows/Fonts/simhei.ttf"),
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
]


def setup_matplotlib():
    """配置 matplotlib：无界面后端 + 中文字体（找不到就用英文标签）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for path in FONT_CANDIDATES:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_confusion_matrix(cm, labels, out_path, title, normalize=False):
    plt = setup_matplotlib()
    data = cm.astype(float)
    if normalize:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() > 0 else 1)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = data[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if value > data.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def evaluate_predictions(y_true, y_pred, tag, out_dir=OUTPUT_DIR, class_names=None):
    """算指标 + 画两张混淆矩阵（计数 / 归一化）+ 存 json。返回指标字典。"""
    class_names = class_names or CLASSES
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    display = [CLASS_NAMES_CN.get(c, c) for c in class_names]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                   target_names=display, output_dict=True, zero_division=0)

    plot_confusion_matrix(cm, display, out_dir / f"混淆矩阵_{tag}.png",
                          f"{tag} 混淆矩阵（计数）")
    plot_confusion_matrix(cm, display, out_dir / f"混淆矩阵_{tag}_归一化.png",
                          f"{tag} 混淆矩阵（按真实类别归一化）", normalize=True)

    metrics = {
        "tag": tag,
        "samples": int(len(y_true)),
        "classes": class_names,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {k: {m: round(v, 4) for m, v in val.items() if isinstance(v, (int, float))}
                      for k, val in report.items() if k in display},
        "confusion_matrix": cm.tolist(),
    }
    (out_dir / f"指标_{tag}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{tag}] 样本 {len(y_true)} | 准确率 {acc:.4f} | 宏F1 {macro_f1:.4f} | 加权F1 {weighted_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                target_names=display, zero_division=0))
    return metrics
