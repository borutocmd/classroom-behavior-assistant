# -*- coding: utf-8 -*-
"""汇总各模型/基线的指标，生成论文用的对比表与柱状图。

用法：
    python src/compare_models.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR  # noqa: E402
from src.evaluate import setup_matplotlib  # noqa: E402


def main():
    metric_files = sorted(Path(OUTPUT_DIR).glob("指标_*.json"))
    if not metric_files:
        print("还没有指标文件，先跑 src/baseline_hog_svm.py 或 src/train_cnn.py")
        return 1
    records = [json.loads(p.read_text(encoding="utf-8")) for p in metric_files]
    records.sort(key=lambda r: r["macro_f1"], reverse=True)

    sample_counts = {r["samples"] for r in records}
    if len(sample_counts) > 1:
        print("⚠️ 警告：以下指标来自不同大小的测试集，不能直接比较：")
        for r in records:
            print(f"    {r['tag']}: 测试样本 {r['samples']}")
        print("    换数据集后请先清理旧指标（bash run_all.sh 会提示；或删除 outputs/指标_*.json 后重跑）\n")

    names = [r["tag"] for r in records]
    acc = [r["accuracy"] for r in records]
    f1 = [r["macro_f1"] for r in records]

    lines = ["# 模型对比（测试集）", "",
             f"测试样本数：{records[0]['samples']}", "",
             "| 模型 / 基线 | 准确率 | 宏 F1 | 加权 F1 |", "| --- | --- | --- | --- |"]
    for r in records:
        lines.append(f"| {r['tag']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} |")
    lines.append("")
    best = records[0]
    lines.append(f"最优：**{best['tag']}**（宏 F1 {best['macro_f1']:.4f}）。")
    lines.append("")
    lines.append("## 逐类表现（最优模型）")
    lines.append("")
    lines.append("| 类别 | 精确率 | 召回率 | F1 | 样本数 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, val in best["per_class"].items():
        lines.append(f"| {name} | {val['precision']:.3f} | {val['recall']:.3f} | "
                     f"{val['f1-score']:.3f} | {int(val['support'])} |")
    (Path(OUTPUT_DIR) / "模型对比.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plt = setup_matplotlib()
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars1 = ax.bar(x - width / 2, acc, width, label="准确率")
    bars2 = ax.bar(x + width / 2, f1, width, label="宏 F1")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("指标")
    ax.set_title("课堂行为识别：基线与模型对比（测试集）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(OUTPUT_DIR) / "模型对比.png", dpi=160)
    plt.close(fig)

    print("\n".join(lines))
    print(f"\n对比表与柱状图已写入 {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
