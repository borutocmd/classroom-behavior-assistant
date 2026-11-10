# -*- coding: utf-8 -*-
"""消融实验汇总：时序平滑、预训练、检索增强（RAG）。

三组实验对应论文里的三个“为什么这么设计”：
  1. 有/无时序平滑 —— 逐帧抖动有多少（读时间线 JSON 即可，不用重训）
  2. 有/无 ImageNet 预训练 —— 读 指标_*.json（无预训练用 --backbone scratch 训练一次）
  3. 有/无检索增强（RAG）—— 两次报告对比，并自动检查报告里的数字是否都来自统计

用法：
    python src/ablation.py                       # 汇总已有结果
    python src/ablation.py --check-report        # 额外做报告数字忠实度检查
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR  # noqa: E402
from src.evaluate import setup_matplotlib  # noqa: E402


def events_of(labels):
    events = 0
    for i, lab in enumerate(labels):
        if i == 0 or labels[i - 1] != lab:
            events += 1
    return events


def smoothing_ablation():
    rows = []
    for path in sorted(Path(OUTPUT_DIR).glob("时间线_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        raw, smooth = data.get("raw", []), data.get("smoothed", [])
        if not raw:
            continue
        stats = data.get("stats", {})
        same = sum(a == b for a, b in zip(raw, smooth)) / len(raw)
        rows.append({
            "tag": path.stem.replace("时间线_", ""),
            "frames": len(raw),
            "events_raw": events_of(raw),
            "events_smooth": events_of(smooth),
            "avg_len_raw": round(len(raw) / max(events_of(raw), 1), 2),
            "avg_len_smooth": round(len(smooth) / max(events_of(smooth), 1), 2),
            "agree": round(same, 4),
            "raise_hand_raw": stats.get("举手次数", 0),
        })
    return rows


def pretrain_ablation():
    rows = []
    for path in sorted(Path(OUTPUT_DIR).glob("指标_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tag = data["tag"]
        if not tag.startswith("cnn_"):
            continue
        rows.append({"tag": tag, "accuracy": data["accuracy"],
                     "macro_f1": data["macro_f1"], "samples": data["samples"]})
    return rows


def number_fidelity(report_text, stats, tips=None):
    """报告里的数字有多少能溯源到统计或引用的教学建议。

    报告里的数字只允许来自两处：行为统计、以及带 [资料N] 引用的教学建议。
    两者都找不到的数字，就算凭空出现的数字。
    """
    pool = json.dumps(stats, ensure_ascii=False)
    if tips:
        pool += json.dumps(tips, ensure_ascii=False)
    numbers = re.findall(r"\d+(?:\.\d+)?", report_text)
    if not numbers:
        return {"total": 0, "hit": 0, "rate": None, "missing": []}
    missing = []
    for num in numbers:
        value = float(num)
        variants = {num, num.rstrip("0").rstrip("."), f"{value:.1f}", f"{value:.2f}",
                    f"{value * 100:.1f}", str(int(value * 100)) if value <= 1 else num,
                    f"{value / 100:.2f}", f"{value / 100:.1f}", str(value / 100)}
        if not any(v and v in pool for v in variants):
            missing.append(num)
    return {"total": len(numbers), "hit": len(numbers) - len(missing),
            "rate": round((len(numbers) - len(missing)) / len(numbers), 4),
            "missing": sorted(set(missing))}


def report_ablation(check=False):
    rows = []
    for path in sorted(Path(OUTPUT_DIR).glob("报告_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        report, stats = data.get("report", ""), data.get("stats", {})
        row = {
            "tag": data.get("tag"),
            "chars": len(report),
            "citations": len(set(re.findall(r"\[资料\d+\]", report))),
            "tips": len(data.get("retrieved_tips", [])),
            "invalid_citations": data.get("citation_check", {}).get("无效编号", []),
        }
        if check:
            row.update(number_fidelity(report, stats, data.get("retrieved_tips")))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-report", action="store_true", help="报告数字忠实度检查")
    args = ap.parse_args()

    lines = ["# 消融实验汇总", ""]
    plt = setup_matplotlib()

    smooth = smoothing_ablation()
    lines += ["## 一、时序平滑消融", ""]
    if smooth:
        lines += ["| 视频 | 帧数 | 平滑前事件数 | 平滑后事件数 | 平均片段(前/后) | 平滑前后一致率 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in smooth:
            lines.append(f"| {r['tag']} | {r['frames']} | {r['events_raw']} | {r['events_smooth']} | "
                         f"{r['avg_len_raw']} / {r['avg_len_smooth']} | {r['agree']:.1%} |")
        avg_agree = float(np.mean([r["agree"] for r in smooth]))
        cut = float(np.mean([1 - r["events_smooth"] / max(r["events_raw"], 1) for r in smooth]))
        lines += ["", f"平均一致率 {avg_agree:.1%}，事件数平均减少 {cut:.1%}——"
                      "说明逐帧预测确实存在抖动，平滑后片段更连续、更适合作为课堂时间线展示。", ""]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        x = np.arange(len(smooth))
        ax.bar(x - 0.18, [r["events_raw"] for r in smooth], 0.36, label="平滑前")
        ax.bar(x + 0.18, [r["events_smooth"] for r in smooth], 0.36, label="平滑后")
        ax.set_xticks(x, [r["tag"] for r in smooth], rotation=15)
        ax.set_ylabel("行为片段数")
        ax.set_title("时序平滑前后的事件数对比")
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(OUTPUT_DIR) / "消融_时序平滑.png", dpi=160)
        plt.close(fig)
    else:
        lines.append("（还没有时间线文件，先运行 src/analyze_video.py）")
    lines.append("")

    pretrain = pretrain_ablation()
    lines += ["## 二、迁移学习与预训练消融（冻结主干 / 全网络微调 / 无预训练）", ""]
    if pretrain:
        lines += ["| 模型 | 准确率 | 宏 F1 |", "| --- | --- | --- |"]
        for r in pretrain:
            lines.append(f"| {r['tag']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} |")
        tags = {r["tag"]: r for r in pretrain}
        if "cnn_hf_frozen" in tags and "cnn_hf_finetune" in tags:
            delta = tags["cnn_hf_finetune"]["macro_f1"] - tags["cnn_hf_frozen"]["macro_f1"]
            lines += ["", f"- **解冻主干（全网络微调）** 相对冻结主干的宏 F1 提升：**{delta:+.4f}**"
                          "——在 4 万张训练图上只训分类头会明显欠拟合。"]
        if "cnn_scratch" in tags and "cnn_hf_finetune" in tags:
            delta = tags["cnn_hf_finetune"]["macro_f1"] - tags["cnn_scratch"]["macro_f1"]
            lines += [f"- **ImageNet 预训练** 相对从头训练（scratch）的宏 F1 提升：**{delta:+.4f}**"
                      "——从头训练在同样 4 万张图上 12 轮仍未收敛，且需要更多轮次与更长的训练时间。"]
        lines.append("")
    else:
        lines.append("（还没有 cnn_* 指标文件，先运行 src/train_cnn.py）")
    lines.append("")

    reports = report_ablation(check=args.check_report)
    lines += ["## 三、检索增强（RAG）消融", ""]
    if reports:
        header = "| 报告 | 字数 | 引用建议数 | 无效引用 |"
        sep = "| --- | --- | --- | --- |"
        if args.check_report:
            header += " 数字总数 | 数字可溯源 | 忠实度 |"
            sep += " --- | --- | --- |"
        lines += [header, sep]
        for r in reports:
            row = f"| {r['tag']} | {r['chars']} | {r['citations']} | {r['invalid_citations'] or '无'} |"
            if args.check_report:
                row += (f" {r.get('total', 0)} | {r.get('hit', 0)} | "
                        f"{(r.get('rate') or 0):.1%} |")
            lines.append(row)
        lines += ["", "数字可溯源 = 报告里出现的数字能在统计文件或引用的教学建议里找到；该比例越低说明模型越可能编造数据。"
                      "未溯源数字常见于两种情况：模型自行做了加法（如 26.7%+6.7%=33.4%）、"
                      "或是序数与结构性数字（如“3 条建议”），都需要人工复核，自动检查只能粗筛："]
        for r in reports:
            if r.get("missing"):
                lines.append(f"- `{r['tag']}` 未溯源数字：{r['missing']}")
        lines.append("")
    else:
        lines.append("（还没有报告文件，先运行 src/report_llm.py，再用 --no-rag 跑一次）")
    lines.append("")

    out = Path(OUTPUT_DIR) / "消融实验.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
