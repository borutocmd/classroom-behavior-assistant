# -*- coding: utf-8 -*-
"""把 outputs/ 里的所有结果汇总成一页，交给写论文的组员直接用。

用法：
    python src/summarize_results.py
输出：
    outputs/实验结果汇总.md
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES_CN, CLASSES, DATA_ROOT, OUTPUT_DIR  # noqa: E402
from src.ablation import pretrain_ablation, report_ablation, smoothing_ablation  # noqa: E402

FIGURE_USAGE = {
    "数据分布.png": "第 4 章 数据与划分 —— 说明类别是否均衡",
    "混淆矩阵_cnn_hf_frozen_归一化.png": "第 5 章 实验 —— 主模型逐类表现，重点分析易混类别",
    "混淆矩阵_baseline_hog_svm_归一化.png": "第 5 章 实验 —— 基线对照",
    "混淆矩阵_cnn_scratch_frozen_归一化.png": "第 5 章 消融 —— 无预训练对照",
    "训练曲线_hf_frozen.png": "第 5 章 实验 —— 收敛情况",
    "模型对比.png": "第 5 章 实验 —— 基线 vs 主模型 vs 无预训练",
    "时间线_mixed.png": "第 5 章 系统效果 —— 平滑前后时间线对比（最能体现设计）",
    "消融_时序平滑.png": "第 5 章 消融 —— 事件数对比",
}


def main():
    out = Path(OUTPUT_DIR)
    lines = ["# 实验结果汇总", ""]
    lines.append(f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}")
    lines.append(f"- 数据目录：`{DATA_ROOT}`")
    demo = "synthetic" in str(DATA_ROOT) or "demo" in str(DATA_ROOT)
    lines.append(f"- 数据来源：{'⚠️ 合成/演示数据（仅用于流程验证，论文需替换为真实数据）' if demo else '真实采集数据'}")
    lines.append("")

    # 一、数据集
    summary_path = out / "数据集概况.json"
    lines += ["## 一、数据集", ""]
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        splits = data.get("splits", {})
        lines.append(f"总帧数 **{data.get('frames_total')}**，"
                     f"训练 {splits.get('train')} / 验证 {splits.get('val')} / 测试 {splits.get('test')}，"
                     f"划分种子 {data.get('seed')}（按来源分层，防止同源帧跨集合）")
        lines.append("")
        lines += ["| 类别 | 训练 | 验证 | 测试 | 来源数 |", "| --- | --- | --- | --- | --- |"]
        for name, val in data.get("per_class", {}).items():
            lines.append(f"| {name} | {val['train']} | {val['val']} | {val['test']} | {val['sources']} |")
        lines += ["", f"来源泄漏：{'有 ❌' if data.get('leaked_sources') else '无 ✅'}；"
                      f"近似重复帧 {data.get('duplicate_frames', 0)} 张；"
                      f"分辨率 {list(data.get('resolutions', {}).keys())}"]
    else:
        lines.append("（还没有 数据集概况.json，先运行 src/check_dataset.py）")
    lines.append("")

    # 二、模型指标
    lines += ["## 二、模型与基线（测试集）", ""]
    metrics = []
    for path in sorted(out.glob("指标_*.json")):
        metrics.append(json.loads(path.read_text(encoding="utf-8")))
    if metrics:
        metrics.sort(key=lambda r: r["macro_f1"], reverse=True)
        lines += ["| 模型 / 基线 | 准确率 | 宏 F1 | 加权 F1 | 测试样本 |", "| --- | --- | --- | --- | --- |"]
        for m in metrics:
            lines.append(f"| {m['tag']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                         f"{m['weighted_f1']:.4f} | {m['samples']} |")
        best = metrics[0]
        lines += ["", f"最优模型 **{best['tag']}**，逐类 F1："
                  + "、".join(f"{k} {v['f1-score']:.3f}" for k, v in best["per_class"].items())]
    else:
        lines.append("（还没有指标文件，先运行 src/baseline_hog_svm.py 与 src/train_cnn.py）")
    lines.append("")

    # 三、消融
    lines += ["## 三、消融实验", ""]
    smooth = smoothing_ablation()
    if smooth:
        lines += ["**3.1 时序平滑**（逐帧预测的抖动程度）", "",
                  "| 视频 | 帧数 | 平滑前事件 | 平滑后事件 | 平均片段(前/后) | 一致率 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in smooth:
            lines.append(f"| {r['tag']} | {r['frames']} | {r['events_raw']} | {r['events_smooth']} | "
                         f"{r['avg_len_raw']} / {r['avg_len_smooth']} | {r['agree']:.1%} |")
        lines.append("")
    pretrain = pretrain_ablation()
    if pretrain:
        lines += ["**3.2 预训练**（有无 ImageNet 预训练）", "", "| 模型 | 准确率 | 宏 F1 |", "| --- | --- | --- |"]
        for r in pretrain:
            lines.append(f"| {r['tag']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} |")
        tags = {r["tag"]: r for r in pretrain}
        if {"cnn_hf_frozen", "cnn_scratch_frozen"} <= set(tags):
            lines.append("")
            lines.append(f"提升：宏 F1 {tags['cnn_scratch_frozen']['macro_f1']:.4f} → "
                         f"{tags['cnn_hf_frozen']['macro_f1']:.4f}"
                         f"（{tags['cnn_hf_frozen']['macro_f1'] - tags['cnn_scratch_frozen']['macro_f1']:+.4f}）")
        lines.append("")
    reports = report_ablation(check=True)
    if reports:
        lines += ["**3.3 检索增强（RAG）与数字忠实度**", "",
                  "| 报告 | 字数 | 引用建议 | 无效引用 | 数字可溯源率 |", "| --- | --- | --- | --- | --- |"]
        for r in reports:
            lines.append(f"| {r['tag']} | {r['chars']} | {r['citations']} | "
                         f"{r['invalid_citations'] or '无'} | {(r.get('rate') or 0):.1%} |")
        lines.append("")
    lines.append("")

    # 四、图表清单
    lines += ["## 四、图表清单（建议放哪一节）", ""]
    lines += ["| 文件 | 用途 |", "| --- | --- |"]
    for name, usage in FIGURE_USAGE.items():
        if (out / name).exists():
            lines.append(f"| `outputs/{name}` | {usage} |")
    others = sorted(p.name for p in out.glob("*.png") if p.name not in FIGURE_USAGE)
    for name in others:
        lines.append(f"| `outputs/{name}` | 备选/过程图 |")
    lines.append("")

    # 五、复现命令
    lines += ["## 五、复现命令", "",
              "```bash",
              "bash install.sh                      # 一键安装 + 环境自检",
              "bash run_all.sh [视频相对路径]        # 抽帧→划分→体检→基线→训练→索引→时间线→报告→对比→消融",
              "python web/app.py                    # 启动演示网页 http://127.0.0.1:8002",
              "```", ""]
    lines += ["## 六、已实现的类别与业务含义", "",
              "| 类别 | 中文 | 在报告中的含义 |", "| --- | --- | --- |"]
    meanings = {"listening": "抬头听讲，计入抬头率", "raising_hand": "主动参与，统计举手次数",
                "reading": "阅读材料，计入低头率", "writing": "书写记录，计入低头率",
                "desk": "低头/转头等非听讲姿态，计入低头率", "phone": "分心信号，统计次数"}
    for key in CLASSES:
        lines.append(f"| {key} | {CLASS_NAMES_CN.get(key, key)} | {meanings.get(key, '-')} |")
    lines.append("")

    target = out / "实验结果汇总.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n写入 {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
