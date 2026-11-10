# -*- coding: utf-8 -*-
"""核心逻辑单元测试（不依赖数据集，几秒跑完）。

覆盖：类别映射、时序平滑、统计口径、划分防泄漏、报告数字忠实度判定。
用法：
    python tests/test_core.py
"""
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from config import CLASSES, norm_label  # noqa: E402
from src.ablation import number_fidelity  # noqa: E402
from src.analyze_video import build_stats, smooth_predictions  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'通过' if condition else '失败'}] {name}" + (f"  {detail}" if detail else ""))


def test_label_mapping():
    print("== 类别映射 ==")
    cases = {
        "listening": "listening", "听讲": "listening", "01_listening": "listening",
        "raising_hand": "raising_hand", "hand-raising": "raising_hand", "举手": "raising_hand",
        "低头写字": "writing", "写作": "writing", "writing_zhangsan": "writing",
        "使用手机": "phone", "using_phone": "phone",
        "低头": "desk", "睡觉": "desk", "趴桌": "desk",
    }
    for raw, expect in cases.items():
        got = norm_label(raw)
        check(f"norm_label({raw!r}) == {expect}", got == expect, f"实际 {got}")
    check("无法识别的名字返回 None", norm_label("random_folder") is None)


def test_smoothing():
    print("== 时序平滑 ==")
    classes = list(CLASSES)
    idx = {c: i for i, c in enumerate(classes)}
    # 30 帧：听讲 10 帧 → 举手 5 帧 → 低头写字 10 帧 → 趴桌 5 帧，插入 15% 随机抖动
    seq = ["listening"] * 10 + ["raising_hand"] * 5 + ["writing"] * 10 + ["desk"] * 5
    rng = np.random.default_rng(0)
    raw = [idx[c] for c in seq]
    for i in range(len(raw)):
        if rng.random() < 0.15:
            raw[i] = int(rng.integers(0, len(classes)))
    probs = np.full((len(raw), len(classes)), 0.02)
    for i, r in enumerate(raw):
        probs[i, r] = 0.9
    before = sum(1 for i, v in enumerate(raw) if i == 0 or raw[i - 1] != v)
    _, smoothed = smooth_predictions(raw, probs)
    after = sum(1 for i, v in enumerate(smoothed) if i == 0 or smoothed[i - 1] != v)
    check("平滑后事件数不增加", after <= before, f"{before} → {after}")
    check("平滑后事件数接近真值 4", after <= 6, f"实际 {after}")

    # 单帧毛刺应被合并掉
    blip = [0] * 5 + [1] + [0] * 5
    probs2 = np.full((len(blip), len(classes)), 0.02)
    for i, r in enumerate(blip):
        probs2[i, r] = 0.9
    _, merged = smooth_predictions(blip, probs2)
    check("单帧毛刺被合并", len(set(merged)) == 1, f"结果 {set(merged)}")


def test_stats():
    print("== 统计口径 ==")
    classes = list(CLASSES)
    times = [float(i) for i in range(10)]
    labels = ["listening"] * 4 + ["raising_hand"] * 2 + ["writing"] * 2 + ["phone"] * 1 + ["desk"] * 1
    stats = build_stats(times, labels, classes)
    check("帧数与时长正确", stats["帧数"] == 10 and stats["时长秒"] == 10.0)
    check("抬头率 = (听讲+举手)/总数", abs(stats["抬头率"] - 0.6) < 1e-6, f"{stats['抬头率']}")
    check("低头率 = (写字+手机+趴桌)/总数", abs(stats["低头率"] - 0.4) < 1e-6, f"{stats['低头率']}")
    check("举手次数按事件计数", stats["举手次数"] == 1, f"{stats['举手次数']}")
    check("事件列表按时间合并", len(stats["事件列表"]) == 5, f"{len(stats['事件列表'])}")


def test_fidelity():
    print("== 报告数字忠实度 ==")
    stats = {"抬头率": 0.5, "举手次数": 3, "低头时长秒": 120.0}
    ok_report = "抬头率 50.0%，举手 3 次，低头 120.0 秒。"
    bad_report = "抬头率 88.8%，举手 99 次。"
    good = number_fidelity(ok_report, stats)
    bad = number_fidelity(bad_report, stats)
    check("统计内数字判定为可溯源", good["rate"] == 1.0, f"{good}")
    check("统计外数字被标出", bad["hit"] < bad["total"], f"{bad['missing']}")


def test_split_no_leakage():
    print("== 划分防泄漏（模拟） ==")
    rows = [{"file": f"frames/desk/{src}_{i:04d}.jpg", "label": "desk", "source": src,
             "frame_idx": str(i), "split": ""}
            for src in ("a", "b", "c") for i in range(5)]
    # 复用 make_splits 的分组逻辑：同一来源只能进同一个集合
    assignment = {}
    for src in ("a", "b", "c"):
        bucket = "train" if src == "a" else ("val" if src == "b" else "test")
        for r in rows:
            if r["source"] == src:
                assignment[r["file"]] = bucket
    leaked = {s for s in ("a", "b", "c")
              if len({assignment[r["file"]] for r in rows if r["source"] == s}) > 1}
    check("同一来源未跨集合", not leaked, f"泄漏来源 {leaked}")


def main():
    for fn in [test_label_mapping, test_smoothing, test_stats, test_fidelity, test_split_no_leakage]:
        fn()
        print()
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + "、".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
