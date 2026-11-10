# -*- coding: utf-8 -*-
"""把一段视频变成“行为时间线 + 课堂统计”。

流程：视频 → 逐帧分类 → 滑窗多数投票 → 短事件合并 → 时间线 json + 统计 + 可视化。

用法：
    python src/analyze_video.py --video <数据目录>/raw/writing/demo.mp4 --tag demo
    python src/analyze_video.py --frames-dir <目录> --tag demo     # 也可以直接给图片序列
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (CLASS_NAMES_CN, MIN_EVENT_FRAMES, MODEL_TAG, OUTPUT_DIR,  # noqa: E402
                    SMOOTH_WINDOW)
from src.evaluate import setup_matplotlib  # noqa: E402
from src.infer_utils import load_model, predict_batch  # noqa: E402

CLASS_COLORS = ["#4C9F70", "#E4B363", "#5B8DEF", "#B26CD6", "#E06C75", "#8A8F98"]


def read_frames(video: Path, fps: float, size: int):
    """按目标 fps 抽帧，返回 (帧号, PIL 图) 列表。"""
    from PIL import Image
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频 {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / max(fps, 0.01))))
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            rgb = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
            frames.append((i / src_fps, Image.fromarray(rgb)))
        i += 1
    cap.release()
    return frames


def read_image_dir(folder: Path, size: int):
    from PIL import Image
    paths = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    return [(float(i), Image.open(p)) for i, p in enumerate(paths)]


def smooth_predictions(indices, probs, window=SMOOTH_WINDOW, min_event=MIN_EVENT_FRAMES):
    """滑窗多数投票（按置信度加权）+ 过短事件并入相邻类别。"""
    n = len(indices)
    if n == 0:
        return [], []
    half = max(1, window // 2)
    smoothed = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        score = Counter()
        for j in range(lo, hi):
            score[indices[j]] += float(probs[j][indices[j]])
        smoothed.append(score.most_common(1)[0][0])

    # 合并过短事件：把长度 < min_event 的片段改成前一段的类别
    result, start = smoothed[:], 0
    while start < n:
        end = start
        while end + 1 < n and smoothed[end + 1] == smoothed[start]:
            end += 1
        if end - start + 1 < min_event and start > 0:
            for k in range(start, end + 1):
                result[k] = result[start - 1]
        start = end + 1
    return smoothed, result


def to_events(times, labels):
    events = []
    for t, lab in zip(times, labels):
        if events and events[-1]["label"] == lab:
            events[-1]["end"] = round(t, 1)
            events[-1]["frames"] += 1
        else:
            events.append({"label": lab, "start": round(t, 1), "end": round(t, 1), "frames": 1})
    return events


def build_stats(times, labels, classes):
    total = max(len(labels), 1)
    counts = Counter(labels)
    step = (times[1] - times[0]) if len(times) > 1 else 1.0
    events = to_events(times, labels)
    raise_hand_events = [e for e in events if e["label"] == "raising_hand"]
    phone_events = [e for e in events if e["label"] == "phone"]
    head_up = counts.get("listening", 0) + counts.get("raising_hand", 0)
    head_down = counts.get("writing", 0) + counts.get("phone", 0) + counts.get("desk", 0)
    return {
        "帧数": len(labels),
        "时长秒": round(len(labels) * step, 1),
        "各类帧数": {classes[i]: counts.get(classes[i], 0) for i in range(len(classes))},
        "各类占比": {CLASS_NAMES_CN[classes[i]]: round(counts.get(classes[i], 0) / total, 4)
                     for i in range(len(classes))},
        "抬头率": round(head_up / total, 4),
        "低头率": round(head_down / total, 4),
        "举手次数": len(raise_hand_events),
        "手机次数": len(phone_events),
        "低头时长秒": round(head_down * step, 1),
        "事件数": len(events),
        "事件列表": events,
    }


def plot_timeline(times, raw_labels, smooth_labels, classes, stats, out_path):
    plt = setup_matplotlib()
    names = [CLASS_NAMES_CN[c] for c in classes]
    index = {c: i for i, c in enumerate(classes)}
    raw = np.array([[index[c] for c in raw_labels]])
    sm = np.array([[index[c] for c in smooth_labels]])

    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.5, 0.5, 1.6], hspace=0.55)
    for row, (data, title) in enumerate([(raw, "平滑前（逐帧预测）"), (sm, "平滑后（滑窗投票 + 事件合并）")]):
        ax = fig.add_subplot(gs[row])
        ax.imshow(data, aspect="auto", cmap="tab10", vmin=0, vmax=9,
                  extent=[times[0], times[-1] if len(times) > 1 else 1, 0, 1])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11)

    ax = fig.add_subplot(gs[2])
    values = [stats["各类帧数"][c] for c in classes]
    bars = ax.bar(names, values, color=CLASS_COLORS[:len(classes)])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("帧数")
    ax.set_title(f"行为分布（抬头率 {stats['抬头率']:.0%}，举手 {stats['举手次数']} 次，"
                 f"手机 {stats['手机次数']} 次，低头 {stats['低头时长秒']:.0f} 秒）", fontsize=11)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="视频 → 行为时间线 + 课堂统计")
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--model", default=MODEL_TAG, help="models/<tag> 里的模型（默认取 config.MODEL_TAG）")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--tag", default="demo")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    if not args.video and not args.frames_dir:
        print("请用 --video 或 --frames-dir 指定输入")
        return 1

    model, classes, image_size = load_model(args.model)
    size = args.size or image_size
    print(f"模型 {args.model}（{len(classes)} 类），输入 {'视频' if args.video else '图片序列'}")

    if args.video:
        frames = read_frames(Path(args.video), args.fps, size)
    else:
        frames = read_image_dir(Path(args.frames_dir), size)
    if not frames:
        print("没有抽到任何帧")
        return 1
    times = [t for t, _ in frames]
    print(f"抽帧 {len(frames)} 张，开始逐帧推理…")

    raw_idx, raw_conf, prob_chunks = [], [], []
    for i in range(0, len(frames), args.batch):
        chunk = [im for _, im in frames[i:i + args.batch]]
        idx, conf, probs = predict_batch(model, chunk, size)
        raw_idx += idx
        raw_conf += conf
        prob_chunks.append(probs)

    # 用真实概率做加权投票：置信度低的帧对平滑结果影响更小
    probs = np.concatenate(prob_chunks, axis=0) if prob_chunks else np.zeros((0, len(classes)))
    raw_labels = [classes[i] for i in raw_idx]
    _, smooth_idx = smooth_predictions(raw_idx, probs)
    smooth_labels = [classes[i] for i in smooth_idx]

    step = (times[1] - times[0]) if len(times) > 1 else 1.0
    stats = build_stats(times, smooth_labels, classes)
    stats["平均置信度"] = round(float(np.mean(raw_conf)), 4)
    stats["模型"] = args.model
    stats["输入"] = str(args.video or args.frames_dir)
    stats["采样间隔秒"] = round(step, 2)
    stats["平滑前后一致率"] = round(sum(a == b for a, b in zip(raw_labels, smooth_labels)) / len(raw_labels), 4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timeline_path = OUTPUT_DIR / f"时间线_{args.tag}.json"
    timeline_path.write_text(json.dumps({
        "times": [round(t, 2) for t in times],
        "raw": raw_labels,
        "raw_confidence": [round(c, 4) for c in raw_conf],
        "smoothed": smooth_labels,
        "stats": stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / f"统计_{args.tag}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_timeline(times, raw_labels, smooth_labels, classes, stats,
                  OUTPUT_DIR / f"时间线_{args.tag}.png")

    print(f"\n时长 {stats['时长秒']} 秒 | 抬头率 {stats['抬头率']:.1%} | 举手 {stats['举手次数']} 次 | "
          f"手机 {stats['手机次数']} 次 | 低头 {stats['低头时长秒']} 秒")
    print(f"平滑前后一致率 {stats['平滑前后一致率']:.1%}（越低说明逐帧抖动越厉害）")
    print("事件序列：" + " → ".join(f"{CLASS_NAMES_CN[e['label']]}({e['start']:.0f}s)"
                                    for e in stats["事件列表"][:12]))
    print(f"\n时间线 {timeline_path}\n统计图 {OUTPUT_DIR / f'时间线_{args.tag}.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
