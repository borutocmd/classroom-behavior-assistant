# -*- coding: utf-8 -*-
"""演示服务：上传/选择一段课堂视频 → 行为时间线 + 统计 + 观察报告。

启动：
    python web/app.py            # 打开 http://127.0.0.1:8002
接口文档：
    http://127.0.0.1:8002/docs

相比早期版本新增：
  · GET /progress        实时查看 pipeline 运行到哪一步（前端轮询显示）
  · 识别前 / 识别后对比图  随机抽取若干帧，分别给出原帧与叠加了预测标签的帧
"""
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from config import (CLASS_NAMES_CN, DEMO_DIR, MODEL_TAG, OUTPUT_DIR, RAW_DIR,  # noqa: E402
                    WEB_HOST, WEB_PORT, ensure_dirs)

app = FastAPI(title="课堂行为辅助识别", description="视频 → 行为时间线 + 课堂统计 + 观察报告")
ensure_dirs()
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(PROJECT / "web" / "static")), name="static")

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SOURCE_DIRS = {"raw": RAW_DIR, "demo": DEMO_DIR}   # demo 目录放用图片拼出来的演示视频
SNAPSHOT_COUNT = 6                                  # 识别前/后对比抽几帧
_cache = {"model": None, "classes": None, "size": None}

# 前端每秒轮询 /progress 显示进度；每个阶段都会更新这个字典
STAGES = ["接收请求", "读取视频并抽帧", "逐帧行为分类", "时序平滑", "课堂统计",
          "生成对比图", "检索教学建议", "生成观察报告", "整理结果"]
_progress = {"running": False, "stage_index": 0, "stage": "空闲", "detail": "",
             "percent": 0, "elapsed": 0.0, "video": "", "stages": STAGES,
             "finished": False, "error": ""}


class AnalyzeRequest(BaseModel):
    video: str                     # 形如 "raw/desk/xxx.mp4" 或 "demo/demo_mixed.mp4"
    fps: float = 1.0
    with_report: bool = True


def set_progress(index, detail="", running=True, error=""):
    _progress.update({
        "running": running, "stage_index": index, "stage": STAGES[index],
        "detail": detail, "percent": int(round(index / (len(STAGES) - 1) * 100)),
        "elapsed": round(time.perf_counter() - _progress["t0"], 1) if _progress.get("t0") else 0.0,
        "finished": not running and not error, "error": error,
    })


def list_videos():
    items = []
    for tag, root in SOURCE_DIRS.items():
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in VIDEO_EXT:
                items.append({"name": path.stem, "label": f"{tag}/{path.parent.name}",
                              "path": f"{tag}/{path.relative_to(root).as_posix()}",
                              "size_mb": round(path.stat().st_size / 1024 ** 2, 2)})
    return items


def resolve_video(rel: str):
    """把 'raw/...' 或 'demo/...' 解析成真实路径，并限制在数据目录内。"""
    parts = rel.split("/", 1)
    if len(parts) != 2 or parts[0] not in SOURCE_DIRS:
        raise HTTPException(status_code=400, detail="路径要以 raw/ 或 demo/ 开头")
    root = SOURCE_DIRS[parts[0]].resolve()
    target = (root / parts[1]).resolve()
    if root not in target.parents or not target.exists():
        raise HTTPException(status_code=404, detail=f"找不到视频 {rel}")
    return target


def make_comparison_images(frames, times, raw_idx, raw_conf, smooth_idx, classes, tag):
    """随机抽若干帧，生成“识别前（原帧）”与“识别后（叠加预测标签）”两张图。"""
    from PIL import ImageDraw, ImageFont

    font_path = PROJECT / "assets" / "fonts" / "simhei.ttf"
    count = min(SNAPSHOT_COUNT, len(frames))
    picks = sorted(random.sample(range(len(frames)), count))
    pairs = []
    for order, i in enumerate(picks, 1):
        img = frames[i][1].convert("RGB").resize((360, 360))
        before_name = f"对比_{tag}_{order}_识别前.jpg"
        after_name = f"对比_{tag}_{order}_识别后.jpg"
        img.save(OUTPUT_DIR / before_name, quality=92)

        canvas = img.copy()
        draw = ImageDraw.Draw(canvas)
        label = CLASS_NAMES_CN.get(classes[raw_idx[i]], classes[raw_idx[i]])
        smooth_label = CLASS_NAMES_CN.get(classes[smooth_idx[i]], classes[smooth_idx[i]])
        conf = float(raw_conf[i])
        try:
            font_big = ImageFont.truetype(str(font_path), 26)
            font_small = ImageFont.truetype(str(font_path), 20)
        except OSError:
            font_big = font_small = None
        box = f"{label}  {conf:.0%}"
        draw.rectangle([0, 0, 360, 74], fill=(33, 104, 74))
        draw.text((12, 8), box, fill=(255, 255, 255), font=font_big)
        draw.text((12, 44), f"平滑后：{smooth_label}", fill=(214, 240, 224), font=font_small)
        draw.rectangle([0, 332, 360, 360], fill=(0, 0, 0))
        draw.text((12, 334), f"第 {i + 1} 帧 · {times[i]:.0f} 秒", fill=(255, 255, 255),
                  font=font_small)
        canvas.save(OUTPUT_DIR / after_name, quality=92)
        pairs.append({
            "index": i + 1, "second": round(times[i], 1),
            "before": f"/outputs/{before_name}", "after": f"/outputs/{after_name}",
            "label": label, "label_key": classes[raw_idx[i]],
            "smooth_label": smooth_label, "confidence": round(conf, 3),
        })
    return pairs


@app.get("/")
def index():
    return FileResponse(str(PROJECT / "web" / "static" / "index.html"))


@app.get("/progress")
def progress():
    return _progress


@app.get("/examples")
def examples():
    videos = list_videos()
    return {"videos": videos[:20], "total": len(videos)}


@app.get("/model-info")
def model_info():
    info = {"models": []}
    for cfg_path in sorted((PROJECT / "models").glob("*/config.json")):
        info["models"].append(json.loads(cfg_path.read_text(encoding="utf-8")))
    metrics = []
    for metric_path in sorted(OUTPUT_DIR.glob("指标_*.json")):
        data = json.loads(metric_path.read_text(encoding="utf-8"))
        metrics.append({"tag": data["tag"], "accuracy": data["accuracy"],
                        "macro_f1": data["macro_f1"], "samples": data["samples"]})
    info["metrics"] = metrics
    info["model_tag"] = MODEL_TAG
    return info


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    from src.analyze_video import build_stats, plot_timeline, read_frames, smooth_predictions
    from src.infer_utils import load_model, predict_batch
    import numpy as np

    started = time.perf_counter()
    _progress["t0"] = started
    _progress["video"] = req.video
    set_progress(0, f"收到请求：{req.video}")
    try:
        video_path = resolve_video(req.video)

        if _cache["model"] is None:
            set_progress(1, "首次请求，正在加载模型（约 3 秒）…")
            _cache["model"], _cache["classes"], _cache["size"] = load_model(MODEL_TAG)
        model, classes, size = _cache["model"], _cache["classes"], _cache["size"]

        set_progress(1, "按 1 fps 抽帧…")
        frames = read_frames(video_path, req.fps, size)
        if not frames:
            raise HTTPException(status_code=400, detail="视频没有抽到任何帧")
        times = [t for t, _ in frames]

        raw_idx, raw_conf, prob_chunks = [], [], []
        for i in range(0, len(frames), 32):
            set_progress(2, f"分类 {min(i + 32, len(frames))}/{len(frames)} 帧…")
            chunk = [im for _, im in frames[i:i + 32]]
            idx, conf, probs = predict_batch(model, chunk, size)
            raw_idx += idx
            raw_conf += conf
            prob_chunks.append(probs)
        probs = (np.concatenate(prob_chunks, axis=0) if prob_chunks
                 else np.zeros((0, len(classes))))

        set_progress(3, "滑窗概率加权投票 + 合并过短事件…")
        _, smooth_idx = smooth_predictions(raw_idx, probs)
        raw_labels = [classes[i] for i in raw_idx]
        smooth_labels = [classes[i] for i in smooth_idx]

        set_progress(4, "计算抬头率 / 举手次数 / 低头时长…")
        step = (times[1] - times[0]) if len(times) > 1 else 1.0
        stats = build_stats(times, smooth_labels, classes)
        stats["平均置信度"] = round(float(np.mean(raw_conf)), 4)
        stats["平滑前后一致率"] = round(
            sum(a == b for a, b in zip(raw_labels, smooth_labels)) / len(raw_labels), 4)
        stats["模型"] = MODEL_TAG
        stats["输入"] = req.video
        stats["采样间隔秒"] = round(step, 2)

        tag = video_path.stem
        plot_name = f"时间线_{tag}.png"
        plot_timeline(times, raw_labels, smooth_labels, classes, stats, OUTPUT_DIR / plot_name)
        (OUTPUT_DIR / f"统计_{tag}.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

        set_progress(5, f"随机抽取 {SNAPSHOT_COUNT} 帧做识别前/后对比…")
        comparisons = make_comparison_images(frames, times, raw_idx, raw_conf, smooth_idx,
                                             classes, tag)

        result = {"video": req.video, "tag": tag, "stats": stats,
                  "timeline_image": f"/outputs/{plot_name}",
                  "comparisons": comparisons,
                  "timeline": {"times": [round(t, 2) for t in times],
                               "raw": raw_labels, "smoothed": smooth_labels}}

        if req.with_report:
            from src.report_llm import generate_report, retrieve, stats_to_text
            set_progress(6, "把统计转成查询，从 24 条教学建议里检索 Top-3…")
            query = (f"抬头率 {stats['抬头率']:.1%}，低头率 {stats['低头率']:.1%}，"
                     f"举手 {stats['举手次数']} 次，使用手机 {stats['手机次数']} 次，"
                     f"低头时长 {stats['低头时长秒']} 秒的课堂，教师应如何改进")
            tips = retrieve(query)
            set_progress(7, "本地 Qwen3.5-0.8B 生成报告（首次含加载约 15 秒）…")
            report, _prompt = generate_report(stats, tips)
            result["report"] = report
            result["tips"] = [{"citation": t["citation"], "title": t["title"],
                               "content": t["content"], "score": round(t["score"], 3)}
                              for t in tips]
            result["stats_text"] = stats_to_text(stats)

        set_progress(8, "整理 JSON 返回前端…")
        result["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        set_progress(8, f"完成，总耗时 {result['duration_ms'] / 1000:.1f} 秒", running=False)
        return JSONResponse(result)
    except HTTPException as exc:
        set_progress(_progress["stage_index"], str(exc.detail), running=False, error=str(exc.detail))
        raise
    except Exception as exc:  # noqa: BLE001
        set_progress(_progress["stage_index"], f"出错：{exc}", running=False, error=str(exc))
        raise HTTPException(status_code=500, detail=f"分析失败：{exc}") from exc


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="课堂行为辅助识别演示服务")
    parser.add_argument("--host", default=WEB_HOST, help=f"监听地址（默认 {WEB_HOST}）")
    parser.add_argument("--port", type=int, default=WEB_PORT, help=f"端口（默认 {WEB_PORT}）")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载（演示时不要开）")
    args = parser.parse_args()
    uvicorn.run("web.app:app" if args.reload else app, host=args.host, port=args.port,
                reload=args.reload)
