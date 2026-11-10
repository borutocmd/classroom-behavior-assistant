# -*- coding: utf-8 -*-
"""课堂观察报告生成：统计 → 检索教学建议 → 本地 Qwen3.5-0.8B → 报告。

要点：
  1. 报告里的每个数字都必须来自统计文件，Prompt 里明确禁止编造；
  2. 教学建议由 FAISS 检索得到，报告用 [资料N] 引用，便于人工核对；
  3. 没有统计输入时不调用大模型，直接走门控分支。

用法：
    python src/report_llm.py --stats outputs/统计_demo.json --tag demo
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# 必须在 import torch 之前设置：禁用原生 JIT，避免 triton 在无 C 编译器时加载失败
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import faiss
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (EMBED_DIR, KB_DIR, MAX_NEW_TOKENS, OUTPUT_DIR,  # noqa: E402
                    QWEN_DIR, TOP_K_TIPS)

SYSTEM_PROMPT = (
    "你是一名教学法教研员，负责根据课堂行为统计为任课教师写一段课后观察报告。\n"
    "必须遵守以下规则：\n"
    "1. 只依据给定的统计数字与检索到的教学建议，禁止编造任何数据或事实；\n"
    "2. 报告中的每个数字都必须来自统计表，出现数字时用（统计）标注来源；\n"
    "3. 给出建议时，每条建议的句末必须紧跟 [资料N] 标注依据；没有资料支撑的建议一律不写；\n"
    "4. 统计未涉及的方面，直接写“本次统计未涉及”，不要推测；\n"
    "5. 报告用简洁中文，先给一段总体描述，再给 3 条可执行的改进建议，最后一句写使用这类数据时应注意的边界。"
)


def load_tips():
    path = KB_DIR / "teaching_tips.json"
    return json.loads(path.read_text(encoding="utf-8"))


def retrieve(query: str, k=TOP_K_TIPS):
    index_path = KB_DIR / "tips.faiss"
    if not index_path.exists():
        raise FileNotFoundError("缺少 kb/tips.faiss，先运行 python src/build_index.py")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(EMBED_DIR))
    vector = model.encode([query], normalize_embeddings=True,
                          show_progress_bar=False).astype("float32")
    index = faiss.read_index(str(index_path))
    scores, positions = index.search(vector, k)
    tips = load_tips()
    return [dict(tips[int(i)], score=float(s), citation=f"资料{n}")
            for n, (i, s) in enumerate(zip(positions[0], scores[0]), 1)]


def stats_to_text(stats):
    dist = "，".join(f"{k} {v:.1%}" for k, v in stats["各类占比"].items())
    return (f"样本时长 {stats['时长秒']} 秒（{stats['帧数']} 帧，采样间隔 {stats['采样间隔秒']} 秒）。\n"
            f"行为分布：{dist}。\n"
            f"抬头率 {stats['抬头率']:.1%}，低头率 {stats['低头率']:.1%}，"
            f"低头时长 {stats['低头时长秒']} 秒。\n"
            f"举手次数 {stats['举手次数']} 次，使用手机 {stats['手机次数']} 次，"
            f"行为片段共 {stats['事件数']} 段。")


def load_llm():
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(str(QWEN_DIR), local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(QWEN_DIR), local_files_only=True, dtype=dtype).to(device).eval()
    return processor, model


@torch.inference_mode()
def generate_report(stats, tips):
    context = ("\n\n".join(f"[{t['citation']}] {t['title']}：{t['content']}" for t in tips)
               if tips else "（本次没有提供教学建议资料，请只依据统计数字写报告，"
                            "不要给出具体教学建议。）")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"下面是本次课堂的行为统计：\n\n<课堂统计>\n{stats_to_text(stats)}\n</课堂统计>\n\n"
            f"下面是检索到的教学建议：\n\n<教学建议>\n{context}\n</教学建议>\n\n"
            "请写出课后观察报告。"},
    ]
    processor, model = load_llm()
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_dict=True,
                                           return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                         repetition_penalty=1.05)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
    text = re.sub(r"\[\s*(?:资料\s*)?(\d+)\s*\]", r"[资料\1]", text)
    prompt = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    return text, prompt


def main():
    ap = argparse.ArgumentParser(description="生成课堂观察报告")
    ap.add_argument("--stats", default=str(OUTPUT_DIR / "统计_demo.json"))
    ap.add_argument("--tag", default="demo")
    ap.add_argument("--no-rag", action="store_true", help="消融实验：不检索教学建议，直接让模型写报告")
    args = ap.parse_args()
    if args.no_rag:
        args.tag = f"{args.tag}_noRAG"

    stats_path = Path(args.stats)
    if not stats_path.exists():
        print(f"找不到统计文件 {stats_path}，先运行 src/analyze_video.py")
        return 1
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    if stats.get("帧数", 0) == 0:
        print("统计为空，按门控要求不调用大模型：根据当前统计无法生成报告。")
        return 0

    if args.no_rag:
        tips = []
        print("消融模式：不检索教学建议，仅用统计数字生成报告")
    else:
        query = (f"抬头率 {stats['抬头率']:.1%}，低头率 {stats['低头率']:.1%}，"
                 f"举手 {stats['举手次数']} 次，使用手机 {stats['手机次数']} 次，"
                 f"低头时长 {stats['低头时长秒']} 秒的课堂，教师应如何改进")
        tips = retrieve(query)
        print("检索到的教学建议：")
        for t in tips:
            print(f"  [{t['citation']}] {t['title']}（相似度 {t['score']:.3f}）")

    print("\n调用本地 Qwen3.5-0.8B 生成报告…")
    report, prompt = generate_report(stats, tips)

    cited = set(re.findall(r"\[(资料\d+)\]", report))
    valid = {t["citation"] for t in tips}
    result = {
        "tag": args.tag,
        "stats_source": str(stats_path),
        "stats": {k: v for k, v in stats.items() if k != "事件列表"},
        "retrieved_tips": [{k: v for k, v in t.items()} for t in tips],
        "report": report,
        "citation_check": {"无效编号": sorted(cited - valid), "是否漏引用": not cited},
        "prompt": prompt,
        "model": "Qwen3.5-0.8B",
    }
    md_path = OUTPUT_DIR / f"课堂观察报告_{args.tag}.md"
    md_path.write_text(
        f"# 课堂观察报告（{args.tag}）\n\n"
        f"## 一、行为统计\n\n{stats_to_text(stats)}\n\n"
        f"## 二、生成报告\n\n{report}\n\n"
        f"## 三、依据的教学建议\n\n"
        + "\n".join(f"- [{t['citation']}] **{t['title']}**（相似度 {t['score']:.3f}）：{t['content']}"
                    for t in tips)
        + f"\n\n## 四、引用检查\n\n无效编号：{result['citation_check']['无效编号'] or '无'}；"
          f"漏引用：{result['citation_check']['是否漏引用']}\n",
        encoding="utf-8")
    (OUTPUT_DIR / f"报告_{args.tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\n报告已写入 {md_path}")
    print(f"引用检查：无效编号 {result['citation_check']['无效编号'] or '无'}，"
          f"漏引用 {result['citation_check']['是否漏引用']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
