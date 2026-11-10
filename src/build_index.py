# -*- coding: utf-8 -*-
"""把教学建议知识库编码成向量并建 FAISS 索引（复用 Day03 的检索思路）。

用法：
    python src/build_index.py
"""
import json
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EMBED_DIR, KB_DIR  # noqa: E402

TIPS_PATH = KB_DIR / "teaching_tips.json"
INDEX_PATH = KB_DIR / "tips.faiss"


def tip_text(tip):
    return f"{tip['title']}。{tip['content']} 关键词：{'、'.join(tip.get('tags', []))}"


def main():
    tips = json.loads(TIPS_PATH.read_text(encoding="utf-8"))
    print(f"教学建议 {len(tips)} 条，用 {EMBED_DIR.name} 编码…")
    model = SentenceTransformer(str(EMBED_DIR))
    vectors = model.encode([tip_text(t) for t in tips],
                           normalize_embeddings=True, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"索引维度 {vectors.shape[1]}，写入 {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
