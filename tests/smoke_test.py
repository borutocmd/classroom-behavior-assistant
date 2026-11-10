# -*- coding: utf-8 -*-
"""环境自检：依赖、显卡、路径、模型资产。全部通过时退出码为 0。"""
import importlib
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("CLASSROOM_DATA", Path.home() / "project" / "classroom_data"))
sys.path.insert(0, str(PROJECT))
from config import EMBED_DIR, QWEN_DIR  # noqa: E402  单一事实来源

ok = True


def check(name, fn):
    global ok
    try:
        info = fn()
        print(f"  [通过] {name:26s} {info}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  [失败] {name:26s} {type(exc).__name__}: {exc}")


def dep(mod):
    def inner():
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "已安装")
    return inner


print("== 1. 依赖 ==")
for mod in ["torch", "torchvision", "cv2", "numpy", "sklearn", "pandas",
            "matplotlib", "transformers", "faiss", "jieba", "fastapi", "uvicorn", "multipart"]:
    check(mod, dep(mod))

print("== 2. 显卡 ==")
def gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，将只能用 CPU 训练（很慢）")
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return f"{name} / {mem:.1f} GB"
check("CUDA", gpu)

print("== 3. 目录 ==")
for path in [PROJECT, DATA_ROOT, DATA_ROOT / "frames", DATA_ROOT / "weights"]:
    check(str(path), lambda p=path: "存在" if p.exists() else (_ for _ in ()).throw(FileNotFoundError("不存在")))

print("== 4. 可复用资产 ==")
check("本地 Qwen3.5-0.8B", lambda: "存在" if QWEN_DIR.exists()
      else (_ for _ in ()).throw(FileNotFoundError("未找到，报告生成会失败")))
check("bge-small-zh 向量模型", lambda: "存在" if EMBED_DIR.exists()
      else (_ for _ in ()).throw(FileNotFoundError("未找到，教学建议检索会失败")))

print()
print("自检结果：" + ("全部通过 ✅" if ok else "有失败项 ❌（把以上输出发给助手）"))
sys.exit(0 if ok else 1)
