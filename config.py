# -*- coding: utf-8 -*-
"""集中配置：路径与超参数。命令行脚本统一 `from config import ...`。

数据根目录可用环境变量 CLASSROOM_DATA 覆盖（默认放 WSL，避免 OneDrive 同步大量图片）。
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------- 路径 ----------
DATA_ROOT = Path(os.environ.get("CLASSROOM_DATA", Path.home() / "project" / "classroom_data"))
RAW_DIR = DATA_ROOT / "raw"            # 原始视频/图片
FRAME_DIR = DATA_ROOT / "frames"       # 抽帧结果：frames/<类别>/xxx.jpg
RUN_DIR = DATA_ROOT / "runs"           # 训练日志与检查点
DEMO_DIR = DATA_ROOT / "demo_videos"   # 演示视频（用图片拼出来，不参与训练）
# 预训练权重放共享目录：换数据根目录时不用重新下载（可用 CLASSROOM_WEIGHTS 覆盖）
WEIGHT_DIR = Path(os.environ.get("CLASSROOM_WEIGHTS", Path.home() / "project" / "models_cache"))

LABELS_CSV = DATA_ROOT / "labels.csv"  # file,label,split
SPLITS_JSON = DATA_ROOT / "splits.json"
MODEL_DIR = PROJECT_ROOT / "models"    # 训练产物（随代码提交）
OUTPUT_DIR = PROJECT_ROOT / "outputs"  # 指标、图表、报告
KB_DIR = PROJECT_ROOT / "kb"           # 教学建议知识库

# ---------- 数据 ----------
# 类别与数据来源对应：
#   listening    听讲        ← SCB-Dataset「听讲」子集（纯图片，无标注框）
#   raising_hand 举手        ← SCB-Dataset5 子集 A（hand-raising）
#   reading      阅读        ← SCB-Dataset5 子集 A（read）
#   writing      书写        ← SCB-Dataset5 子集 A（write）
#   desk         低头/转头   ← SCB-Dataset5「BowTurnHead」子集
# 使用手机（phone）暂缺真实数据，录好自采视频后把它加回 CLASSES 即可（模型会自动改成 6 分类）。
CLASSES = ["listening", "raising_hand", "reading", "writing", "desk"]
CLASS_NAMES_CN = {
    "listening": "听讲", "raising_hand": "举手", "reading": "阅读",
    "writing": "书写", "desk": "低头/转头", "phone": "使用手机",
}
# 兼容常见数据集里的同义标签（中英文、大小写、下划线差异）。新增写法都加在这里，
# prepare_frames / check_raw / import_yolo_dataset 共用同一张表，避免某一处漏认导致整类被跳过。
LABEL_ALIASES = {
    "listening": "listening", "listen": "listening", "听讲": "listening",
    "抬头": "listening", "抬头听讲": "listening",
    "hand-raising": "raising_hand", "hand_raising": "raising_hand", "raise_hand": "raising_hand",
    "raising_hand": "raising_hand", "raisinghand": "raising_hand", "handup": "raising_hand",
    "举手": "raising_hand", "举手发言": "raising_hand", "学生举手": "raising_hand",
    "reading": "reading", "read": "reading", "阅读": "reading", "看书": "reading", "朗读": "reading",
    "writing": "writing", "write": "writing", "低头写字": "writing",
    "写作": "writing", "写字": "writing", "记笔记": "writing", "书写": "writing", "读写": "writing",
    "using_phone": "phone", "phone": "phone", "usingphone": "phone", "手机": "phone",
    "使用手机": "phone", "玩手机": "phone", "看手机": "phone",
    "head-down": "desk", "head_down": "desk", "sleeping": "desk", "desk": "desk",
    "bow_head": "desk", "turn_head": "desk", "bowturnhead": "desk",
    "低头": "desk", "睡觉": "desk", "趴桌": "desk", "趴桌子": "desk", "睡": "desk",
    "转头": "desk", "低头转头": "desk",
}

# ---------- 训练 ----------
IMAGE_SIZE = 224          # 显存不足时改 128
BATCH_SIZE = 32           # 显存不足时改 16
EPOCHS = 15
LR_HEAD = 1e-3            # 只训练分类头时的学习率
LR_FINETUNE = 1e-4        # 解冻主干后的学习率
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
BACKBONE = "microsoft/resnet-18"   # 从 hf-mirror 下载；也可用 torchvision 权重（本机不通）
# 推理/演示默认用哪个模型（models/<tag>/）。真实数据上的对照：
#   hf_finetune  全网络微调  准确率 0.7142 / 宏F1 0.7812  ← 最优
#   hf_frozen    冻结主干    准确率 0.4810 / 宏F1 0.5795
MODEL_TAG = "hf_finetune"

# ---------- 时序平滑 ----------
SMOOTH_WINDOW = 5         # 滑窗长度（帧数，1 fps 时约 5 秒）
MIN_EVENT_FRAMES = 2      # 事件最短持续帧数，过滤抖动

# ---------- 报告生成 ----------
# 模型目录：默认指向项目内 models/（不入库），可用环境变量覆盖
QWEN_DIR = Path(os.environ.get("QWEN_DIR", PROJECT_ROOT / "models" / "Qwen3.5-0.8B"))
EMBED_DIR = Path(os.environ.get("EMBED_DIR", PROJECT_ROOT / "models" / "embeddings_model"))
MAX_NEW_TOKENS = 320
TOP_K_TIPS = 3            # 检索到的教学建议条数
WEB_HOST = "127.0.0.1"
WEB_PORT = 8002           # 与 day06 的 8001 区分开


def ensure_dirs():
    for path in [DATA_ROOT, RAW_DIR, FRAME_DIR, WEIGHT_DIR, RUN_DIR, DEMO_DIR, MODEL_DIR, OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def norm_label(name: str):
    """把文件夹名/标签名归一化到 6 个标准类别之一，无法识别时返回 None。

    支持 "writing"、"低头写字"、"01_writing"、"writing_张三" 等写法。
    """
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    for alias, label in LABEL_ALIASES.items():
        if key.startswith(alias + "_") or key.endswith("_" + alias):
            return label
    for alias, label in LABEL_ALIASES.items():
        if alias in key:
            return label
    return None


if __name__ == "__main__":
    ensure_dirs()
    print("PROJECT_ROOT", PROJECT_ROOT)
    print("DATA_ROOT   ", DATA_ROOT)
    print("类别        ", list(zip(CLASSES, CLASS_NAMES_CN.values())))
