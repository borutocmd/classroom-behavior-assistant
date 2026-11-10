# -*- coding: utf-8 -*-
"""主模型：ResNet18 迁移学习（也可用 --backbone scratch 从头训练作对照）。

用法：
    python tools/fetch_backbone.py                 # 先把预训练权重下到本地（只做一次）
    python src/train_cnn.py                        # 默认：预训练主干 + 冻结卷积层
    python src/train_cnn.py --backbone scratch     # 对照实验：无预训练
    python src/train_cnn.py --no-freeze            # 全网络微调
"""
import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (BATCH_SIZE, CLASSES, EPOCHS, IMAGE_SIZE, LR_FINETUNE, LR_HEAD,  # noqa: E402
                    MODEL_DIR, OUTPUT_DIR, RUN_DIR, SEED, WEIGHT_DECAY, WEIGHT_DIR)
from src.dataset import class_to_idx, load_rows, make_loader  # noqa: E402
from src.evaluate import evaluate_predictions, setup_matplotlib  # noqa: E402

BACKBONE_DIR = WEIGHT_DIR / "resnet18"


def find_last_linear(model):
    last = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last = (name, module)
    return last


def replace_head(model, num_classes):
    """把分类头换成 num_classes 类，返回新头的参数名前缀。"""
    name, module = find_last_linear(model)
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    new_head = nn.Linear(module.in_features, num_classes)
    if name.rsplit(".", 1)[0] if "." in name else "":
        leaf = name.rsplit(".", 1)[1]
        if leaf.isdigit():
            parent[int(leaf)] = new_head
        else:
            setattr(parent, leaf, new_head)
    else:
        setattr(model, name, new_head)
    return name


def build_model(backbone: str, num_classes: int, freeze: bool, device):
    if backbone == "hf":
        if not BACKBONE_DIR.exists():
            raise FileNotFoundError(
                f"未找到本地预训练权重 {BACKBONE_DIR}，先运行：python tools/fetch_backbone.py")
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(
            str(BACKBONE_DIR), num_labels=num_classes, ignore_mismatched_sizes=True)
    else:
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    head_name = replace_head(model, num_classes) if backbone == "hf" else "fc"
    if freeze:
        for _, param in model.named_parameters():
            param.requires_grad = False
        prefix = head_name.split(".")[0]
        for name, param in model.named_parameters():
            if name.startswith(prefix) or name.startswith(head_name):
                param.requires_grad = True
    return model.to(device), head_name


def run_epoch(model, loader, criterion, device, optimizer=None, show_progress=False):
    """跑一轮。传 optimizer 表示训练（开梯度），否则为评估（no_grad）。"""
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    preds, targets = [], []
    context = contextlib.nullcontext() if train else torch.no_grad()
    iterator = loader
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(loader, desc="  训练中" if train else "  验证中", ncols=88, leave=False)
        except ImportError:
            iterator = loader
    with context:
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = model(pixel_values=images).logits if hasattr(model, "config") else model(images)
                loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss) * labels.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum())
            seen += labels.size(0)
            preds += pred.cpu().tolist()
            targets += labels.cpu().tolist()
    return total_loss / max(seen, 1), correct / max(seen, 1), np.array(targets), np.array(preds)


def plot_curves(history, out_path):
    plt = setup_matplotlib()
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="训练损失")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="验证损失")
    axes[0].set_xlabel("轮次")
    axes[0].set_ylabel("损失")
    axes[0].legend()
    axes[0].set_title("损失曲线")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="验证准确率")
    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], label="验证宏F1")
    axes[1].set_xlabel("轮次")
    axes[1].set_ylabel("指标")
    axes[1].legend()
    axes[1].set_title("验证指标")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="训练课堂行为分类模型")
    ap.add_argument("--backbone", choices=["hf", "scratch"], default="hf")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-freeze", action="store_true", help="解冻主干，全网络微调")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="只取前 N 条训练/验证样本做快速试训（0 表示用全部；例如 --limit 2000 约 1 分钟一轮）")
    ap.add_argument("--progress", action="store_true", help="显示每个 batch 的进度条")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or f"{args.backbone}{'_finetune' if args.no_freeze else '_frozen'}"
    if args.limit:
        tag = f"{tag}_limit{args.limit}"

    train_rows = load_rows("train")
    val_rows = load_rows("val")
    test_rows = load_rows("test")
    if not (train_rows and val_rows and test_rows):
        print("train/val/test 有一个为空，先运行 prepare_frames.py 与 make_splits.py")
        return 1
    if args.limit:
        rng = random.Random(SEED)
        rng.shuffle(train_rows)
        rng.shuffle(val_rows)
        train_rows, val_rows = train_rows[:args.limit], val_rows[:max(200, args.limit // 5)]
        print(f"快速试训模式：只取 train {len(train_rows)} / val {len(val_rows)} 条")
    print(f"数据：train {len(train_rows)} / val {len(val_rows)} / test {len(test_rows)}，设备 {device}")

    train_loader = make_loader(train_rows, True, args.image_size, args.batch_size, args.workers)
    val_loader = make_loader(val_rows, False, args.image_size, args.batch_size, args.workers)
    test_loader = make_loader(test_rows, False, args.image_size, args.batch_size, args.workers)

    model, head = build_model(args.backbone, len(CLASSES), not args.no_freeze, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"主干 {args.backbone}，分类头 {head}，可训练参数 {trainable:,} / {total:,}"
          f"（{trainable / total:.1%}）")

    lr = args.lr or (LR_HEAD if not args.no_freeze else LR_FINETUNE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    run_dir = RUN_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], {"val_macro_f1": -1.0, "epoch": 0, "state": None}
    from sklearn.metrics import f1_score

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, device, optimizer,
                                          args.progress)
        va_loss, va_acc, va_true, va_pred = run_epoch(model, val_loader, criterion, device,
                                                      show_progress=args.progress)
        va_f1 = float(f1_score(va_true, va_pred, average="macro", zero_division=0))
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
                        "val_loss": round(va_loss, 4), "val_acc": round(va_acc, 4),
                        "val_macro_f1": round(va_f1, 4), "seconds": round(time.time() - t0, 1)})
        print(f"  第 {epoch:2d}/{args.epochs} 轮  训练损失 {tr_loss:.4f} 训练准确率 {tr_acc:.4f} | "
              f"验证损失 {va_loss:.4f} 验证准确率 {va_acc:.4f} 验证宏F1 {va_f1:.4f}")
        if va_f1 > best["val_macro_f1"]:
            best = {"val_macro_f1": va_f1, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    _, _, test_true, test_pred = run_epoch(model, test_loader, criterion, device)
    metrics = evaluate_predictions(test_true, test_pred, f"cnn_{tag}", OUTPUT_DIR)
    plot_curves(history, OUTPUT_DIR / f"训练曲线_{tag}.png")
    (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    save_dir = MODEL_DIR / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")
    (save_dir / "config.json").write_text(json.dumps({
        "tag": tag, "backbone": args.backbone, "head": head,
        "classes": CLASSES, "class_to_idx": class_to_idx(),
        "image_size": args.image_size, "freeze_backbone": not args.no_freeze,
        "best_epoch": best["epoch"], "best_val_macro_f1": round(best["val_macro_f1"], 4),
        "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n最优轮次 {best['epoch']}（验证宏F1 {best['val_macro_f1']:.4f}）"
          f"，测试集准确率 {metrics['accuracy']:.4f} / 宏F1 {metrics['macro_f1']:.4f}")
    print(f"模型保存到 {save_dir}，图表与指标在 {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
