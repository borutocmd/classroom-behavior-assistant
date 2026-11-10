#!/bin/bash
# 全流程一键运行：抽帧 → 划分 → 体检 → 基线 → 主模型 → 视频分析 → 报告 → 对比 → 消融
# 用法：
#   bash run_all.sh                      # 只跑到模型对比（不分析视频）
#   bash run_all.sh desk/xxx.mp4         # 视频相对数据目录的 raw/ 路径
#   bash run_all.sh /tmp/xxx.mp4         # 也可以直接给绝对路径
# 环境变量：
#   CLASSROOM_DATA   数据根目录（默认 ~/project/classroom_data）
#   EPOCHS           训练轮数（默认 15）
set -uo pipefail
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${CLASSROOM_DATA:-$HOME/project/classroom_data}"
VIDEO="${1:-}"
cd "$HERE" || exit 1
export PYTHONPATH="$HERE"

step() { echo; echo "==================== $* ===================="; }

if [ "${CLEAN:-0}" = "1" ]; then
  echo "清理旧指标与图表（CLEAN=1）"
  rm -f outputs/指标_*.json outputs/混淆矩阵_*.png outputs/训练曲线_*.png \
        outputs/时间线_*.json outputs/时间线_*.png outputs/统计_*.json \
        outputs/报告_*.json outputs/课堂观察报告_*.md outputs/模型对比.* \
        outputs/消融*.md outputs/消融_*.png
fi

step "0/9 环境自检"
$PY tests/smoke_test.py || { echo "自检未通过，先修环境"; exit 1; }

step "1/9 抽帧 + 自动标注"
$PY src/prepare_frames.py --fps 1 --size 256 || exit 1

step "2/9 分层划分（按来源）"
$PY src/make_splits.py || exit 1

step "3/9 数据集体检"
$PY src/check_dataset.py || exit 1

step "4/9 HOG + SVM 基线"
$PY src/baseline_hog_svm.py || exit 1

step "5/9 ResNet18 微调（预训练 + 冻结主干）"
$PY tools/fetch_backbone.py >/dev/null 2>&1
$PY src/train_cnn.py --backbone hf --epochs "${EPOCHS:-15}" || exit 1

step "6/9 教学建议索引"
$PY src/build_index.py || exit 1

TAG=""
if [ -n "$VIDEO" ]; then
  case "$VIDEO" in
    /*) VIDEOPATH="$VIDEO" ;;
    *)  VIDEOPATH="$DATA_ROOT/raw/$VIDEO" ;;
  esac
  TAG="$(basename "$VIDEO" .mp4)"
  step "7/9 视频 → 行为时间线 + 统计（$TAG）"
  $PY src/analyze_video.py --video "$VIDEOPATH" --tag "$TAG" --fps 1 || exit 1
  step "8/9 生成课堂观察报告"
  $PY src/report_llm.py --stats "outputs/统计_$TAG.json" --tag "$TAG" || exit 1
else
  echo
  echo "（未提供视频参数，跳过第 7、8 步；用法： bash run_all.sh desk/xxx.mp4）"
fi

step "9/9 模型对比 + 消融汇总"
$PY src/compare_models.py || exit 1
$PY src/ablation.py --check-report || exit 1

echo
echo "全部完成。产出在 outputs/，模型在 models/，启动演示： $PY web/app.py"
