#!/usr/bin/env bash
# 课堂行为辅助识别 —— 一键安装
# 用法： bash install.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIRROR="${PIP_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"
DATA_ROOT="${CLASSROOM_DATA:-$HOME/project/classroom_data}"

echo "==> 项目目录：$HERE"
echo "==> 数据目录：$DATA_ROOT"

# 1) 找 Python 环境：优先复用已有 venv（含 torch），否则新建
CANDIDATES=("${CLASSROOM_VENV:-}" "/home/muli/project/NLP_lab/.venv" "$HERE/.venv")
PY=""
for c in "${CANDIDATES[@]}"; do
  [ -n "$c" ] && [ -x "$c/bin/python" ] && PY="$c/bin/python" && break
done

if [ -z "$PY" ]; then
  echo "==> 未找到可用环境，新建 venv（首次需下载 torch，约 2.5 GB，请耐心等待）"
  python3 -m venv "$HERE/.venv" || { echo "!! 创建 venv 失败，请检查 python3-venv 是否安装"; exit 1; }
  PY="$HERE/.venv/bin/python"
  "$PY" -m pip install --upgrade pip -i "$MIRROR"
fi
echo "==> 使用 Python：$PY"
"$PY" -c "import torch,sys;print('   torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>/dev/null \
  || echo "   注意：该环境还没有 torch，将在下一步安装"

# 2) 安装依赖（已装的自动跳过）
echo "==> 安装依赖（镜像：$MIRROR）"
"$PY" -m pip install -r "$HERE/requirements.txt" -i "$MIRROR" || { echo "!! 依赖安装失败，检查网络或换镜像：PIP_MIRROR=... bash install.sh"; exit 1; }

# 3) 数据目录骨架
mkdir -p "$DATA_ROOT"/{raw,frames,weights,runs} "$HERE/models" "$HERE/outputs"
echo "==> 数据目录已就绪：$DATA_ROOT"

# 4) 自检
echo "==> 运行自检"
"$PY" "$HERE/tests/smoke_test.py"
RC=$?
if [ $RC -eq 0 ]; then
  echo "==> 安装完成，可以开始： $PY src/prepare_frames.py --help"
else
  echo "!! 自检未通过，请把上面的输出发给助手"
fi
exit $RC
