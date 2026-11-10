#!/bin/bash
# 启动演示服务（默认用真实数据目录 classroom_data）
# 项目目录由脚本自身位置推断：文件夹改名/移动都不会失效
set -uo pipefail
PY=/home/muli/project/NLP_lab/.venv/bin/python
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLASSROOM_DATA="${CLASSROOM_DATA:-/home/muli/project/classroom_data}"
export PYTHONPATH="$PROJ"
cd "$PROJ" || exit 1
echo "项目目录： $PROJ"

pkill -f "web/app.py" 2>/dev/null
sleep 1
setsid nohup "$PY" web/app.py > /tmp/demo_web.log 2>&1 < /dev/null &
disown 2>/dev/null || true

for i in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8002/ 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 1
done
if [ "$code" = "200" ]; then
  echo "服务已启动： http://127.0.0.1:8002"
  echo "数据目录： $CLASSROOM_DATA"
  echo "可选视频："
  curl -s http://127.0.0.1:8002/examples | $PY -c "
import sys, json
d = json.load(sys.stdin)
for v in d['videos']:
    print(f\"   {v['path']}  ({v['size_mb']} MB)\")
print(f'   共 {d[\"total\"]} 段')
"
  echo "停止服务：双击 停止演示.bat"
else
  echo "启动失败，请看日志： /tmp/demo_web.log"
  tail -20 /tmp/demo_web.log
  exit 1
fi
