@echo off
chcp 65001 >nul
echo 正在停止演示服务...
wsl.exe -e bash -c "pkill -f web/app.py"
echo 已停止（如果之前没在运行，这条命令也不会有影响）。
timeout /t 3 >nul
