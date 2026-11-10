@echo off
chcp 65001 >nul
title 课堂行为辅助识别 - 启动演示
echo ============================================
echo   课堂行为辅助识别 . 演示服务
echo ============================================
echo.
rem 用本 bat 所在目录推导 WSL 路径，文件夹改名/移动都不受影响（要求项目在 E 盘）
set "P=%~dp0"
set "P=%P:\=/%"
set "WSLPATH=/mnt/e%P:~2%启动演示.sh"
echo 正在启动（首次约 10 秒）...
echo.
wsl.exe -e bash "%WSLPATH%"
echo.
echo 正在打开浏览器...
start "" http://127.0.0.1:8002
echo.
echo 页面用法：
echo   1) 下拉框选视频（推荐 demo/demo_long_b，90 秒 9 段）
echo   2) 勾选"同时生成观察报告"，点"开始分析"
echo   3) 右侧面板会实时显示 Pipeline 走到哪一步
echo      结果区最上面是"识别前 / 识别后"对比图
echo.
echo 停止服务：双击 停止演示.bat
pause
