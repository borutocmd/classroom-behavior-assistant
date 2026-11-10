# -*- coding: utf-8 -*-
"""脚本自查：语法编译 + 命令行入口 + 关键文件完整性。

运行前跑一遍，避免换台机器就跑不起来。用法：
    python tests/check_scripts.py
"""
import compileall
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PY = sys.executable

CLI_SCRIPTS = [
    "src/prepare_frames.py", "src/make_splits.py", "src/check_dataset.py",
    "src/inspect_dataset_source.py", "src/import_yolo_dataset.py", "src/check_raw.py",
    "src/baseline_hog_svm.py", "src/train_cnn.py", "src/analyze_video.py",
    "src/build_index.py", "src/report_llm.py", "src/ablation.py",
    "src/compare_models.py", "src/summarize_results.py",
    "tools/fetch_backbone.py", "tools/make_demo_data.py", "tools/make_demo_video.py",
    "tools/build_demo_videos.py", "web/app.py",
]

REQUIRED_FILES = [
    "README.md", "requirements.txt", "install.sh", "run_all.sh", "config.py",
    "kb/teaching_tips.json", "web/static/index.html",
    "docs/系统设计说明.md",
    "docs/训练说明.md", "截图/系统运行截图_新版界面.png", "截图/方案截图.png",
]

ok = True

print("== 1. 语法编译 ==")
if compileall.compile_dir(str(PROJECT / "src"), quiet=2) and \
   compileall.compile_dir(str(PROJECT / "tools"), quiet=2) and \
   compileall.compile_dir(str(PROJECT / "tests"), quiet=2):
    print("  [通过] src/ tools/ tests/ 全部编译通过")
else:
    ok = False
    print("  [失败] 有文件编译不通过")

print("== 2. 命令行入口（--help 能跑） ==")
for rel in CLI_SCRIPTS:
    path = PROJECT / rel
    if not path.exists():
        ok = False
        print(f"  [缺失] {rel}")
        continue
    result = subprocess.run([PY, str(path), "--help"], capture_output=True, text=True,
                            cwd=str(PROJECT), timeout=120,
                            env={"PYTHONPATH": str(PROJECT), "PATH": "/usr/bin:/bin",
                                 "HOME": str(Path.home())})
    if result.returncode == 0:
        print(f"  [通过] {rel}")
    else:
        ok = False
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        print(f"  [失败] {rel} → {' / '.join(tail)}")

print("== 3. 关键文件 ==")
for rel in REQUIRED_FILES:
    if (PROJECT / rel).exists():
        print(f"  [通过] {rel}")
    else:
        ok = False
        print(f"  [缺失] {rel}")

print()
print("自查结果：" + ("全部通过 ✅" if ok else "有失败项 ❌"))
sys.exit(0 if ok else 1)
