# 系统架构

一条从视频到报告的数据流，每一层只做一件事，层与层之间用文件/字典解耦。

```
视频
 └─[prepare_frames] 抽帧 + 自动标注
      └─[make_splits] 按来源分层划分
           └─[dataset] 预处理与增强 ──▶ [train_cnn] 训练 ──▶ [evaluate] 评测
                                                                    │
                                                                    ▼
视频 ─[analyze_video] 逐帧推理 + 时序平滑 ──▶ 时间线 / 统计 ──▶ [report_llm] 生成报告
                                                                    ▲
                                        [build_index] 教学建议 → 向量 → FAISS 索引
                                                                    │
                                                        [web/app] FastAPI + 单页前端
```

## 各模块职责

| 模块 | 输入 | 输出 | 一句话职责 |
| --- | --- | --- | --- |
| `src/prepare_frames.py` | 原始视频/图片目录 | `frames/<类别>/*.jpg` + `labels.csv` | 按固定 fps 抽帧，用文件夹名/文件名作为弱标签 |
| `src/make_splits.py` | `labels.csv` | `splits.json` | 按**来源**（同一次拍摄）分层划分，避免同场景帧跨集合造成泄漏 |
| `src/dataset.py` | `splits.json` | DataLoader | 统一 resize/归一化，训练与推理共用同一套预处理 |
| `src/train_cnn.py` | 数据集 | `models/<tag>/model.pt` | ResNet18 迁移学习；`--no-freeze` 全网络微调、`--backbone scratch` 从头训练作对照 |
| `src/evaluate.py` | 模型 + 测试集 | 指标 JSON + 混淆矩阵 | 计算准确率/宏 F1，画混淆矩阵定位类别混淆 |
| `src/analyze_video.py` | 视频 + 模型 + `SMOOTH_WINDOW` | 时间线 JSON + 统计 | 逐帧分类后做滑窗投票与短事件合并，输出行为时间线与课堂统计 |
| `src/build_index.py` | `kb/teaching_tips.json` | `kb/tips.faiss` | 把教学建议编码成向量并建索引 |
| `src/report_llm.py` | 统计 + 检索结果 | 观察报告（Markdown） | 组装 Prompt 调本地小模型；`--no-rag` 可控实验；输出经数字忠实度检查 |
| `web/app.py` | 视频路径 + 参数 | 网页 / JSON | 长任务放后台线程，`/progress` 上报九个阶段的进度 |

## 关键设计取舍

- **按来源分层划分**而不是随机划分：同一段录制里的相邻帧高度相似，随机划分会让测试集"见过"训练集，指标虚高。
- **滑窗投票 + 短事件合并是两个机制**：前者压单帧抖动，后者吃掉过短事件；只做其中一个都不够。
- **数字忠实度检查**：小模型容易在报告里编造统计，生成后把报告里的数字与统计字典对照，非统计来源的数字标出来。
