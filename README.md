# AI Video Editor

AI 驱动的直播回放剪辑工具，自动从长时间直播录像（~4小时）中检测舞蹈片段，切割为独立短视频，输出抖音/TikTok 风格竖屏视频（9:16, 1080x1920）。

**核心设计原则：** 宁可漏检，不可误检 —— 高精度优先于高召回。

## 工作流程

```
直播回放 → Stage 1: CLIP 粗筛 → Stage 2: 精确定位 → Stage 3: 节奏分析 → Stage 4: 智能剪辑 → 短视频输出
```

### 四阶段流水线

| 阶段 | 文件 | 功能 |
|------|------|------|
| Stage 1 | `stage1_scan.py` | CLIP 模型粗筛，1fps 抽帧，合并高分帧为候选段 |
| Stage 2 | `stage2_detect.py` | MediaPipe Pose 关键点 + InsightFace 人脸 ID + VideoMAE 动作分类，精确定位舞蹈段 |
| Stage 3 | `stage3_analyze.py` | Librosa 节奏分析，BPM 检测，节拍追踪，识别高光段和 beat drop |
| Stage 4 | `stage4_edit.py` | 竖屏裁剪，景别循环，变速策略，卡点闪白，FFmpeg 输出 |

### 模型说明

- **CLIP** (`openai/clip-vit-base-patch32`) — Stage 1 粗筛
- **MediaPipe Pose** — Stage 2 人体关键点检测
- **InsightFace** (`buffalo_l`) — Stage 2 人脸 ID（切屏检测）
- **VideoMAE** (`MCG-NJU/videomae-base-finetuned-kinetics`) — Stage 2 舞蹈动作分类，支持微调加载自定义模型

Stage 2 峰值显存约 4-5GB，各阶段模型按需加载、用后释放。

## 安装

**前置要求：** Python 3.10+，CUDA 12.4，FFmpeg/ffprobe（需在 PATH 中）

```bash
# Windows
./install.ps1

# Linux/macOS
./install.sh
```

或手动安装：

```bash
pip install -r requirements.txt
```

## 使用方法

### 完整流水线

```bash
python pipeline.py <视频路径> [输出目录]

# 示例
python pipeline.py D:/直播回放/recording.mp4
python pipeline.py D:/直播回放/recording.mp4 ./output
```

### 单独运行各阶段

```bash
python stage1_scan.py <视频路径>
python stage2_detect.py <视频路径>
python stage3_analyze.py <视频路径>
python stage4_edit.py <视频路径>
```

### 微调 VideoMAE 舞蹈分类器

```bash
python train_videomae.py \
  --positive-dir <舞蹈片段目录> \
  --negative-video <长视频路径> \
  --output-dir models/videomae-dance \
  --epochs 10 --batch-size 4
```

- 正样本：目录中的短视频舞蹈片段
- 负样本：从长视频中随机抽取的非舞蹈片段
- 微调后模型保存到 `models/videomae-dance/`，Stage 2 会优先加载（无则回退到预训练模型）

## 配置

所有阈值和输出参数集中在 `config.py`，主要可调项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CLIP_THRESHOLD` | 0.60 | CLIP 相似度阈值，越高越严格 |
| `DANCE_SCORE_THRESHOLD` | 0.10 | VideoMAE 舞蹈得分阈值 |
| `MOTION_THRESHOLD` | 15 | 运动量阈值（关键点位移） |
| `OUTPUT_CRF` | 18 | 输出视频质量（越低越清晰） |
| `SPEED_ENTRY` | 2.0 | 入场快进倍速 |
| `SPEED_SLOW` | 0.5 | 高光慢放倍速 |
| `BEATS_PER_CUT` | 3 | 每 N 拍切一次镜头 |

## 输出

- 输出目录默认为 `output/`
- 中间结果以 JSON 格式保存（`stage1_candidates.json`、`stage2_confirmed.json`、`stage3_analyzed.json`）
- 每个检测到的舞蹈段生成一个独立的竖屏短视频

## 项目结构

```
ai-video-editor/
├── pipeline.py          # 主流程入口
├── stage1_scan.py       # Stage 1: CLIP 粗筛
├── stage2_detect.py     # Stage 2: 精确定位
├── stage3_analyze.py    # Stage 3: 节奏分析
├── stage4_edit.py       # Stage 4: 智能剪辑
├── config.py            # 全局配置
├── train_videomae.py    # VideoMAE 微调脚本
├── requirements.txt     # Python 依赖
├── install.ps1          # Windows 安装脚本
├── install.sh           # Linux/macOS 安装脚本
├── utils/
│   ├── video_utils.py   # FFmpeg 帧提取、裁剪、拼接
│   ├── audio_utils.py   # 音频提取、节奏分析
│   └── display_utils.py # 进度显示和日志
├── models/              # 微调模型存放目录
├── temp/                # 临时文件
└── output/              # 输出目录
```

## 已知限制

- 无断点续跑，流水线失败需从头开始
- FFmpeg 通过 `subprocess.run` 直接调用（虽安装了 `ffmpeg-python` 但未使用）
- `config.py` 中的阈值比 `plan.md` 中的设计值更宽松（如 CLIP 阈值 0.60 vs 设计值 0.85）

## License

MIT
