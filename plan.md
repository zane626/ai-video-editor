# AI 剪辑工具 — 实现计划

## Context

用户需要一个本地部署的 AI 视频剪辑工具，用于从 4 小时左右的女主播直播回放中自动检测跳舞片段，切割为独立短视频，并以抖音风格（快节奏卡点、变速、转场）输出适合短视频平台的内容。

**硬件环境**：i7-14700KF / 64GB RAM / RTX 5070 12GB VRAM

**核心设计决策**：
- 宁可漏切不能切错（高精度优先）
- 多人同框（不同人）不切；同人切屏保留中间屏
- 单视频处理，不做批量
- 抖音快节奏卡点风格输出

---

## 项目结构

```
ai-video-editor/
├── install.sh              # 一键环境安装
├── requirements.txt        # Python 依赖
├── config.py               # 全局配置（阈值、输出参数等）
├── pipeline.py             # 主流程入口，串联各 Stage
├── stage1_scan.py          # 抽帧 + CLIP 粗筛
├── stage2_detect.py        # 姿态检测 + 人数判断 + 切屏识别 + VideoMAE 确认
├── stage3_analyze.py       # 音乐节奏分析（BPM / beat / 能量）
├── stage4_edit.py          # 抖音风智能剪辑 + FFmpeg 输出
└── utils/
    ├── video_utils.py      # FFmpeg 抽帧、裁剪、合并等工具函数
    ├── audio_utils.py      # 音频提取、格式转换
    └── display_utils.py    # 进度条、日志输出
```

---

## Stage 1：抽帧 + CLIP 粗筛

**文件**：`stage1_scan.py`

**输入**：直播回放视频路径（4 小时，~1080p）

**处理流程**：
1. FFmpeg 按 1fps 抽帧，输出为临时图片序列（或直接读取为 numpy array）
2. 每帧送入 CLIP (ViT-B/32)，与 dance 相关 prompt 计算相似度
3. Prompt 设计：
   - 正样本：`"a woman dancing"`, `"a person dancing energetically"`, `"dance performance"`
   - 负样本：`"a person sitting"`, `"a person talking"`, `"a static scene"`
   - 用正样本最高分 - 负样本最高分作为最终得分
4. 连续得分 > 0.85 的帧合并为候选段
5. 每段前后各扩展 5 秒缓冲
6. 合并间隔 < 10 秒的相邻段

**输出**：候选时间段列表 `[{"start_sec": 1425.0, "end_sec": 1542.0}, ...]`

**模型**：
- `openai/clip-vit-base-patch32`（~350MB，显存 ~1.5GB）
- 通过 HuggingFace Transformers 加载

**性能预估**：14400 帧 × ~7ms/帧 ≈ 1.5 分钟

---

## Stage 2：精确定位 + 人数过滤 + 动作确认

**文件**：`stage2_detect.py`

**输入**：Stage 1 的候选时间段 + 原始视频

**处理流程**（对每个候选段）：

### 2a. 抽取段内帧（2fps）
- FFmpeg 从候选段中按 2fps 抽帧

### 2b. MediaPipe Pose 人体检测
- 每帧检测人体关键点
- 统计人体数量：
  - `person_count == 1` → 通过
  - `person_count > 1` → 进入切屏判断（2c）
  - `person_count == 0` → 标记为无效帧

### 2c. 切屏判断（InsightFace）
- 对检测到的每张人脸提取 embedding
- 两两计算余弦相似度
- 所有人脸相似度 > 0.7 → 判定为同人切屏 → 通过，记录为 split_screen
- 存在相似度 < 0.7 → 不同人 → 丢弃该候选段

### 2d. 运动量分析
- 计算相邻帧关键点位移总和
- 运动量曲线的均值 > 阈值 → 确认在跳舞（非静止）

### 2e. VideoMAE 动作分类确认
- 从候选段中均匀采样 16 帧
- 送入 VideoMAE (Kinetics-400)
- 检查 `dancing ballet / breakdancing / belly dancing / tap dancing` 等类别得分之和 > 0.8 → 确认跳舞
- 否则丢弃

### 2f. 边界精确化
- 基于运动量曲线，找到高运动量区间的精确起止帧
- 向外扩展 1 秒作为安全边距

**输出**：确认的跳舞片段列表，包含：
```python
{
    "start_sec": float,
    "end_sec": float,
    "is_split_screen": bool,
    "middle_screen_crop": (x, y, w, h) or None,  # 切屏时的中间屏坐标
    "confidence": float
}
```

**模型**：
- MediaPipe Pose（内置，CPU/GPU <1GB）
- InsightFace buffalo_l（~300MB，显存 ~1GB）
- VideoMAE videomae-base（~350MB，显存 ~2-3GB）

---

## Stage 3：音乐节奏分析

**文件**：`stage3_analyze.py`

**输入**：确认的跳舞片段 + 原始视频

**处理流程**：
1. FFmpeg 提取每段的音频轨（WAV）
2. librosa 分析：
   - BPM（节拍速度）
   - beat 时间点列表（精确到毫秒）
   - 能量曲线（onset strength）
   - 高潮段落检测（能量高于均值 1.5 倍的连续区间）
3. 基于能量曲线标记每段内的"高光"区间

**输出**：每段的节奏信息
```python
{
    "bpm": 128.0,
    "beat_times": [0.47, 0.94, 1.41, ...],  # 秒
    "energy_curve": np.array,                 # 逐帧能量
    "highlight_ranges": [(3.2, 5.8), (12.1, 15.0)],  # 高光区间
}
```

**依赖**：librosa（纯 CPU，无模型）

---

## Stage 4：抖音风智能剪辑

**文件**：`stage4_edit.py`

**输入**：跳舞片段信息 + 节奏分析结果

**处理流程**：

### 4a. 竖屏裁剪参数
- 普通单人段：以人体 bounding box 中心裁剪 9:16
- 切屏段：裁剪中间屏区域
- 人体移动时平滑跟随（低通滤波，避免画面抖动）

### 4b. 变速策略（对齐 beat）
| 段落类型 | 速度 | 触发条件 |
|----------|------|----------|
| 入场 | 1.5x-2x | 前 1-2 秒 |
| 平缓段 | 1.2x-1.5x | 能量低于均值 |
| 高光段 | 0.5x-0.75x | 能量高于 1.5 倍均值 |
| 转折 | 1.0x | beat drop 位置 |

速度切换点强制对齐最近的 beat 位置。

### 4c. 卡点剪辑
- 画面切换点 = beat 位置
- 每 2-4 个 beat 切换一次景别（全景/中景/特写，通过裁剪模拟）
- beat drop 处插入闪白帧

### 4d. 转场效果
- beat drop 位置：闪白（0.1s 白帧）
- 段落衔接：溶解（xfade transition）
- 高光慢放前：缩放冲击（zoom in → 慢放）

### 4e. FFmpeg 生成
- 将所有效果编码为 FFmpeg filter_complex 命令
- 输出：H.264, CRF 18, 9:16 竖屏, 1080x1920
- 音频：原声 + atempo 变速

**输出**：每段跳舞一个独立 MP4 文件，15-30 秒

---

## config.py 全局配置

```python
# 粗筛阈值
CLIP_THRESHOLD = 0.85        # CLIP 相似度阈值（越高越严格）
CLIP_FPS = 1                 # 粗筛抽帧率
BUFFER_SEC = 5               # 候选段前后缓冲秒数
MERGE_GAP_SEC = 10           # 相邻段合并间隔

# 精检阈值
POSE_FPS = 2                 # 精检抽帧率
FACE_SIMILARITY_THRESHOLD = 0.7   # 人脸相似度（切屏判断）
DANCE_SCORE_THRESHOLD = 0.8       # VideoMAE 动作确认阈值
MOTION_THRESHOLD = 50             # 运动量阈值

# 输出参数
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_CRF = 18
OUTPUT_FPS = 30

# 变速参数
SPEED_FAST = 1.5             # 平缓段速度
SPEED_SLOW = 0.5             # 高光段速度
SPEED_ENTRY = 2.0            # 入场速度
ENTRY_DURATION_SEC = 1.5     # 入场快进时长

# 转场参数
FLASH_DURATION_SEC = 0.1     # 闪白时长
TRANSITION_DURATION_SEC = 0.3 # 溶解时长
BEATS_PER_CUT = 3            # 每 N 拍切一次镜头
```

---

## pipeline.py 主流程

```python
def run(video_path, output_dir):
    # Stage 1
    candidates = stage1_scan(video_path)

    # Stage 2
    confirmed = stage2_detect(video_path, candidates)

    # Stage 3
    analyzed = stage3_analyze(video_path, confirmed)

    # Stage 4
    for i, segment in enumerate(analyzed):
        stage4_edit(video_path, segment, output_dir, index=i)

    print(f"完成，共输出 {len(analyzed)} 个短视频")
```

---

## 模型加载策略

- 不同时加载所有模型，按 Stage 顺序加载 → 推理 → 释放
- Stage 1 结束后释放 CLIP
- Stage 2 中 MediaPipe 常驻，InsightFace 和 VideoMAE 按需加载
- Stage 3 无需 GPU 模型
- Stage 4 无需 GPU 模型（FFmpeg 纯 CPU/GPU 编解码）

**峰值显存**：Stage 2 同时运行 InsightFace + VideoMAE ≈ 4-5GB，12GB 足够

---

## 验证方案

1. **单元验证**：准备一段 5 分钟的测试视频（含 2-3 段跳舞），跑完整 pipeline
2. **准确率验证**：人工标注跳舞段起止时间，与 AI 检测结果对比
3. **切屏验证**：准备含切屏的片段，确认中间屏裁剪正确
4. **输出验证**：检查输出短视频的竖屏比例、卡点精度、变速流畅度
5. **性能验证**：4 小时视频端到端处理时间（预期 30-60 分钟）

---

## 实现顺序

1. `config.py` + `utils/` — 基础配置和工具函数
2. `stage1_scan.py` — CLIP 粗筛（先跑通，验证检测效果）
3. `stage2_detect.py` — 精确定位（最关键模块）
4. `stage3_analyze.py` — 节奏分析
5. `stage4_edit.py` — 智能剪辑输出
6. `pipeline.py` — 串联全流程
7. `install.sh` + `requirements.txt` — 部署脚本

---

## 模型部署清单

| 模型 | 用途 | 下载方式 | 大小 |
|------|------|----------|------|
| `openai/clip-vit-base-patch32` | CLIP 粗筛 | HuggingFace auto | ~350MB |
| MediaPipe Pose | 姿态估计 | pip install mediapipe 内置 | ~30MB |
| `buffalo_l` (InsightFace) | 人脸识别 | 首次运行自动下载 | ~300MB |
| `MCG-NJU/videomae-base-finetuned-kinetics` | 动作分类 | HuggingFace auto | ~350MB |
| librosa | 音乐分析 | pip install（纯算法） | 很小 |

**总下载量**：~1GB
**峰值显存**：~4-5GB（Stage 2）
