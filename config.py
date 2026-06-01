"""
AI 剪辑工具 — 全局配置
"""

import os

# ============================================================
# 路径配置
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(PROJECT_DIR, "temp")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Stage 1: CLIP 粗筛
# ============================================================
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_THRESHOLD = 0.60           # 相似度阈值（sigmoid 映射后），越高越严格
CLIP_FPS = 1                    # 粗筛抽帧率（帧/秒）
BUFFER_SEC = 5                  # 候选段前后缓冲秒数
MERGE_GAP_SEC = 10              # 相邻段间隔 < 此值则合并

CLIP_POSITIVE_PROMPTS = [
    "a woman dancing",
    "a person dancing energetically",
    "a dance performance",
    "a person dancing to music",
]

CLIP_NEGATIVE_PROMPTS = [
    "a person sitting",
    "a person talking",
    "a static scene with no movement",
    "a person standing still",
    "an empty room",
]

# ============================================================
# Stage 2: 精确定位
# ============================================================
POSE_FPS = 2                    # 精检抽帧率
PERSON_COUNT_SINGLE = 1         # 单人判定

# InsightFace 切屏判断
FACE_SIMILARITY_THRESHOLD = 0.7 # 人脸 embedding 相似度阈值

# VideoMAE 动作确认
VIDEOMAE_MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
DANCE_SCORE_THRESHOLD = 0.10    # 动作分类中 dancing 类别总分阈值（VideoMAE 对舞蹈识别偏保守）
VIDEOMAE_SAMPLE_FRAMES = 16     # 采样帧数

# 运动量
MOTION_THRESHOLD = 15           # 相邻帧关键点位移阈值（归一化坐标，33关键点）
MOTION_SMOOTH_WINDOW = 5        # 运动量曲线平滑窗口

# 边界扩展
BOUNDARY_PADDING_SEC = 1        # 精确边界向外扩展秒数

# Kinetics-400 中跳舞相关类别关键词
DANCE_CLASS_KEYWORDS = [
    "dancing", "dance", "ballet", "breakdanc",
    "belly dancing", "tap dancing", "salsa",
    "swing dancing", "ballroom", "tango",
]

# ============================================================
# Stage 3: 节奏分析
# ============================================================
HIGHLIGHT_ENERGY_RATIO = 1.5   # 能量高于均值 N 倍为高光
AUDIO_SAMPLE_RATE = 22050       # librosa 采样率

# ============================================================
# Stage 4: 智能剪辑输出
# ============================================================
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_CRF = 18
OUTPUT_FPS = 30

# 变速参数
SPEED_ENTRY = 2.0               # 入场快进倍速
SPEED_FAST = 1.5                # 平缓段倍速
SPEED_NORMAL = 1.0              # 正常速度
SPEED_SLOW = 0.5                # 高光慢放倍速
ENTRY_DURATION_SEC = 1.5        # 入场快进时长

# 卡点参数
BEATS_PER_CUT = 3               # 每 N 拍切一次景别
FLASH_DURATION_SEC = 0.1        # 闪白时长
TRANSITION_DURATION_SEC = 0.3   # 转场溶解时长

# 景别裁剪比例（相对于人体 bbox）
SHOT_TYPES = {
    "full": 1.0,                # 全景
    "medium": 0.7,              # 中景
    "close": 0.4,               # 特写
}
