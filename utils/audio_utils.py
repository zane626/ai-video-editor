"""
音频处理工具函数
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple

from config import AUDIO_SAMPLE_RATE, HIGHLIGHT_ENERGY_RATIO
from utils.video_utils import extract_audio


def analyze_rhythm(audio_path: str) -> dict:
    """
    分析音频的节奏信息：BPM、beat 位置、能量曲线、高光区间

    Returns:
        {
            "bpm": float,
            "beat_times": List[float],      # beat 时间点（秒）
            "energy_curve": np.ndarray,      # 归一化能量曲线
            "highlight_ranges": List[Tuple[float, float]],  # 高光区间
        }
    """
    y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)

    # BPM 和 beat 检测
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # 确保 tempo 是标量
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        tempo = float(tempo)

    # 能量曲线（onset strength）
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # 归一化到 0-1
    if onset_env.max() > 0:
        energy_norm = onset_env / onset_env.max()
    else:
        energy_norm = onset_env

    # 高光区间：能量高于均值 * HIGHLIGHT_ENERGY_RATIO 的连续区域
    energy_mean = energy_norm.mean()
    threshold = energy_mean * HIGHLIGHT_ENERGY_RATIO
    is_highlight = energy_norm > threshold

    # 转换为时间轴
    frame_times = librosa.frames_to_time(np.arange(len(energy_norm)), sr=sr)

    # 提取连续高光区间
    highlight_ranges = []
    in_range = False
    range_start = 0.0
    for i, (high, t) in enumerate(zip(is_highlight, frame_times)):
        if high and not in_range:
            range_start = t
            in_range = True
        elif not high and in_range:
            highlight_ranges.append((range_start, t))
            in_range = False
    if in_range:
        highlight_ranges.append((range_start, frame_times[-1]))

    return {
        "bpm": tempo,
        "beat_times": beat_times,
        "energy_curve": energy_norm,
        "energy_times": frame_times.tolist(),
        "highlight_ranges": highlight_ranges,
    }


def find_beat_drops(beat_times: List[float], energy_curve: np.ndarray,
                    energy_times: List[float], bpm: float) -> List[float]:
    """
    检测 beat drop 位置（能量突然上升的 beat 点）

    Returns:
        beat_drop_times: List[float] 秒
    """
    if len(beat_times) < 2:
        return []

    beat_drops = []
    # 在每个 beat 位置附近检查能量跳变
    for i in range(1, len(beat_times)):
        t_prev = beat_times[i - 1]
        t_curr = beat_times[i]

        # 找到 energy_times 中最接近的时间索引
        idx_prev = np.argmin(np.abs(np.array(energy_times) - t_prev))
        idx_curr = np.argmin(np.abs(np.array(energy_times) - t_curr))

        if idx_curr < len(energy_curve) and idx_prev < len(energy_curve):
            energy_jump = energy_curve[idx_curr] - energy_curve[idx_prev]
            # 能量跳变超过 0.3 视为 beat drop
            if energy_jump > 0.3:
                beat_drops.append(t_curr)

    return beat_drops


def extract_segment_audio(video_path: str, start_sec: float, end_sec: float,
                          output_path: str = None) -> str:
    """提取视频片段的音频"""
    return extract_audio(
        video_path,
        output_path=output_path,
        start_sec=start_sec,
        end_sec=end_sec,
        sample_rate=AUDIO_SAMPLE_RATE
    )
