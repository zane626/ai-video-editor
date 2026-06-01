"""
Stage 3：音乐节奏分析

对每段跳舞视频进行音频分析：
- BPM（节拍速度）
- beat 时间点（精确到毫秒）
- 能量曲线（onset strength）
- 高光区间检测
- beat drop 检测
"""

import os
import numpy as np
from typing import List, Dict

from config import TEMP_DIR, AUDIO_SAMPLE_RATE, HIGHLIGHT_ENERGY_RATIO
from utils.audio_utils import analyze_rhythm, find_beat_drops, extract_segment_audio
from utils.display_utils import ProgressTracker, log_info, log_stage


def stage3_analyze(video_path: str,
                   segments: List[Dict]) -> List[Dict]:
    """
    Stage 3 主入口：对每段跳舞视频进行节奏分析

    Args:
        video_path: 原始视频路径
        segments: Stage 2 输出的确认跳舞段列表

    Returns:
        附加了节奏信息的段落列表
    """
    log_stage(3, "音乐节奏分析 — BPM / beat / 能量 / 高光")

    if not segments:
        log_info("无跳舞段，跳过 Stage 3")
        return []

    analyzed = []
    progress = ProgressTracker(len(segments), "Stage 3 节奏分析")

    for idx, seg in enumerate(segments):
        start_sec = seg["start_sec"]
        end_sec = seg["end_sec"]

        # 提取段内音频
        audio_path = os.path.join(TEMP_DIR, f"seg_{idx}_audio.wav")
        try:
            extract_segment_audio(video_path, start_sec, end_sec, audio_path)
        except Exception as e:
            log_info(f"  段 {idx+1}: 音频提取失败 ({e})，使用默认节奏")
            analyzed.append(_default_segment(seg))
            progress.update()
            continue

        # 分析节奏
        try:
            rhythm = analyze_rhythm(audio_path)
        except Exception as e:
            log_info(f"  段 {idx+1}: 节奏分析失败 ({e})，使用默认节奏")
            analyzed.append(_default_segment(seg))
            progress.update()
            continue

        # 检测 beat drops
        beat_drops = find_beat_drops(
            rhythm["beat_times"],
            rhythm["energy_curve"],
            rhythm["energy_times"],
            rhythm["bpm"]
        )

        # 将 beat 时间转换为绝对时间（相对于原始视频）
        abs_beat_times = [t + start_sec for t in rhythm["beat_times"]]
        abs_beat_drops = [t + start_sec for t in beat_drops]
        abs_highlights = [
            (s + start_sec, e + start_sec)
            for s, e in rhythm["highlight_ranges"]
        ]

        result = {
            **seg,
            "rhythm": {
                "bpm": round(rhythm["bpm"], 1),
                "beat_times": rhythm["beat_times"],       # 段内相对时间
                "beat_times_abs": abs_beat_times,          # 原始视频绝对时间
                "beat_drops": beat_drops,                   # 段内相对时间
                "beat_drops_abs": abs_beat_drops,           # 原始视频绝对时间
                "energy_curve": rhythm["energy_curve"].tolist(),
                "energy_times": rhythm["energy_times"],
                "highlight_ranges": rhythm["highlight_ranges"],  # 段内相对时间
                "highlight_ranges_abs": abs_highlights,    # 原始视频绝对时间
            }
        }

        log_info(
            f"  段 {idx+1}: BPM={rhythm['bpm']:.0f}, "
            f"beats={len(rhythm['beat_times'])}, "
            f"beat_drops={len(beat_drops)}, "
            f"highlights={len(rhythm['highlight_ranges'])}"
        )

        analyzed.append(result)
        progress.update()

        # 清理临时音频
        if os.path.exists(audio_path):
            os.remove(audio_path)

    log_info(f"Stage 3 完成，分析了 {len(analyzed)} 段")
    return analyzed


def _default_segment(seg: Dict) -> Dict:
    """节奏分析失败时的默认值"""
    duration = seg["end_sec"] - seg["start_sec"]
    # 假设 120 BPM
    beat_interval = 0.5
    beat_times = list(np.arange(0, duration, beat_interval))
    return {
        **seg,
        "rhythm": {
            "bpm": 120.0,
            "beat_times": beat_times,
            "beat_times_abs": [t + seg["start_sec"] for t in beat_times],
            "beat_drops": [],
            "beat_drops_abs": [],
            "energy_curve": [0.5] * int(duration * 10),
            "energy_times": list(np.arange(0, duration, 0.1)),
            "highlight_ranges": [(duration * 0.2, duration * 0.8)],
            "highlight_ranges_abs": [
                (seg["start_sec"] + duration * 0.2, seg["start_sec"] + duration * 0.8)
            ],
        }
    }


if __name__ == "__main__":
    import sys
    import json as json_mod

    if len(sys.argv) < 3:
        print("用法: python stage3_analyze.py <视频路径> <segments_json>")
        sys.exit(1)

    video_path = sys.argv[1]
    seg_arg = sys.argv[2]

    try:
        with open(seg_arg, "r") as f:
            segments = json_mod.load(f)
    except (FileNotFoundError, json_mod.JSONDecodeError):
        segments = json_mod.loads(seg_arg)

    result = stage3_analyze(video_path, segments)
    print(json_mod.dumps(result, indent=2))
