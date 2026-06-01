"""
Stage 4：抖音风智能剪辑

对每段跳舞视频进行最终剪辑：
- 竖屏裁剪（9:16，人体居中跟随）
- beat 卡点剪辑（景别切换对齐 beat）
- 变速策略（入场快进、平缓加速、高光慢放）
- 转场效果（闪白、缩放）
- FFmpeg 输出
"""

import os
import subprocess
import numpy as np
from typing import List, Dict, Tuple

from config import (
    OUTPUT_DIR, TEMP_DIR,
    OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_CRF, OUTPUT_FPS,
    SPEED_ENTRY, SPEED_FAST, SPEED_NORMAL, SPEED_SLOW,
    ENTRY_DURATION_SEC, BEATS_PER_CUT,
    FLASH_DURATION_SEC, TRANSITION_DURATION_SEC,
    SHOT_TYPES,
)
from utils.video_utils import get_video_info, extract_frames
from utils.display_utils import ProgressTracker, log_info, log_warn, log_stage


# ============================================================
# 竖屏裁剪计算
# ============================================================

def compute_vertical_crop(video_path: str, start_sec: float, end_sec: float,
                          is_split_screen: bool = False,
                          middle_crop: Tuple[int, int, int, int] = None) -> dict:
    """
    计算竖屏裁剪参数

    Returns:
        {"crop_x": int, "crop_y": int, "crop_w": int, "crop_h": int}
    """
    info = get_video_info(video_path)
    src_w, src_h = info["width"], info["height"]

    if is_split_screen and middle_crop:
        # 切屏场景：使用检测到的中间屏坐标
        mx, my, mw, mh = middle_crop
        # 在中间屏区域内裁出 9:16
        target_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT  # 0.5625
        if mw / mh > target_ratio:
            # 中间屏太宽，以高度为准
            crop_h = mh
            crop_w = int(crop_h * target_ratio)
            crop_x = mx + (mw - crop_w) // 2
            crop_y = my
        else:
            # 中间屏太高，以宽度为准
            crop_w = mw
            crop_h = int(crop_w / target_ratio)
            crop_x = mx
            crop_y = my + (mh - crop_h) // 2
        return {"crop_x": crop_x, "crop_y": crop_y, "crop_w": crop_w, "crop_h": crop_h}

    # 普通场景：以画面中心偏上为裁剪中心（主播通常在画面中央偏上）
    target_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT

    if src_w / src_h > target_ratio:
        # 源视频太宽，以高度为准
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
        crop_x = (src_w - crop_w) // 2
        crop_y = 0
    else:
        # 源视频太高，以宽度为准
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
        crop_x = 0
        crop_y = int(src_h * 0.1)  # 偏上裁剪

    return {"crop_x": crop_x, "crop_y": crop_y, "crop_w": crop_w, "crop_h": crop_h}


# ============================================================
# 剪辑时间轴构建
# ============================================================

def build_edit_timeline(segment: dict) -> List[dict]:
    """
    构建剪辑时间轴：将一段跳舞视频分割为多个子片段，
    每个子片段有自己的速度、景别、转场效果

    Returns:
        [
            {
                "src_start": float,      # 原视频中的起始时间
                "src_end": float,        # 原视频中的结束时间
                "speed": float,          # 播放速度
                "shot_type": str,        # 景别: full/medium/close
                "flash": bool,           # 是否有闪白
                "transition": str or None,  # 转场类型
            },
            ...
        ]
    """
    rhythm = segment.get("rhythm", {})
    beat_times = rhythm.get("beat_times", [])  # 段内相对时间
    beat_drops = set(rhythm.get("beat_drops", []))
    highlights = rhythm.get("highlight_ranges", [])
    start_sec = segment["start_sec"]
    end_sec = segment["end_sec"]
    duration = end_sec - start_sec

    if not beat_times or len(beat_times) < 2:
        # 没有 beat 信息，简单处理
        return [{
            "src_start": start_sec,
            "src_end": end_sec,
            "speed": SPEED_NORMAL,
            "shot_type": "medium",
            "flash": False,
            "transition": None,
        }]

    # 确定景别切换点（每 N 个 beat 切一次）
    shot_types_list = list(SHOT_TYPES.keys())
    cut_points = [beat_times[i] for i in range(0, len(beat_times), BEATS_PER_CUT)]
    if cut_points[-1] < duration - 0.5:
        cut_points.append(duration)

    # 确定高光区间集合
    highlight_set = set()
    for hs, he in highlights:
        for t in np.arange(hs, he, 0.05):
            highlight_set.add(round(t, 2))

    # 构建子片段
    timeline = []
    shot_idx = 0

    for i in range(len(cut_points) - 1):
        seg_start = cut_points[i]
        seg_end = cut_points[i + 1]

        # 决定速度
        seg_mid = (seg_start + seg_end) / 2
        is_highlight = round(seg_mid, 2) in highlight_set
        is_entry = seg_start < ENTRY_DURATION_SEC

        if is_entry:
            speed = SPEED_ENTRY
        elif is_highlight:
            speed = SPEED_SLOW
        else:
            speed = SPEED_FAST

        # 决定景别
        shot_type = shot_types_list[shot_idx % len(shot_types_list)]

        # 判断是否有 beat drop → 闪白
        has_flash = any(
            abs(bt - seg_start) < 0.3
            for bt in beat_drops
        )

        timeline.append({
            "src_start": round(start_sec + seg_start, 3),
            "src_end": round(start_sec + seg_end, 3),
            "speed": speed,
            "shot_type": shot_type,
            "flash": has_flash,
            "transition": "flash" if has_flash else None,
        })

        shot_idx += 1

    return timeline


# ============================================================
# FFmpeg 命令生成
# ============================================================

def generate_edit_command(video_path: str, output_path: str,
                          timeline: List[dict],
                          crop_params: dict) -> List[str]:
    """
    生成 FFmpeg 命令来执行剪辑

    策略：为每个子片段生成独立的滤镜，最后拼接
    """
    if not timeline:
        return []

    # 如果只有一个子片段且无特殊效果，简化处理
    if len(timeline) == 1:
        seg = timeline[0]
        return _simple_segment_cmd(
            video_path, output_path, seg, crop_params
        )

    # 多子片段：分别处理后拼接
    temp_clips = []
    for i, seg in enumerate(timeline):
        clip_path = os.path.join(TEMP_DIR, f"clip_{i:03d}.mp4")
        cmd = _simple_segment_cmd(video_path, clip_path, seg, crop_params)
        if cmd:
            temp_clips.append(clip_path)
            subprocess.run(cmd, check=True, capture_output=True)

    if not temp_clips:
        return []

    # 拼接
    from utils.video_utils import concat_videos
    concat_videos(temp_clips, output_path)

    # 清理临时文件
    for p in temp_clips:
        if os.path.exists(p):
            os.remove(p)

    return []  # 已直接执行


def _simple_segment_cmd(video_path: str, output_path: str,
                        seg: dict, crop_params: dict) -> List[str]:
    """为单个子片段生成 FFmpeg 命令"""
    cx, cy, cw, ch = (
        crop_params["crop_x"], crop_params["crop_y"],
        crop_params["crop_w"], crop_params["crop_h"]
    )

    src_start = seg["src_start"]
    src_end = seg["src_end"]
    duration = src_end - src_start
    speed = seg["speed"]
    shot_type = seg["shot_type"]

    # 景别缩放
    shot_scale = SHOT_TYPES.get(shot_type, 1.0)
    if shot_scale < 1.0:
        # 裁剪中心区域实现景别变化
        new_cw = int(cw * shot_scale)
        new_ch = int(ch * shot_scale)
        cx = cx + (cw - new_cw) // 2
        cy = cy + (ch - new_ch) // 2
        cw, ch = new_cw, new_ch

    # 视频滤镜链
    vf_parts = [
        f"crop={cw}:{ch}:{cx}:{cy}",
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}",
    ]

    # 变速
    if abs(speed - 1.0) > 0.01:
        vf_parts.append(f"setpts={1.0/speed}*PTS")

    # 闪白
    if seg.get("flash"):
        vf_parts.append(
            f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.8:t=fill:enable='between(t,0,{FLASH_DURATION_SEC})'"
        )

    vf = ",".join(vf_parts)

    # 音频变速
    af = _build_atempo_chain(speed) if abs(speed - 1.0) > 0.01 else None

    cmd = [
        "ffmpeg", "-v", "quiet", "-y",
        "-ss", str(src_start),
        "-t", str(duration),
        "-i", video_path,
        "-vf", vf,
    ]
    if af:
        cmd += ["-af", af]

    cmd += [
        "-c:v", "libx264", "-crf", str(OUTPUT_CRF),
        "-r", str(OUTPUT_FPS),
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]

    return cmd


def _build_atempo_chain(speed: float) -> str:
    """构建 atempo 链（atempo 只支持 0.5-2.0）"""
    parts = []
    remaining = speed
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.4f}")
    return ",".join(parts)


# ============================================================
# Stage 4 主流程
# ============================================================

def stage4_edit(video_path: str, segments: List[Dict],
                output_dir: str = None) -> List[str]:
    """
    Stage 4 主入口：对每段跳舞视频进行抖音风剪辑输出

    Args:
        video_path: 原始视频路径
        segments: Stage 3 输出的带节奏信息的段落列表
        output_dir: 输出目录

    Returns:
        输出文件路径列表
    """
    log_stage(4, "抖音风智能剪辑 — 卡点 + 变速 + 转场 + 竖屏")

    if not segments:
        log_info("无跳舞段，跳过 Stage 4")
        return []

    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    output_paths = []
    progress = ProgressTracker(len(segments), "Stage 4 剪辑输出")

    for idx, seg in enumerate(segments):
        output_path = os.path.join(output_dir, f"dance_{idx + 1:03d}.mp4")

        # 计算裁剪参数
        crop_params = compute_vertical_crop(
            video_path,
            seg["start_sec"], seg["end_sec"],
            seg.get("is_split_screen", False),
            seg.get("middle_screen_crop")
        )

        # 构建剪辑时间轴
        timeline = build_edit_timeline(seg)

        log_info(
            f"  段 {idx+1}: {len(timeline)} 个子片段, "
            f"BPM={seg.get('rhythm', {}).get('bpm', '?')}, "
            f"输出 → {os.path.basename(output_path)}"
        )

        # 生成并执行 FFmpeg 命令
        try:
            generate_edit_command(video_path, output_path, timeline, crop_params)
            if os.path.exists(output_path):
                output_paths.append(output_path)
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                log_info(f"  段 {idx+1}: 输出完成 ({file_size:.1f} MB)")
            else:
                log_warn(f"  段 {idx+1}: 输出文件未生成")
        except subprocess.CalledProcessError as e:
            log_warn(f"  段 {idx+1}: FFmpeg 执行失败 — {e}")

        progress.update()

    log_info(f"Stage 4 完成，共输出 {len(output_paths)} 个短视频")
    return output_paths


if __name__ == "__main__":
    import sys
    import json as json_mod

    if len(sys.argv) < 3:
        print("用法: python stage4_edit.py <视频路径> <segments_json>")
        sys.exit(1)

    video_path = sys.argv[1]
    seg_arg = sys.argv[2]

    try:
        with open(seg_arg, "r") as f:
            segments = json_mod.load(f)
    except (FileNotFoundError, json_mod.JSONDecodeError):
        segments = json_mod.loads(seg_arg)

    outputs = stage4_edit(video_path, segments)
    print(f"输出文件: {outputs}")
