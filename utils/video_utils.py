"""
视频处理工具函数 — 基于 FFmpeg
"""

import subprocess
import json
import os
import tempfile
import numpy as np
import cv2
from typing import List, Tuple, Optional

from config import TEMP_DIR


def get_video_info(video_path: str) -> dict:
    """获取视频基本信息：时长、分辨率、帧率等"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    info = json.loads(result.stdout.decode("utf-8", errors="replace"))

    video_stream = None
    audio_stream = None
    for stream in info.get("streams", []):
        if stream["codec_type"] == "video" and video_stream is None:
            video_stream = stream
        elif stream["codec_type"] == "audio" and audio_stream is None:
            audio_stream = stream

    duration = float(info["format"]["duration"])
    width = int(video_stream["width"]) if video_stream else 0
    height = int(video_stream["height"]) if video_stream else 0

    fps_str = video_stream.get("r_frame_rate", "30/1") if video_stream else "30/1"
    num, den = map(int, fps_str.split("/"))
    fps = num / den if den else 30

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "has_audio": audio_stream is not None,
    }


def extract_frames(video_path: str, fps: float = 1.0,
                   start_sec: float = 0, end_sec: float = None,
                   max_frames: int = None,
                   show_progress: bool = False) -> Tuple[List[np.ndarray], List[float]]:
    """
    从视频中按指定 fps 抽帧（使用 OpenCV，支持进度显示）

    Returns:
        frames: BGR 格式的 numpy array 列表
        timestamps: 每帧对应的时间戳（秒）
    """
    info = get_video_info(video_path)
    if end_sec is None:
        end_sec = info["duration"]

    duration = end_sec - start_sec
    if max_frames and duration * fps > max_frames:
        fps = max_frames / duration

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = video_fps / fps  # 每隔多少原始帧取一帧

    # 跳到起始位置
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    frames = []
    timestamps = []
    frame_count = 0
    next_sample_frame = 0
    end_frame = int(end_sec * video_fps)

    if show_progress:
        from utils.display_utils import ProgressTracker
        expected_frames = int(duration * fps)
        progress = ProgressTracker(expected_frames, desc="抽帧")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame > end_frame:
            break

        # 按间隔采样
        if current_frame >= next_sample_frame:
            frames.append(frame)
            timestamps.append(start_sec + len(frames) / fps - 1 / fps)
            next_sample_frame += frame_interval
            if show_progress:
                progress.update(1)

        frame_count += 1

    cap.release()

    if show_progress:
        progress.finish()

    return frames, timestamps


def extract_audio(video_path: str, output_path: str = None,
                  start_sec: float = 0, end_sec: float = None,
                  sample_rate: int = 22050) -> str:
    """提取视频音频轨为 WAV 文件"""
    if output_path is None:
        output_path = os.path.join(TEMP_DIR, "audio_temp.wav")

    cmd = ["ffmpeg", "-v", "quiet", "-y"]
    if start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    if end_sec is not None:
        cmd += ["-t", str(end_sec - start_sec)]
    cmd += [
        "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        output_path
    ]

    subprocess.run(cmd, check=True)
    return output_path


def crop_video(input_path: str, output_path: str,
               x: int, y: int, w: int, h: int,
               start_sec: float = 0, end_sec: float = None,
               target_width: int = 1080, target_height: int = 1920):
    """裁剪视频指定区域并缩放到目标尺寸"""
    cmd = ["ffmpeg", "-v", "quiet", "-y"]
    if start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    if end_sec is not None:
        cmd += ["-t", str(end_sec - start_sec)]
    cmd += [
        "-i", input_path,
        "-vf", f"crop={w}:{h}:{x}:{y},scale={target_width}:{target_height}",
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd, check=True)


def apply_speed_filter(input_path: str, output_path: str,
                       video_speed: float = 1.0, audio_speed: float = 1.0):
    """应用变速滤镜"""
    vf = f"setpts={1.0/video_speed}*PTS"

    # atempo 只支持 0.5-2.0，超出需要链式调用
    af_parts = []
    remaining = audio_speed
    while remaining > 2.0:
        af_parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        af_parts.append("atempo=0.5")
        remaining /= 0.5
    af_parts.append(f"atempo={remaining:.4f}")
    af = ",".join(af_parts)

    cmd = [
        "ffmpeg", "-v", "quiet", "-y",
        "-i", input_path,
        "-vf", vf, "-af", af,
        "-c:v", "libx264", "-crf", "18",
        output_path
    ]
    subprocess.run(cmd, check=True)


def add_flash_transition(input_path: str, output_path: str,
                         flash_times: List[float], flash_duration: float = 0.1):
    """在指定时间点插入闪白效果"""
    if not flash_times:
        os.rename(input_path, output_path)
        return

    # 构建 FFmpeg 滤镜：在指定时间点叠加白色
    enable_conditions = []
    for t in flash_times:
        enable_conditions.append(
            f"between(t,{t:.3f},{t + flash_duration:.3f})"
        )
    enable_expr = "+".join(enable_conditions)

    vf = f"drawbox=x=0:y=0:w=iw:h=ih:color=white@1.0:t=fill:enable='{enable_expr}'"

    cmd = [
        "ffmpeg", "-v", "quiet", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)


def concat_videos(video_paths: List[str], output_path: str):
    """拼接多个视频片段"""
    list_file = os.path.join(TEMP_DIR, "concat_list.txt")
    with open(list_file, "w") as f:
        for path in video_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    cmd = [
        "ffmpeg", "-v", "quiet", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
    os.remove(list_file)
