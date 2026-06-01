"""
Stage 2：精确定位 + 人数过滤 + 动作确认

对 Stage 1 的候选段进行精检：
- MediaPipe Pose 人体检测 + 人数统计
- InsightFace 人脸识别（切屏判断：同人 vs 多人）
- 运动量分析（确认在跳舞而非静止）
- VideoMAE 动作分类确认
- 边界精确化
"""

import os
import torch
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from config import (
    POSE_FPS, FACE_SIMILARITY_THRESHOLD, DANCE_SCORE_THRESHOLD,
    VIDEOMAE_MODEL_NAME, VIDEOMAE_FINETUNED_PATH, VIDEOMAE_SAMPLE_FRAMES,
    MOTION_THRESHOLD, MOTION_SMOOTH_WINDOW,
    BOUNDARY_PADDING_SEC, DANCE_CLASS_KEYWORDS,
)
from utils.video_utils import get_video_info, extract_frames
from utils.display_utils import ProgressTracker, log_info, log_warn, log_stage


# ============================================================
# MediaPipe Pose
# ============================================================

def init_pose_detector():
    """初始化 MediaPipe Pose"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose, mp_pose


def detect_pose_batch(pose_detector, mp_pose, frames: List[np.ndarray]) -> List[dict]:
    """
    批量检测人体姿态

    Returns:
        每帧的检测结果列表:
        [{"person_count": int, "keypoints": np.ndarray or None, "bboxes": list}, ...]
    """
    results = []
    for frame in frames:
        rgb = frame[:, :, ::-1]
        det = pose_detector.process(rgb)

        if det.pose_landmarks:
            # 单人模式，检测到 1 个人
            kps = np.array([
                [lm.x, lm.y, lm.visibility]
                for lm in det.pose_landmarks.landmark
            ])
            results.append({
                "person_count": 1,
                "keypoints": kps,
                "bboxes": [_keypoints_to_bbox(kps, frame.shape[1], frame.shape[0])],
            })
        else:
            results.append({
                "person_count": 0,
                "keypoints": None,
                "bboxes": [],
            })

    return results


def _keypoints_to_bbox(kps: np.ndarray, width: int, height: int,
                       padding: float = 0.1) -> Tuple[int, int, int, int]:
    """从关键点推算 bounding box (x, y, w, h)"""
    # 只取可见的关键点
    visible = kps[kps[:, 2] > 0.3]
    if len(visible) == 0:
        return (0, 0, width, height)

    x_min = visible[:, 0].min() * width
    x_max = visible[:, 0].max() * width
    y_min = visible[:, 1].min() * height
    y_max = visible[:, 1].max() * height

    # 添加 padding
    w = x_max - x_min
    h = y_max - y_min
    pad_w = w * padding
    pad_h = h * padding

    x = int(max(0, x_min - pad_w))
    y = int(max(0, y_min - pad_h))
    w = int(min(width - x, w + 2 * pad_w))
    h = int(min(height - y, h + 2 * pad_h))

    return (x, y, w, h)


# ============================================================
# InsightFace 切屏判断
# ============================================================

def init_face_analyzer():
    """初始化 InsightFace"""
    try:
        import insightface
        app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except ImportError:
        log_warn("InsightFace 未安装，跳过切屏检测（多人段将全部丢弃）")
        return None


def is_same_person(face_analyzer, frame: np.ndarray,
                   threshold: float = FACE_SIMILARITY_THRESHOLD) -> bool:
    """
    判断画面中的多个人是否为同一个人（切屏场景）

    Returns:
        True = 同一人切屏（保留）
        False = 不同人（丢弃）
    """
    if face_analyzer is None:
        return False

    faces = face_analyzer.get(frame)
    if len(faces) <= 1:
        return True  # 只检测到 0 或 1 张脸，可能是切屏但脸小

    # 提取 embedding 并两两比较
    embeddings = [f.embedding for f in faces]
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            if sim < threshold:
                return False  # 不同的人
    return True  # 所有脸都是同一个人


# ============================================================
# 运动量分析
# ============================================================

def compute_motion_curve(pose_results: List[dict]) -> np.ndarray:
    """
    计算相邻帧关键点位移总和的运动量曲线

    Returns:
        motion: np.ndarray, shape (N-1,)
    """
    motion = []
    for i in range(1, len(pose_results)):
        kps_prev = pose_results[i - 1]["keypoints"]
        kps_curr = pose_results[i]["keypoints"]

        if kps_prev is None or kps_curr is None:
            motion.append(0.0)
            continue

        # 只取 x, y（忽略 visibility）
        diff = kps_curr[:, :2] - kps_prev[:, :2]
        displacement = np.sqrt((diff ** 2).sum(axis=1)).sum()
        motion.append(displacement)

    return np.array(motion)


def smooth_curve(data: np.ndarray, window: int = MOTION_SMOOTH_WINDOW) -> np.ndarray:
    """简单滑动平均平滑"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")


def find_motion_boundaries(motion: np.ndarray, threshold: float,
                           fps: float) -> Tuple[Optional[float], Optional[float]]:
    """
    基于运动量找到跳舞的精确起止时间

    Returns:
        (start_sec, end_sec) or (None, None) if no active region
    """
    if len(motion) == 0:
        return None, None

    # 用原始运动量判断是否有高运动区域（避免平滑削峰）
    if motion.max() < threshold:
        return None, None

    # 用平滑曲线找边界（更稳定）
    smoothed = smooth_curve(motion)
    above = smoothed > threshold * 0.5  # 平滑后用更低的阈值找边界

    if not above.any():
        return None, None

    # 找第一个和最后一个超过阈值的位置
    indices = np.where(above)[0]
    start_idx = indices[0]
    end_idx = indices[-1]

    start_sec = start_idx / fps
    end_sec = (end_idx + 1) / fps  # +1 因为 motion 比 frames 少一帧

    return start_sec, end_sec


# ============================================================
# VideoMAE 动作分类
# ============================================================

def load_videomae_model():
    """加载 VideoMAE 模型（优先加载微调版本）"""
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    # 优先加载微调后的模型
    if os.path.isdir(VIDEOMAE_FINETUNED_PATH):
        model_path = VIDEOMAE_FINETUNED_PATH
        log_info(f"加载微调 VideoMAE 模型: {model_path}")
    else:
        model_path = VIDEOMAE_MODEL_NAME
        log_info(f"加载预训练 VideoMAE 模型: {model_path}")

    model = VideoMAEForVideoClassification.from_pretrained(model_path)
    processor = VideoMAEImageProcessor.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # 判断是否为微调后的二分类模型
    is_finetuned = model.config.num_labels == 2
    log_info(f"VideoMAE 已加载到 {device} ({'二分类微调' if is_finetuned else 'Kinetics-400 多分类'})")
    return model, processor, device, is_finetuned


@torch.no_grad()
def classify_dance(model, processor, device: str,
                   frames: List[np.ndarray],
                   is_finetuned: bool = False) -> float:
    """
    用 VideoMAE 对视频片段进行动作分类

    Returns:
        dance_score: dancing 相关类别的总得分 (0-1)
    """
    from PIL import Image

    # 均匀采样 VIDEOMAE_SAMPLE_FRAMES 帧
    n = len(frames)
    if n >= VIDEOMAE_SAMPLE_FRAMES:
        indices = np.linspace(0, n - 1, VIDEOMAE_SAMPLE_FRAMES, dtype=int)
    else:
        indices = np.arange(n)

    sampled = [frames[i] for i in indices]
    pil_images = [Image.fromarray(f[:, :, ::-1]) for f in sampled]

    inputs = processor(pil_images, return_tensors="pt").to(device)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1).cpu().numpy()[0]

    if is_finetuned:
        # 微调后的二分类模型：直接取 dance 类别的概率
        dance_score = float(probs[1])
    else:
        # 原始 Kinetics-400 模型：汇总 dance 相关类别得分
        dance_score = 0.0
        for idx, label in model.config.id2label.items():
            label_lower = label.lower()
            if any(kw in label_lower for kw in DANCE_CLASS_KEYWORDS):
                dance_score += probs[idx]

    return dance_score


# ============================================================
# 切屏场景裁剪
# ============================================================

def detect_split_screen_layout(frame: np.ndarray, person_count: int) -> Optional[Tuple[int, int, int, int]]:
    """
    检测切屏布局并返回中间屏的裁剪坐标

    Returns:
        (x, y, w, h) 中间屏坐标，或 None（非切屏）
    """
    if person_count <= 1:
        return None

    h, w = frame.shape[:2]

    # 常见切屏布局：左右两分屏、三屏、四宫格
    # 通过检测画面中的分割线（垂直/水平的边缘线）
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测垂直分割线（用 Sobel）
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)

    # 按列求和，找峰值 = 分割线位置
    col_sum = sobel_x.mean(axis=0)

    # 找显著峰值（排除边缘区域）
    margin = int(w * 0.05)
    threshold = col_sum[margin:-margin].mean() + 2 * col_sum[margin:-margin].std()

    peaks = []
    for x in range(margin, w - margin):
        if col_sum[x] > threshold:
            # 合并相邻峰值
            if peaks and x - peaks[-1] < w * 0.05:
                continue
            peaks.append(x)

    if len(peaks) == 0:
        return None

    # 根据分割线数量判断布局
    if len(peaks) == 1:
        # 两分屏：取右半部分（通常主播在右边）
        split_x = peaks[0]
        # 选择中间的区域
        mid_x = w // 2
        crop_w = split_x
        return (mid_x - crop_w // 2, 0, crop_w, h)

    elif len(peaks) >= 2:
        # 三屏或更多：取中间屏
        # 按 x 坐标排序
        peaks.sort()
        # 取中间两个分割线之间的区域
        if len(peaks) >= 2:
            # 取中间区域
            mid_start = peaks[len(peaks) // 2 - 1]
            mid_end = peaks[len(peaks) // 2]
            crop_x = mid_start
            crop_w = mid_end - mid_start
            return (crop_x, 0, crop_w, h)

    return None


# ============================================================
# Stage 2 主流程
# ============================================================

def stage2_detect(video_path: str,
                  candidates: List[Dict]) -> List[Dict]:
    """
    Stage 2 主入口：精确定位

    Args:
        video_path: 视频文件路径
        candidates: Stage 1 输出的候选段列表

    Returns:
        确认的跳舞片段列表
    """
    log_stage(2, "精确定位 — 姿态检测 + 人数过滤 + 动作确认")

    if not candidates:
        log_info("无候选段，跳过 Stage 2")
        return []

    info = get_video_info(video_path)

    # 初始化模型
    pose_detector, mp_pose = init_pose_detector()
    face_analyzer = init_face_analyzer()

    # VideoMAE 延迟加载（只在有候选段通过前几步时才加载）
    videomae_model = None
    videomae_processor = None
    videomae_device = None
    videomae_is_finetuned = False

    confirmed = []
    progress = ProgressTracker(len(candidates), "Stage 2 精检")

    for idx, cand in enumerate(candidates):
        start_sec = cand["start_sec"]
        end_sec = cand["end_sec"]

        # 2a. 抽帧（2fps）
        frames, timestamps = extract_frames(
            video_path, fps=POSE_FPS,
            start_sec=start_sec, end_sec=end_sec
        )

        if len(frames) < 3:
            log_info(f"  段 {idx+1}: 帧数不足 ({len(frames)}), 跳过")
            progress.update()
            continue

        # 2b. 姿态检测
        pose_results = detect_batch(pose_detector, mp_pose, frames)

        # 统计主要人数（取众数）
        person_counts = [r["person_count"] for r in pose_results]
        main_count = max(set(person_counts), key=person_counts.count)
        detected_frames = sum(1 for c in person_counts if c > 0)

        if main_count == 0:
            log_info(f"  段 {idx+1}: 未检测到人体 (0/{len(frames)} 帧), 跳过")
            progress.update()
            continue

        log_info(f"  段 {idx+1}: 检测到 {detected_frames}/{len(frames)} 帧有人, 主要人数={main_count}")

        # 2c. 多人判断
        is_split = False
        if main_count > 1:
            # 检测是否为切屏（同一个人）
            # 取几帧采样检测
            sample_indices = np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int)
            all_same = all(
                is_same_person(face_analyzer, frames[i])
                for i in sample_indices
            )
            if all_same:
                is_split = True
                log_info(f"  段 {idx+1}: 检测到同人切屏，保留中间屏")
            else:
                progress.update()
                continue

        # 2d. 运动量分析
        motion = compute_motion_curve(pose_results)
        motion_start, motion_end = find_motion_boundaries(
            motion, threshold=MOTION_THRESHOLD, fps=POSE_FPS
        )

        if motion_start is None:
            smoothed_max = smooth_curve(motion).max() if len(motion) > 0 else 0
            log_info(f"  段 {idx+1}: 运动量不足 (raw_max={motion.max():.1f}, smooth_max={smoothed_max:.1f} < {MOTION_THRESHOLD}), 跳过")
            progress.update()
            continue

        log_info(f"  段 {idx+1}: 运动量通过 (max={motion.max():.1f}, 边界 {motion_start:.1f}s-{motion_end:.1f}s)")

        # 2e. VideoMAE 动作确认（延迟加载）
        if videomae_model is None:
            videomae_model, videomae_processor, videomae_device, videomae_is_finetuned = load_videomae_model()

        dance_score = classify_dance(
            videomae_model, videomae_processor, videomae_device, frames,
            is_finetuned=videomae_is_finetuned
        )

        if dance_score < DANCE_SCORE_THRESHOLD:
            log_info(f"  段 {idx+1}: VideoMAE 得分 {dance_score:.4f} < {DANCE_SCORE_THRESHOLD}，丢弃")
            progress.update()
            continue

        # 2f. 边界精确化
        # 将运动量边界转换为绝对时间
        abs_start = start_sec + motion_start - BOUNDARY_PADDING_SEC
        abs_end = start_sec + motion_end + BOUNDARY_PADDING_SEC
        abs_start = max(0, abs_start)
        abs_end = min(info["duration"], abs_end)

        # 切屏时的中间屏裁剪坐标
        middle_crop = None
        if is_split:
            # 取运动量最高的一帧来判断布局
            peak_idx = np.argmax(motion) if len(motion) > 0 else len(frames) // 2
            peak_idx = min(peak_idx, len(frames) - 1)
            middle_crop = detect_split_screen_layout(frames[peak_idx], main_count)

        confirmed.append({
            "start_sec": round(abs_start, 2),
            "end_sec": round(abs_end, 2),
            "is_split_screen": is_split,
            "middle_screen_crop": middle_crop,
            "confidence": round(float(dance_score), 4),
            "motion_amplitude": smooth_curve(motion).tolist() if len(motion) > 0 else [],
        })

        log_info(
            f"  段 {idx+1}: 确认跳舞 [{_sec_to_time(abs_start)} - {_sec_to_time(abs_end)}] "
            f"score={dance_score:.4f} {'(切屏)' if is_split else ''}"
        )
        progress.update()

    # 释放模型
    pose_detector.close()
    if videomae_model is not None:
        del videomae_model, videomae_processor
        if videomae_device == "cuda":
            torch.cuda.empty_cache()

    log_info(f"Stage 2 完成，确认 {len(confirmed)}/{len(candidates)} 个跳舞段")
    return confirmed


def detect_batch(pose_detector, mp_pose, frames):
    """批量姿态检测的别名"""
    return detect_pose_batch(pose_detector, mp_pose, frames)


def _sec_to_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    import sys
    import json as json_mod

    if len(sys.argv) < 3:
        print("用法: python stage2_detect.py <视频路径> <candidates_json>")
        print("  candidates_json: Stage 1 输出的 JSON 字符串或文件路径")
        sys.exit(1)

    video_path = sys.argv[1]
    cand_arg = sys.argv[2]

    # 尝试作为文件路径读取，否则作为 JSON 字符串
    try:
        with open(cand_arg, "r") as f:
            candidates = json_mod.load(f)
    except (FileNotFoundError, json_mod.JSONDecodeError):
        candidates = json_mod.loads(cand_arg)

    result = stage2_detect(video_path, candidates)
    print(json_mod.dumps(result, indent=2))
