"""
Stage 1：抽帧 + CLIP 粗筛

从 4 小时直播回放中快速找出"可能在跳舞"的候选时间段。
- FFmpeg 按 1fps 抽帧
- CLIP 计算每帧与 dance 相关 prompt 的相似度
- 连续高分帧合并为候选段（加缓冲、合并相邻段）
"""

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict

from config import (
    CLIP_MODEL_NAME, CLIP_THRESHOLD, CLIP_FPS,
    BUFFER_SEC, MERGE_GAP_SEC,
    CLIP_POSITIVE_PROMPTS, CLIP_NEGATIVE_PROMPTS,
)
from utils.video_utils import get_video_info, extract_frames
from utils.display_utils import ProgressTracker, log_info, log_stage


def load_clip_model():
    """加载 CLIP 模型和处理器"""
    log_info(f"加载 CLIP 模型: {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    log_info(f"CLIP 模型已加载到 {device}")
    return model, processor, device


@torch.no_grad()
def score_frames_batch(model, processor, device: str,
                       frames: List[np.ndarray],
                       batch_size: int = 32) -> np.ndarray:
    """
    批量计算帧的跳舞得分

    Returns:
        scores: np.ndarray, shape (N,), 每帧的 dance 得分 (0-1)
    """
    from utils.display_utils import ProgressTracker
    all_scores = []
    total_batches = (len(frames) + batch_size - 1) // batch_size
    progress = ProgressTracker(total_batches, desc="CLIP 推理")

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        # BGR -> RGB -> PIL
        from PIL import Image
        pil_images = [Image.fromarray(f[:, :, ::-1]) for f in batch]

        # 正样本 prompt
        inputs_pos = processor(
            text=CLIP_POSITIVE_PROMPTS,
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).to(device)
        outputs_pos = model(**inputs_pos)
        # logits_per_image: (batch, num_prompts), 原始 cosine similarity × temperature
        # 取正样本平均得分
        pos_mean = outputs_pos.logits_per_image.mean(dim=-1).cpu().numpy()

        # 负样本 prompt
        inputs_neg = processor(
            text=CLIP_NEGATIVE_PROMPTS,
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).to(device)
        outputs_neg = model(**inputs_neg)
        neg_mean = outputs_neg.logits_per_image.mean(dim=-1).cpu().numpy()

        # 最终得分 = sigmoid(正样本均值 - 负样本均值)
        # 将 logit 差值映射到 0-1
        diff = pos_mean - neg_mean
        scores = 1 / (1 + np.exp(-diff))
        all_scores.append(scores)
        progress.update(1)

    progress.finish()
    return np.concatenate(all_scores)


def merge_candidates(timestamps: List[float], scores: np.ndarray,
                     threshold: float, buffer_sec: float,
                     merge_gap_sec: float) -> List[Dict[str, float]]:
    """
    将连续高分帧合并为候选时间段

    Returns:
        [{"start_sec": float, "end_sec": float, "avg_score": float}, ...]
    """
    # 找出超过阈值的帧
    above = scores >= threshold
    if not above.any():
        return []

    # 提取连续为 True 的区间
    segments = []
    in_seg = False
    seg_start = 0
    seg_scores = []

    for i, (is_above, t) in enumerate(zip(above, timestamps)):
        if is_above and not in_seg:
            seg_start = i
            seg_scores = [scores[i]]
            in_seg = True
        elif is_above and in_seg:
            seg_scores.append(scores[i])
        elif not is_above and in_seg:
            segments.append({
                "start_idx": seg_start,
                "end_idx": i - 1,
                "start_sec": timestamps[seg_start],
                "end_sec": timestamps[i - 1],
                "avg_score": np.mean(seg_scores),
            })
            in_seg = False

    # 处理最后一段
    if in_seg:
        segments.append({
            "start_idx": seg_start,
            "end_idx": len(timestamps) - 1,
            "start_sec": timestamps[seg_start],
            "end_sec": timestamps[-1],
            "avg_score": np.mean(seg_scores),
        })

    if not segments:
        return []

    # 添加缓冲
    duration = timestamps[-1]
    for seg in segments:
        seg["start_sec"] = max(0, seg["start_sec"] - buffer_sec)
        seg["end_sec"] = min(duration, seg["end_sec"] + buffer_sec)

    # 合并相邻段
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg["start_sec"] - merged[-1]["end_sec"] <= merge_gap_sec:
            merged[-1]["end_sec"] = seg["end_sec"]
            merged[-1]["avg_score"] = max(merged[-1]["avg_score"], seg["avg_score"])
        else:
            merged.append(seg)

    return [
        {
            "start_sec": round(s["start_sec"], 2),
            "end_sec": round(s["end_sec"], 2),
            "avg_score": round(s["avg_score"], 4),
        }
        for s in merged
    ]


def stage1_scan(video_path: str) -> List[Dict[str, float]]:
    """
    Stage 1 主入口：抽帧 + CLIP 粗筛

    Args:
        video_path: 视频文件路径

    Returns:
        候选时间段列表
    """
    log_stage(1, "CLIP 粗筛 — 快速定位跳舞候选段")

    # 获取视频信息
    info = get_video_info(video_path)
    duration = info["duration"]
    log_info(f"视频时长: {duration/3600:.1f} 小时, 分辨率: {info['width']}x{info['height']}")

    # 抽帧
    log_info(f"按 {CLIP_FPS}fps 抽帧中...")
    frames, timestamps = extract_frames(video_path, fps=CLIP_FPS, show_progress=True)
    log_info(f"共抽取 {len(frames)} 帧")

    # 加载模型并计算得分
    model, processor, device = load_clip_model()
    log_info("计算 CLIP 相似度得分...")
    scores = score_frames_batch(model, processor, device, frames)
    log_info(f"得分范围: [{scores.min():.4f}, {scores.max():.4f}], 均值: {scores.mean():.4f}")

    # 释放 CLIP 模型
    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()
    log_info("CLIP 模型已释放")

    # 合并候选段
    candidates = merge_candidates(
        timestamps, scores,
        threshold=CLIP_THRESHOLD,
        buffer_sec=BUFFER_SEC,
        merge_gap_sec=MERGE_GAP_SEC
    )

    log_info(f"粗筛完成，找到 {len(candidates)} 个候选段:")
    for i, c in enumerate(candidates):
        start_str = _sec_to_time(c["start_sec"])
        end_str = _sec_to_time(c["end_sec"])
        duration_sec = c["end_sec"] - c["start_sec"]
        log_info(f"  [{i+1}] {start_str} - {end_str} ({duration_sec:.0f}s, score={c['avg_score']:.4f})")

    return candidates


def _sec_to_time(sec: float) -> str:
    """秒数转 HH:MM:SS 格式"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python stage1_scan.py <视频路径>")
        sys.exit(1)
    video_path = sys.argv[1]
    candidates = stage1_scan(video_path)
    print(f"\n候选段: {candidates}")
