"""
VideoMAE 微调脚本 — 用自定义数据训练舞蹈二分类器

用法:
    python train_videomae.py  --positive-dir F:\剪映\6月3日 --negative-video F:\剪映\[肥肥陈不肥]-直播回放-[2026-05-03_11_22_15].flv  --output-dir models/videomae-dance  --epochs 10 --batch-size 4
"""

import argparse
import gc
import os
import random
import sys
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config import VIDEOMAE_MODEL_NAME, VIDEOMAE_SAMPLE_FRAMES
from utils.video_utils import get_video_info, extract_frames


# ============================================================
# 数据集（基于预处理缓存）
# ============================================================

class CachedDanceDataset(Dataset):
    """
    基于预处理缓存的数据集

    所有帧提取和 VideoMAE 预处理在训练前一次性完成，
    训练时直接读取缓存的 tensor，避免重复 I/O。
    """

    def __init__(self, cached_data):
        """
        Args:
            cached_data: list of (pixel_values_tensor, label)
        """
        self.data = cached_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixel_values, label = self.data[idx]
        return {"pixel_values": pixel_values}, label


def collate_fn(batch):
    """自定义 collate：堆叠 pixel_values 和 labels"""
    inputs_list, labels = zip(*batch)
    pixel_values = torch.stack([x["pixel_values"] for x in inputs_list])
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return {"pixel_values": pixel_values}, label_tensor


# ============================================================
# 负样本生成
# ============================================================

def generate_negative_segments(video_path: str, num_segments: int,
                                segment_duration: float = 10.0,
                                exclude_ranges: list = None) -> list:
    """
    从长视频中随机生成非舞蹈片段

    Args:
        video_path: 长视频路径
        num_segments: 需要生成的负样本数量
        segment_duration: 每段时长（秒）
        exclude_ranges: 排除的时间区间列表 [(start, end), ...]

    Returns:
        list of dict with "video_path", "start_sec", "end_sec", "label"
    """
    info = get_video_info(video_path)
    duration = info["duration"]

    if exclude_ranges is None:
        exclude_ranges = []

    # 合并重叠的排除区间
    exclude_ranges = sorted(exclude_ranges)
    merged = []
    for start, end in exclude_ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    exclude_ranges = merged

    # 计算可用时间段
    available_ranges = []
    prev_end = 0
    for start, end in exclude_ranges:
        if start > prev_end:
            available_ranges.append((prev_end, start))
        prev_end = max(prev_end, end)
    if prev_end < duration:
        available_ranges.append((prev_end, duration))

    # 过滤掉太短的区间
    available_ranges = [(s, e) for s, e in available_ranges if e - s >= segment_duration]

    if not available_ranges:
        print("警告: 没有可用的负样本区间，跳过负样本生成")
        return []

    segments = []
    attempts = 0
    max_attempts = num_segments * 10

    while len(segments) < num_segments and attempts < max_attempts:
        attempts += 1

        # 随机选择一个可用区间
        range_start, range_end = random.choice(available_ranges)
        max_start = range_end - segment_duration
        if max_start <= range_start:
            continue

        start_sec = random.uniform(range_start, max_start)
        end_sec = start_sec + segment_duration

        segments.append({
            "video_path": video_path,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "label": 0,
        })

    return segments


# ============================================================
# 数据预处理（一次性提取所有帧并缓存为 tensor）
# ============================================================

def preprocess_samples(samples, processor, sample_frames=VIDEOMAE_SAMPLE_FRAMES):
    """
    预处理所有样本：提取帧 -> 采样 -> VideoMAE processor -> 缓存 tensor

    训练前只调用一次，避免每个 epoch 重复 I/O。

    Args:
        samples: list of dict with "video_path", "start_sec"(opt), "end_sec"(opt), "label"
        processor: VideoMAEImageProcessor
        sample_frames: 采样帧数

    Returns:
        list of (pixel_values_tensor, label)
    """
    cached = []
    for i, sample in enumerate(samples):
        video_path = sample["video_path"]
        start_sec = sample.get("start_sec", 0)
        end_sec = sample.get("end_sec", None)
        label = sample["label"]

        filename = os.path.basename(video_path)
        time_range = f"[{start_sec:.0f}s"
        time_range += f"-{end_sec:.0f}s]" if end_sec else "]"
        print(f"\r  预处理 [{i+1}/{len(samples)}] {filename} {time_range}...",
              end="", flush=True)

        # 只抽取 VideoMAE 需要的帧数，避免长片段占满内存
        frames, _ = extract_frames(
            video_path, fps=2,
            start_sec=start_sec, end_sec=end_sec,
            max_frames=sample_frames,
        )

        if len(frames) == 0:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * sample_frames

        # 均匀采样（帧数可能略少于 sample_frames）
        n = len(frames)
        if n >= sample_frames:
            indices = np.linspace(0, n - 1, sample_frames, dtype=int)
        else:
            indices = np.arange(n)

        sampled = [frames[i] for i in indices]

        # BGR -> RGB -> PIL -> VideoMAE processor
        pil_images = [Image.fromarray(f[:, :, ::-1]) for f in sampled]
        inputs = processor(pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (T, C, H, W)

        cached.append((pixel_values, label))

        del frames, sampled, pil_images, inputs
        gc.collect()

    print()  # 换行
    return cached


# ============================================================
# 训练
# ============================================================

def train(positive_dir: str, negative_video: str, output_dir: str,
          epochs: int = 10, batch_size: int = 4, lr: float = 1e-4,
          num_negative_ratio: float = 1.0):
    """
    训练 VideoMAE 舞蹈分类器

    Args:
        positive_dir: 正样本视频目录
        negative_video: 原始长视频路径（用于生成负样本）
        output_dir: 模型保存目录
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        num_negative_ratio: 负样本数量相对于正样本的倍数
    """
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 1. 准备数据（预处理只需 processor，避免与模型争抢内存）
    # 正样本
    positive_samples = []
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"}
    for filename in sorted(os.listdir(positive_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in video_extensions:
            positive_samples.append({
                "video_path": os.path.join(positive_dir, filename),
                "label": 1,
            })

    if not positive_samples:
        print(f"错误: 在 {positive_dir} 中未找到视频文件")
        sys.exit(1)

    print(f"正样本数量: {len(positive_samples)}")

    # 负样本
    negative_samples = []
    if negative_video and os.path.exists(negative_video):
        num_neg = max(1, int(len(positive_samples) * num_negative_ratio))
        print(f"正在从长视频生成 {num_neg} 个负样本...")

        # 计算正样本的平均时长作为负样本时长
        durations = []
        for s in positive_samples:
            info = get_video_info(s["video_path"])
            durations.append(info["duration"])
        avg_duration = sum(durations) / len(durations) if durations else 10.0

        negative_samples = generate_negative_segments(
            negative_video, num_neg,
            segment_duration=avg_duration,
            exclude_ranges=[]
        )
        print(f"负样本数量: {len(negative_samples)}")

    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)

    # 划分训练集和验证集 (80/20)
    split_idx = max(1, int(len(all_samples) * 0.8))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"训练集: {len(train_samples)}, 验证集: {len(val_samples)}")

    print(f"加载 VideoMAE 处理器: {VIDEOMAE_MODEL_NAME}")
    processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL_NAME)

    # 2. 预处理缓存（一次性提取所有帧）
    print("正在预处理训练集（提取帧 + VideoMAE 编码）...")
    t0 = time.time()
    train_cached = preprocess_samples(train_samples, processor)
    print(f"  训练集预处理完成 ({time.time() - t0:.1f}s)")

    print("正在预处理验证集...")
    t0 = time.time()
    val_cached = preprocess_samples(val_samples, processor)
    print(f"  验证集预处理完成 ({time.time() - t0:.1f}s)")

    train_dataset = CachedDanceDataset(train_cached)
    val_dataset = CachedDanceDataset(val_cached)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0,
    )

    # 3. 加载模型（预处理完成后再加载，节省内存）
    print(f"加载预训练模型: {VIDEOMAE_MODEL_NAME}")
    model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_MODEL_NAME)

    # 替换分类头为二分类
    num_labels = 2
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_labels)
    model.config.num_labels = num_labels
    model.config.id2label = {0: "non-dance", 1: "dance"}
    model.config.label2id = {"non-dance": 0, "dance": 1}

    # 冻结 backbone，只训练分类头
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,} "
          f"({trainable_params/total_params*100:.2f}%)")

    model = model.to(device)

    # 4. 训练
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    print(f"\n开始训练 ({epochs} epochs)...")
    train_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            pixel_values = inputs["pixel_values"].to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            print(f"\r  Epoch {epoch+1}/{epochs} "
                  f"[{batch_idx+1}/{len(train_loader)}] "
                  f"loss={loss.item():.4f}", end="", flush=True)

        scheduler.step()
        train_acc = train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0

        # --- Val ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                pixel_values = inputs["pixel_values"].to(device)
                labels = labels.to(device)

                outputs = model(pixel_values=pixel_values)
                loss = criterion(outputs.logits, labels)

                val_loss += loss.item()
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        epoch_time = time.time() - epoch_start

        print(f"\n  Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s): "
              f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}")

        # 保存最佳模型
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print(f"  -> 模型已保存到 {output_dir} (val_acc={val_acc:.4f})")

    total_time = time.time() - train_start
    print(f"\n训练完成! 总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型保存在: {output_dir}")
    print(f"运行 pipeline 时将自动加载微调后的模型")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="VideoMAE 舞蹈分类器微调训练"
    )
    parser.add_argument(
        "--positive-dir", required=True,
        help="正样本视频目录（每个视频文件为一段舞蹈）"
    )
    parser.add_argument(
        "--negative-video", required=True,
        help="原始长视频路径（用于生成负样本）"
    )
    parser.add_argument(
        "--output-dir", default="models/videomae-dance",
        help="模型保存目录 (默认: models/videomae-dance)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="训练轮数 (默认: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="批次大小 (默认: 4)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="学习率 (默认: 1e-4)"
    )
    parser.add_argument(
        "--negative-ratio", type=float, default=1.0,
        help="负样本数量相对于正样本的倍数 (默认: 1.0)"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.positive_dir):
        print(f"错误: 正样本目录不存在: {args.positive_dir}")
        sys.exit(1)

    if not os.path.isfile(args.negative_video):
        print(f"错误: 长视频文件不存在: {args.negative_video}")
        sys.exit(1)

    train(
        positive_dir=args.positive_dir,
        negative_video=args.negative_video,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_negative_ratio=args.negative_ratio,
    )


if __name__ == "__main__":
    main()
