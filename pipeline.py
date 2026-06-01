"""
AI 剪辑工具 — 主流程入口

串联 Stage 1-4 的完整流水线：
  直播回放 → CLIP 粗筛 → 精确定位 → 节奏分析 → 抖音风剪辑 → 短视频输出
"""

import os
import sys
import time
import json

from config import OUTPUT_DIR
from stage1_scan import stage1_scan
from stage2_detect import stage2_detect
from stage3_analyze import stage3_analyze
from stage4_edit import stage4_edit
from utils.display_utils import log_info, log_error


def run(video_path: str, output_dir: str = None):
    """
    执行完整的 AI 剪辑流水线

    Args:
        video_path: 输入视频路径
        output_dir: 输出目录（默认使用 config 中的 OUTPUT_DIR）
    """
    if not os.path.exists(video_path):
        log_error(f"视频文件不存在: {video_path}")
        return

    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()
    log_info(f"开始处理: {video_path}")
    log_info(f"输出目录: {output_dir}")

    # ── Stage 1: CLIP 粗筛 ──────────────────────────────────
    stage1_start = time.time()
    candidates = stage1_scan(video_path)
    log_info(f"Stage 1 耗时: {time.time() - stage1_start:.1f}s")

    if not candidates:
        log_info("未检测到跳舞候选段，流程结束")
        return

    # 保存中间结果
    _save_intermediate(output_dir, "stage1_candidates.json", candidates)

    # ── Stage 2: 精确定位 ──────────────────────────────────
    stage2_start = time.time()
    confirmed = stage2_detect(video_path, candidates)
    log_info(f"Stage 2 耗时: {time.time() - stage2_start:.1f}s")

    if not confirmed:
        log_info("无段落通过精检，流程结束")
        return

    _save_intermediate(output_dir, "stage2_confirmed.json", confirmed)

    # ── Stage 3: 节奏分析 ──────────────────────────────────
    stage3_start = time.time()
    analyzed = stage3_analyze(video_path, confirmed)
    log_info(f"Stage 3 耗时: {time.time() - stage3_start:.1f}s")

    if not analyzed:
        log_info("节奏分析无结果，流程结束")
        return

    # 保存完整分析结果（含节奏数据，较大）
    _save_intermediate(output_dir, "stage3_analyzed.json", analyzed, compact=True)

    # ── Stage 4: 智能剪辑 ──────────────────────────────────
    stage4_start = time.time()
    output_paths = stage4_edit(video_path, analyzed, output_dir)
    log_info(f"Stage 4 耗时: {time.time() - stage4_start:.1f}s")

    # ── 完成 ──────────────────────────────────────────────
    total_time = time.time() - total_start
    log_info(f"\n{'='*60}")
    log_info(f"全部完成！")
    log_info(f"  总耗时: {total_time/60:.1f} 分钟")
    log_info(f"  检测到: {len(candidates)} 个候选段 → {len(confirmed)} 个确认段")
    log_info(f"  输出了: {len(output_paths)} 个短视频")
    log_info(f"  输出目录: {output_dir}")
    if output_paths:
        log_info("  文件列表:")
        for p in output_paths:
            size = os.path.getsize(p) / (1024 * 1024)
            log_info(f"    - {os.path.basename(p)} ({size:.1f} MB)")
    log_info(f"{'='*60}")

    return output_paths


def _save_intermediate(output_dir: str, filename: str, data, compact: bool = False):
    """保存中间结果为 JSON"""
    filepath = os.path.join(output_dir, filename)
    try:
        # 转换不可序列化的类型
        def _default(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if hasattr(obj, '__float__'):
                return float(obj)
            return str(obj)

        with open(filepath, "w", encoding="utf-8") as f:
            if compact:
                json.dump(data, f, ensure_ascii=False, indent=2, default=_default)
            else:
                json.dump(data, f, ensure_ascii=False, indent=2, default=_default)
        log_info(f"中间结果已保存: {filename}")
    except Exception as e:
        log_error(f"保存中间结果失败: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python pipeline.py <视频路径> [输出目录]")
        print()
        print("示例:")
        print("  python pipeline.py D:/直播回放/recording.mp4")
        print("  python pipeline.py D:/直播回放/recording.mp4 ./output")
        sys.exit(1)

    video = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    run(video, out_dir)
