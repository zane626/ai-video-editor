# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

AI-powered video editor that detects dance segments from long livestream recordings (~4 hours), cuts them into independent short videos, and outputs TikTok/Douyin-style vertical videos (9:16, 1080x1920). Chinese-authored; comments and documentation are in Chinese.

**Core design principle:** High precision over recall — better to miss a dance segment than wrongly include a non-dance segment.

## Commands

```bash
# Install dependencies (Windows)
./install.ps1

# Install dependencies (Linux/macOS)
./install.sh

# Run the full pipeline
python pipeline.py <video_path> [output_dir]

# Run individual stages
python stage1_scan.py <video_path>
python stage2_detect.py <video_path>
python stage3_analyze.py <video_path>
python stage4_edit.py <video_path>

# Fine-tune VideoMAE dance classifier
python train_videomae.py \
  --positive-dir <dance_clip_dir> \
  --negative-video <long_video_path> \
  --output-dir models/videomae-dance \
  --epochs 10 --batch-size 4
```

**Fine-tuning:** `train_videomae.py` trains a binary dance classifier on your own data. Positive samples are short dance clips in a directory; negative samples are randomly extracted from the long video. The fine-tuned model is saved to `models/videomae-dance/` and automatically loaded by Stage 2 (falls back to the pretrained Kinetics-400 model if not found).

No test suite exists. Verification is manual — run on a short test video with known dance segments.

## Architecture

**4-stage linear pipeline** (`pipeline.py` orchestrates):

1. **Stage 1 (`stage1_scan.py`)** — CLIP coarse filtering. Extracts frames at 1fps, scores against dance/non-dance prompts, merges high-scoring frames into candidate segments. Releases CLIP model after use to free GPU memory.

2. **Stage 2 (`stage2_detect.py`)** — Precise detection. MediaPipe Pose for body keypoints, InsightFace (buffalo_l) for face ID (split-screen detection), VideoMAE for dance action classification. Refines segment boundaries based on motion curves.

3. **Stage 3 (`stage3_analyze.py`)** — Rhythm analysis. Librosa for BPM detection, beat tracking, onset energy curves. Identifies highlight ranges and beat drops. CPU-only stage.

4. **Stage 4 (`stage4_edit.py`)** — TikTok-style editing. Vertical crop, shot type cycling (full/medium/close), speed strategy (2x entry, 1.5x normal, 0.5x highlights), flash on beat drops. Outputs via FFmpeg subprocess calls (not the ffmpeg-python library).

**Model loading strategy:** Sequential per-stage, released after use to manage GPU memory (~4-5GB peak in Stage 2).

## Key Files

- `config.py` — All thresholds and output parameters. This is the primary tuning point.
- `utils/video_utils.py` — FFmpeg frame extraction, cropping, concatenation
- `utils/audio_utils.py` — Audio extraction, rhythm analysis (librosa)
- `utils/display_utils.py` — Progress bars and logging (simple print wrappers)
- `plan.md` — Detailed implementation plan (Chinese)

## Dependencies

Python 3.10+, PyTorch with CUDA 12.4, FFmpeg/ffprobe on PATH. Key libraries: transformers (CLIP, VideoMAE), mediapipe, insightface, librosa, opencv-python.

## Notable Quirks

- `ffmpeg-python` is installed but unused; FFmpeg is called directly via `subprocess.run`
- No checkpoint/resume — pipeline must restart from beginning on failure
- Threshold values in `config.py` are more lenient than what `plan.md` specifies (e.g., CLIP threshold 0.60 vs planned 0.85)
