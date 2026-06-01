#!/bin/bash
# AI 剪辑工具 — 一键环境安装
# 用法: bash install.sh

set -e

echo "=========================================="
echo "  AI 剪辑工具 — 环境安装"
echo "=========================================="

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "[ERROR] 未找到 conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 创建 conda 环境
ENV_NAME="ai-clip"
echo ""
echo "[1/4] 创建 conda 环境: $ENV_NAME (Python 3.10)"
conda create -n $ENV_NAME python=3.10 -y

# 激活环境
echo ""
echo "[2/4] 激活环境"
source activate $ENV_NAME || conda activate $ENV_NAME

# 安装 PyTorch (CUDA 12.x)
echo ""
echo "[3/4] 安装 PyTorch (CUDA 12.4)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
echo ""
echo "[4/4] 安装其他依赖"
pip install transformers mediapipe insightface onnxruntime-gpu
pip install librosa soundfile opencv-python numpy Pillow ffmpeg-python

# 验证安装
echo ""
echo "=========================================="
echo "  验证安装"
echo "=========================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

from transformers import CLIPModel
import mediapipe
import librosa
print('所有依赖验证通过!')
"

# 预下载模型（可选）
echo ""
read -p "是否预下载 AI 模型？(y/n, 约 1GB): " download_models
if [ "$download_models" = "y" ] || [ "$download_models" = "Y" ]; then
    echo "下载 CLIP 模型..."
    python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

    echo "下载 VideoMAE 模型..."
    python -c "from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor; VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics'); VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')"

    echo "下载 InsightFace 模型..."
    python -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640))"

    echo "所有模型下载完成！"
fi

echo ""
echo "=========================================="
echo "  安装完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  conda activate ai-clip"
echo "  python pipeline.py <视频路径> [输出目录]"
echo ""
