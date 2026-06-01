# AI Video Editor - Environment Setup (Windows PowerShell)
# Usage: Right-click -> Run with PowerShell, or in PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\install.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  AI Video Editor - Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Hardware: i7-14700KF / 64GB RAM / RTX 5070 12GB VRAM" -ForegroundColor DarkGray
Write-Host ""

# Check Python
$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.") {
            $pythonCmd = $cmd
            Write-Host "Found: $ver (via '$cmd')" -ForegroundColor DarkGray
            break
        }
    }
}

if (-not $pythonCmd) {
    Write-Host "[ERROR] Python 3 not found. Please install Python 3.10+ first." -ForegroundColor Red
    Write-Host "Download: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create venv
$VENV_DIR = ".venv"
Write-Host "[1/5] Creating virtual environment: $VENV_DIR" -ForegroundColor Green
if (Test-Path $VENV_DIR) {
    Write-Host "  Virtual environment already exists, reusing..." -ForegroundColor DarkGray
} else {
    & $pythonCmd -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Get venv python/pip paths
$venvPython = Join-Path $VENV_DIR "Scripts\python.exe"
$venvPip = Join-Path $VENV_DIR "Scripts\pip.exe"

# Upgrade pip
Write-Host ""
Write-Host "[2/5] Upgrading pip" -ForegroundColor Green
& $venvPython -m pip install --upgrade pip --quiet

# Install PyTorch (CUDA 12.x for RTX 5070)
Write-Host ""
Write-Host "[3/5] Installing PyTorch (CUDA 12.4 for RTX 5070)" -ForegroundColor Green
& $venvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install PyTorch" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install ML dependencies
Write-Host ""
Write-Host "[4/5] Installing ML dependencies" -ForegroundColor Green
& $venvPip install transformers mediapipe insightface onnxruntime-gpu
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install ML dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install audio/video dependencies
Write-Host ""
Write-Host "[5/5] Installing audio/video dependencies" -ForegroundColor Green
& $venvPip install librosa soundfile opencv-python numpy Pillow ffmpeg-python
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install audio/video dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Verify installation
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Verifying Installation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$verifyScript = @"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
from transformers import CLIPModel
import mediapipe
import librosa
print('All dependencies verified!')
"@

& $venvPython -c $verifyScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Verification had errors, please check manually" -ForegroundColor Yellow
}

# Pre-download models
Write-Host ""
$download = Read-Host "Pre-download AI models? (y/n, ~1GB)"
if ($download -eq "y" -or $download -eq "Y") {
    Write-Host "Downloading CLIP model..." -ForegroundColor Green
    & $venvPython -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

    Write-Host "Downloading VideoMAE model..." -ForegroundColor Green
    & $venvPython -c "from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor; VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics'); VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')"

    Write-Host "Downloading InsightFace model..." -ForegroundColor Green
    & $venvPython -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640))"

    Write-Host "All models downloaded!" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Usage:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python pipeline.py video.mp4 output/"
Write-Host ""
Write-Host "Or without activating:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\python.exe pipeline.py video.mp4 output/"
Write-Host ""
Read-Host "Press Enter to exit"
