#!/bin/bash
# Windows PowerShell 启动脚本
# 用法: .\run_web.ps1

Write-Host "🎬 AI 视频剪辑工具 - Web 界面" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# 检查 Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "错误: 未找到 Python，请确保 Python 已安装" -ForegroundColor Red
    exit 1
}

Write-Host "启动 Flask 服务器..." -ForegroundColor Yellow
Write-Host "访问地址: http://localhost:5000" -ForegroundColor Green
Write-Host "按 Ctrl+C 停止服务器`n" -ForegroundColor Yellow

# 启动服务器
python run_web.py
