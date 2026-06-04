#!/bin/bash
# Linux/macOS 启动脚本
# 用法: chmod +x run_web.sh && ./run_web.sh

echo "🎬 AI 视频剪辑工具 - Web 界面"
echo "================================"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python 3，请确保 Python 3 已安装"
    exit 1
fi

echo "启动 Flask 服务器..."
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务器"
echo ""

# 启动服务器
python3 run_web.py
