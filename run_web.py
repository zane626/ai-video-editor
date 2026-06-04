#!/usr/bin/env python
"""
启动 AI 剪辑工具的 Web 界面
"""

import sys
import os
import webbrowser
import time
from threading import Thread

# 添加项目目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

def open_browser():
    """延迟打开浏览器"""
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("=" * 60)
    print("🎬 AI 视频剪辑工具 - Web 界面")
    print("=" * 60)
    print("\n启动 Flask 服务器...")
    print("访问地址: http://localhost:5000")
    print("\n按 Ctrl+C 停止服务器\n")

    # 在后台打开浏览器
    browser_thread = Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # 启动 Flask 应用
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
        sys.exit(0)
