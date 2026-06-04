# 🎬 AI 视频剪辑工具 - Web 前端使用指南

## 快速开始

### 1. 安装依赖

```bash
# 安装 Flask 和其他依赖
pip install -r requirements.txt
```

### 2. 启动 Web 服务

**Windows:**
```powershell
.\run_web.ps1
```

**Linux/macOS:**
```bash
chmod +x run_web.sh
./run_web.sh
```

**或直接运行 Python:**
```bash
python run_web.py
```

服务器将在 `http://localhost:5000` 启动，浏览器会自动打开。

---

## 功能说明

### 📹 上传视频
- **视频路径**: 输入本地视频的完整路径（支持 Windows 和 Unix 路径格式）
- **输出目录** (可选): 指定输出短视频的保存位置，留空使用默认目录

### ⚙️ 配置参数
展开"配置参数"部分可以查看 `config.py` 中的所有参数设置。主要参数包括：

| 参数 | 用途 | 默认值 |
|------|------|--------|
| `CLIP_THRESHOLD` | CLIP 模型相似度阈值 | 0.60 |
| `VIDEOMAE_SAMPLE_FRAMES` | VideoMAE 采样帧数 | 16 |
| `OUTPUT_WIDTH` / `OUTPUT_HEIGHT` | 输出视频分辨率 | 1080 × 1920 |
| `SPEED_ENTRY` / `SPEED_FAST` / `SPEED_SLOW` | 变速倍数 | 2.0 / 1.5 / 0.5 |

### 📂 输出文件
- 列出所有生成的短视频文件
- 显示文件大小和修改时间
- 支持直接下载文件

### 📋 处理历史
记录所有提交的处理任务，包括视频路径和提交时间。

---

## 处理流程

Web 界面调用的完整 pipeline 包含 4 个阶段：

```
直播回放
    ↓
Stage 1: CLIP 粗筛 (提取候选段)
    ↓
Stage 2: 精确定位 (验证舞蹈动作)
    ↓
Stage 3: 节奏分析 (提取音乐信息)
    ↓
Stage 4: 智能剪辑 (生成短视频)
    ↓
输出 9:16 竖屏短视频 (1080×1920)
```

### 进度显示
- 实时显示处理进度 (0-100%)
- 显示当前阶段的状态信息
- 完成后列出所有生成的输出文件

---

## API 端点

Web 应用提供以下 REST API 端点（供前端调用）：

### GET `/api/status`
获取系统状态和输出文件列表

**响应:**
```json
{
  "output_dir": "/path/to/output",
  "files": [...],
  "job_count": 0
}
```

### POST `/api/submit`
提交新的处理任务

**请求:**
```json
{
  "video_path": "/path/to/video.mp4",
  "output_dir": "/path/to/output" // 可选
}
```

**响应:**
```json
{
  "job_id": "job_1234567890",
  "message": "任务已提交"
}
```

### GET `/api/job/<job_id>`
获取任务状态

**响应:**
```json
{
  "job_id": "job_1234567890",
  "status": "running", // pending, running, completed, failed
  "message": "处理中...",
  "progress": 45,
  "output_files": [...],
  "error": null
}
```

### GET `/api/output-files`
列出所有输出文件

**响应:**
```json
{
  "files": [
    {
      "name": "output_1.mp4",
      "size": 52428800,
      "size_mb": "50.00",
      "modified": "2024-01-01T12:00:00",
      "downloadable": true
    }
  ],
  "total": 1
}
```

### GET `/api/download/<filename>`
下载输出文件

### GET `/api/config`
获取配置文件内容

**响应:**
```json
{
  "content": "# 配置文件内容..."
}
```

---

## 技术架构

### 后端
- **框架**: Flask 2.3+
- **并发**: 使用 threading 在后台运行处理任务
- **任务管理**: 内存中的任务队列和状态字典

### 前端
- **HTML/CSS/JS**: 传统 Web 应用，无框架依赖
- **实时更新**: 每 2 秒轮询一次任务状态
- **响应式设计**: 支持桌面和移动设备

### 文件结构
```
ai-video-editor/
├── app.py                 # Flask 应用主文件
├── run_web.py            # Python 启动脚本
├── run_web.sh            # Linux/macOS 启动脚本
├── run_web.ps1           # Windows PowerShell 启动脚本
├── templates/
│   └── index.html        # 主 HTML 页面
└── static/
    ├── css/
    │   └── style.css     # 样式表
    └── js/
        └── app.js        # 前端逻辑
```

---

## 常见问题

### Q: 如何改变输出视频的分辨率？
A: 在"配置参数"中修改 `OUTPUT_WIDTH` 和 `OUTPUT_HEIGHT`，然后重新启动服务器。

### Q: 处理失败了怎么办？
A: 检查错误信息。常见原因包括：
- 视频路径不存在或格式不支持
- GPU 内存不足
- 输出目录无写入权限

### Q: 能同时处理多个视频吗？
A: 可以提交多个任务，但它们会按顺序执行（串行处理）。若要并行处理，需要修改 `app.py` 中的任务队列逻辑。

### Q: 如何访问远程服务器上的 Web 界面？
A: Flask 应用监听所有网卡 (`0.0.0.0:5000`)，在另一台机器上访问 `http://<服务器IP>:5000` 即可。

---

## 开发和调试

### 启用调试模式
编辑 `app.py` 的最后一行：
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

### 查看日志
Flask 和处理脚本的日志会输出到控制台。

### 修改轮询间隔
编辑 `static/js/app.js` 的 `startPolling()` 函数中的间隔时间（默认 2000ms）。

---

## 安全考虑

- **文件访问**: API 会检查文件路径是否在输出目录内
- **输入验证**: 视频路径会验证文件是否存在
- **错误隐藏**: 生产环境建议关闭 Flask 调试模式

---

## 许可证

遵循主项目的许可证。
