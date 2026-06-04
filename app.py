"""
Flask Web 应用 — 为 AI 剪辑工具提供前端界面
"""

import os
import json
import threading
import queue
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file
from config import OUTPUT_DIR, PROJECT_DIR
from pipeline import run

app = Flask(__name__, template_folder='templates', static_folder='static')

# 任务队列：用于跟踪处理进度
job_queue = queue.Queue()
job_status = {}


class ProcessingJob:
    """处理任务对象"""
    def __init__(self, job_id, video_path, output_dir=None):
        self.job_id = job_id
        self.video_path = video_path
        self.output_dir = output_dir or OUTPUT_DIR
        self.status = "pending"  # pending, running, completed, failed
        self.message = ""
        self.progress = 0  # 0-100
        self.start_time = None
        self.end_time = None
        self.output_files = []
        self.error = None


def process_job(job):
    """在后台线程中处理任务"""
    try:
        job.status = "running"
        job.start_time = datetime.now().isoformat()
        job.progress = 5
        job.message = "正在初始化..."
        job_status[job.job_id] = job

        # Stage 1: CLIP 粗筛
        job.message = "Stage 1: CLIP 粗筛检测舞蹈候选段..."
        job.progress = 10
        job_status[job.job_id] = job

        # Stage 2: 精确定位
        job.message = "Stage 2: 精确定位确认舞蹈动作..."
        job.progress = 35
        job_status[job.job_id] = job

        # Stage 3: 节奏分析
        job.message = "Stage 3: 节奏分析提取音乐信息..."
        job.progress = 60
        job_status[job.job_id] = job

        # 运行 pipeline
        job.message = "Stage 4: 智能剪辑生成短视频..."
        job.progress = 85
        output_paths = run(job.video_path, job.output_dir)

        if output_paths:
            job.output_files = output_paths
            job.status = "completed"
            job.progress = 100
            job.message = f"✓ 完成！输出了 {len(output_paths)} 个短视频"
        else:
            job.status = "completed"
            job.progress = 100
            job.message = "✓ 处理完成，但未生成输出"

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"✗ 错误: {str(e)}"
    finally:
        job.end_time = datetime.now().isoformat()
        job_status[job.job_id] = job


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """获取系统状态和输出文件列表"""
    try:
        # 列出输出目录的文件
        output_files = []
        if os.path.exists(OUTPUT_DIR):
            for filename in os.listdir(OUTPUT_DIR):
                filepath = os.path.join(OUTPUT_DIR, filename)
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath) / (1024 * 1024)
                    output_files.append({
                        'name': filename,
                        'size': f"{size:.2f} MB",
                        'path': filepath,
                        'type': 'video' if filename.endswith(('.mp4', '.mov', '.avi')) else 'other'
                    })

        return jsonify({
            'output_dir': OUTPUT_DIR,
            'files': output_files,
            'job_count': len(job_status)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """获取单个任务的状态"""
    job = job_status.get(job_id)
    if not job:
        return jsonify({'error': '任务不存在'}), 404

    return jsonify({
        'job_id': job.job_id,
        'status': job.status,
        'message': job.message,
        'progress': job.progress,
        'start_time': job.start_time,
        'end_time': job.end_time,
        'output_files': job.output_files,
        'error': job.error
    })


@app.route('/api/submit', methods=['POST'])
def submit_job():
    """提交处理任务"""
    try:
        data = request.json
        video_path = data.get('video_path', '').strip()
        output_dir = data.get('output_dir', '').strip() or OUTPUT_DIR

        # 验证视频文件
        if not video_path:
            return jsonify({'error': '请提供视频路径'}), 400

        video_path = os.path.abspath(video_path)
        if not os.path.exists(video_path):
            return jsonify({'error': f'视频文件不存在: {video_path}'}), 400

        # 创建任务
        job_id = f"job_{int(datetime.now().timestamp() * 1000)}"
        job = ProcessingJob(job_id, video_path, output_dir)
        job_status[job_id] = job

        # 在后台线程中运行
        thread = threading.Thread(target=process_job, args=(job,), daemon=True)
        thread.start()

        return jsonify({
            'job_id': job_id,
            'message': '任务已提交'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """下载输出文件"""
    try:
        filepath = os.path.join(OUTPUT_DIR, filename)

        # 安全检查：确保文件在输出目录内
        if not os.path.abspath(filepath).startswith(os.path.abspath(OUTPUT_DIR)):
            return jsonify({'error': '非法文件访问'}), 403

        if not os.path.exists(filepath):
            return jsonify({'error': '文件不存在'}), 404

        return send_file(filepath, as_attachment=True)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """获取或更新配置"""
    try:
        config_path = os.path.join(PROJECT_DIR, 'config.py')

        if request.method == 'GET':
            # 读取配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({'content': content})

        else:  # POST
            # 更新配置文件（需要谨慎）
            data = request.json
            new_content = data.get('content', '')

            # 简单验证：检查是否包含 Python 代码
            if not new_content or 'import' not in new_content:
                return jsonify({'error': '配置内容无效'}), 400

            # 备份原文件
            backup_path = config_path + '.backup'
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    with open(backup_path, 'w', encoding='utf-8') as bf:
                        bf.write(f.read())

            # 写入新配置
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return jsonify({'message': '配置已更新', 'backup': backup_path})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/output-files', methods=['GET'])
def list_output_files():
    """列出输出文件"""
    try:
        files = []
        if os.path.exists(OUTPUT_DIR):
            for filename in sorted(os.listdir(OUTPUT_DIR)):
                filepath = os.path.join(OUTPUT_DIR, filename)
                if os.path.isfile(filepath):
                    stat = os.stat(filepath)
                    files.append({
                        'name': filename,
                        'size': stat.st_size,
                        'size_mb': f"{stat.st_size / (1024*1024):.2f}",
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'downloadable': True
                    })

        return jsonify({'files': files, 'total': len(files)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
