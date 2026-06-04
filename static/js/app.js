// 应用主逻辑
let currentJobId = null;
let pollInterval = null;
const jobHistory = [];

// 页面加载
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    loadDefaultOutputDir();
    refreshFileList();
    loadConfig();
});

function initializeApp() {
    // 上传表单提交
    document.getElementById('uploadForm').addEventListener('submit', handleSubmit);

    // 配置下载按钮
    document.getElementById('downloadConfigBtn').addEventListener('click', downloadConfig);

    // 配置重置按钮
    document.getElementById('resetConfigBtn').addEventListener('click', resetConfig);
}

function loadDefaultOutputDir() {
    fetch('/api/status')
        .then(r => r.json())
        .then(data => {
            document.getElementById('defaultOutputDir').textContent = data.output_dir;
        });
}

function loadConfig() {
    fetch('/api/config')
        .then(r => r.json())
        .then(data => {
            document.getElementById('configEditor').value = data.content;
        })
        .catch(err => console.error('加载配置失败:', err));
}

async function handleSubmit(e) {
    e.preventDefault();

    const videoPath = document.getElementById('videoPath').value.trim();
    const outputDir = document.getElementById('outputDir').value.trim();

    if (!videoPath) {
        showError('请输入视频路径');
        return;
    }

    try {
        const response = await fetch('/api/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_path: videoPath,
                output_dir: outputDir || null
            })
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || '提交失败');
            return;
        }

        currentJobId = data.job_id;
        showJobStatus(videoPath);
        startPolling();

        // 添加到历史记录
        jobHistory.unshift({
            job_id: currentJobId,
            video_path: videoPath,
            submit_time: new Date().toLocaleString('zh-CN')
        });
        updateJobHistory();

    } catch (err) {
        showError(`提交错误: ${err.message}`);
    }
}

function showJobStatus(videoPath) {
    const jobStatus = document.getElementById('jobStatus');
    jobStatus.classList.remove('hidden');
    document.getElementById('statusMessage').textContent = '初始化中...';
    document.getElementById('progressBar').style.width = '5%';
    document.getElementById('progressText').textContent = '5%';
    document.getElementById('errorBox').classList.add('hidden');
}

function closeJobStatus() {
    document.getElementById('jobStatus').classList.add('hidden');
    if (pollInterval) clearInterval(pollInterval);
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);

    // 立即检查一次
    checkJobStatus();

    // 每 2 秒检查一次
    pollInterval = setInterval(checkJobStatus, 2000);
}

async function checkJobStatus() {
    if (!currentJobId) return;

    try {
        const response = await fetch(`/api/job/${currentJobId}`);
        const job = await response.json();

        // 更新进度条
        document.getElementById('progressBar').style.width = job.progress + '%';
        document.getElementById('progressText').textContent = job.progress + '%';

        // 更新状态消息
        document.getElementById('statusMessage').textContent = job.message;

        // 更新详细信息
        if (job.start_time) {
            const details = `
                <p><strong>开始时间:</strong> ${new Date(job.start_time).toLocaleString('zh-CN')}</p>
                ${job.end_time ? `<p><strong>结束时间:</strong> ${new Date(job.end_time).toLocaleString('zh-CN')}</p>` : ''}
            `;
            document.getElementById('statusDetails').innerHTML = details;
        }

        // 处理完成
        if (job.status === 'completed' || job.status === 'failed') {
            clearInterval(pollInterval);

            if (job.status === 'completed') {
                document.getElementById('progressBar').style.width = '100%';
                document.getElementById('progressText').textContent = '100%';

                if (job.output_files && job.output_files.length > 0) {
                    const fileList = job.output_files
                        .map(f => `<p>✓ ${f}</p>`)
                        .join('');
                    document.getElementById('statusDetails').innerHTML += `
                        <div style="margin-top: 1rem; padding: 1rem; background-color: #ecfdf5; border-radius: 4px;">
                            <strong>输出文件:</strong>
                            ${fileList}
                        </div>
                    `;
                }

                // 刷新文件列表
                setTimeout(refreshFileList, 500);
            } else if (job.status === 'failed') {
                showJobError(job.error);
            }
        }

    } catch (err) {
        console.error('检查状态失败:', err);
    }
}

function showJobError(error) {
    const errorBox = document.getElementById('errorBox');
    errorBox.innerHTML = `<strong>错误:</strong> ${error}`;
    errorBox.classList.remove('hidden');
}

function showError(message) {
    alert(message);
}

async function refreshFileList() {
    try {
        const response = await fetch('/api/output-files');
        const data = await response.json();

        const fileList = document.getElementById('fileList');

        if (data.files.length === 0) {
            fileList.innerHTML = '<p class="empty">暂无输出文件</p>';
            return;
        }

        fileList.innerHTML = data.files.map(file => `
            <div class="file-item">
                <div class="file-info">
                    <div class="file-name">🎬 ${escapeHtml(file.name)}</div>
                    <div class="file-meta">${file.size_mb} MB · ${new Date(file.modified).toLocaleString('zh-CN')}</div>
                </div>
                <div class="file-actions">
                    <button class="btn btn-primary btn-small" onclick="downloadFile('${escapeHtml(file.name)}')">
                        ⬇ 下载
                    </button>
                </div>
            </div>
        `).join('');

    } catch (err) {
        console.error('刷新文件列表失败:', err);
        document.getElementById('fileList').innerHTML = `<p class="error">加载失败: ${err.message}</p>`;
    }
}

function downloadFile(filename) {
    const url = `/api/download/${encodeURIComponent(filename)}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function downloadConfig() {
    const content = document.getElementById('configEditor').value;
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'config.py';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function resetConfig() {
    if (confirm('确定要重置配置吗？这将刷新为服务器的当前配置。')) {
        loadConfig();
    }
}

function updateJobHistory() {
    const historyDiv = document.getElementById('jobHistory');

    if (jobHistory.length === 0) {
        historyDiv.innerHTML = '<p class="empty">暂无处理记录</p>';
        return;
    }

    historyDiv.innerHTML = jobHistory.map((job, idx) => `
        <div class="job-item ${job.status || ''}">
            <div class="job-title">${idx + 1}. ${escapeHtml(job.video_path)}</div>
            <div class="job-time">${job.submit_time}</div>
        </div>
    `).join('');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
