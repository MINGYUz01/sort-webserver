// 全局变量
let currentFileId = null;
let uploadInProgress = false;
let currentFile = null; // 存储当前选择的文件

// DOM元素
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const uploadBtn = document.getElementById('uploadBtn');
const statusSection = document.getElementById('statusSection');
const progressBar = document.getElementById('progressBar');
const statusMessage = document.getElementById('statusMessage');
const resultSection = document.getElementById('resultSection');
const originalVideo = document.getElementById('originalVideo');
const processedVideo = document.getElementById('processedVideo');
const originalSource = document.getElementById('originalSource');
const processedSource = document.getElementById('processedSource');
const downloadOriginal = document.getElementById('downloadOriginal');
const downloadProcessed = document.getElementById('downloadProcessed');
const newUploadBtn = document.getElementById('newUploadBtn');

// 事件监听器
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

function initializeEventListeners() {
    // 上传区域点击事件
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // 文件选择事件
    fileInput.addEventListener('change', handleFileSelect);
    
    // 拖拽事件
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // 上传按钮事件
    uploadBtn.addEventListener('click', handleUpload);
    
    // 新上传按钮事件
    newUploadBtn.addEventListener('click', resetInterface);
}

// 文件选择处理
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        currentFile = file; // 存储文件
        validateAndDisplayFile(file);
    }
}

// 拖拽事件处理
function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = event.dataTransfer.files[0];
    if (file) {
        currentFile = file; // 存储文件
        validateAndDisplayFile(file);
    }
}

// 文件验证和显示
function validateAndDisplayFile(file) {
    // 验证文件类型
    if (!file.type.includes('mp4')) {
        showError('请选择MP4格式的视频文件');
        return;
    }
    
    // 验证文件大小（最大100MB）
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
        showError('文件大小不能超过100MB');
        return;
    }
    
    // 显示文件信息
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    // 显示文件信息区域
    fileInfo.style.display = 'flex';
    fileInfo.classList.add('fade-in');
}

// 文件大小格式化
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 上传处理
async function handleUpload() {
    if (uploadInProgress) return;
    
    if (!currentFile) {
        showError('请先选择文件');
        return;
    }
    
    uploadInProgress = true;
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<div class="loading"></div> 处理中...';
    
    // 显示处理状态
    showProcessingStatus();
    
    try {
        // 创建FormData
        const formData = new FormData();
        formData.append('file', currentFile);
        
        // 发送上传请求
        const response = await fetch('/upload/', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            let errorMessage = '上传失败';
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || '上传失败';
            } catch (jsonError) {
                // 如果JSON解析失败，尝试获取文本错误信息
                try {
                    const text = await response.text();
                    errorMessage = text || '上传失败';
                } catch (textError) {
                    errorMessage = `服务器错误: ${response.status} ${response.statusText}`;
                }
            }
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        currentFileId = result.file_id;
        
        // 开始轮询处理状态
        await pollProcessingStatus();
        
    } catch (error) {
        console.error('上传错误:', error);
        showError('处理失败: ' + error.message);
        
        // 停止进度条动画并重置界面
        if (window.progressInterval) {
            clearInterval(window.progressInterval);
        }
        progressBar.style.width = '0%';
        statusSection.style.display = 'none';
        resetUploadButton();
    }
}

// 显示处理状态
function showProcessingStatus() {
    statusSection.style.display = 'block';
    statusSection.classList.add('fade-in');
    
    // 模拟进度条动画
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 10;
        if (progress > 90) progress = 90;
        progressBar.style.width = progress + '%';
    }, 500);
    
    // 保存interval以便清除
    window.progressInterval = interval;
}

// 轮询处理状态
async function pollProcessingStatus() {
    if (!currentFileId) return;
    
    try {
        const response = await fetch(`/api/status/${currentFileId}`);
        if (!response.ok) {
            let errorMessage = '状态查询失败';
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || '状态查询失败';
            } catch (jsonError) {
                // 如果JSON解析失败，尝试获取文本错误信息
                try {
                    const text = await response.text();
                    errorMessage = text || '状态查询失败';
                } catch (textError) {
                    errorMessage = `服务器错误: ${response.status} ${response.statusText}`;
                }
            }
            throw new Error(errorMessage);
        }
        
        const status = await response.json();
        
        if (status.status === 'completed') {
            // 处理完成
            clearInterval(window.progressInterval);
            progressBar.style.width = '100%';
            statusMessage.textContent = '处理完成！';
            
            // 显示结果
            setTimeout(() => {
                showResults();
            }, 1000);
            
        } else {
            // 继续轮询
            setTimeout(() => pollProcessingStatus(), 2000);
        }
        
    } catch (error) {
        console.error('状态查询错误:', error);
        // 如果状态查询失败，显示错误并重置界面
        showError('状态查询失败: ' + error.message);
        resetUploadButton();
        statusSection.style.display = 'none';
    }
}

// 显示结果
function showResults() {
    if (!currentFileId) return;
    
    // 设置视频源
    originalSource.src = `/download/original/${currentFileId}`;
    processedSource.src = `/download/processed/${currentFileId}`;
    
    // 重新加载视频
    originalVideo.load();
    processedVideo.load();
    
    // 设置下载链接
    downloadOriginal.href = `/download/original/${currentFileId}`;
    downloadProcessed.href = `/download/processed/${currentFileId}`;
    
    // 显示结果区域
    statusSection.style.display = 'none';
    resultSection.style.display = 'block';
    resultSection.classList.add('fade-in');
    
    // 重置上传按钮状态
    resetUploadButton();
}

// 重置界面
function resetInterface() {
    // 重置所有状态
    currentFileId = null;
    uploadInProgress = false;
    currentFile = null; // 清空当前文件
    
    // 重置文件输入
    fileInput.value = '';
    
    // 隐藏所有区域
    fileInfo.style.display = 'none';
    statusSection.style.display = 'none';
    resultSection.style.display = 'none';
    
    // 重置上传按钮
    resetUploadButton();
    
    // 清除进度条
    if (window.progressInterval) {
        clearInterval(window.progressInterval);
    }
    progressBar.style.width = '0%';
}

// 重置上传按钮
function resetUploadButton() {
    uploadBtn.disabled = false;
    uploadBtn.innerHTML = '<i class="fas fa-upload"></i> 上传并处理';
    uploadInProgress = false;
}

// 显示错误信息
function showError(message) {
    // 创建错误提示
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #f8d7da;
        color: #721c24;
        padding: 15px 20px;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        z-index: 1000;
        max-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    `;
    errorDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle" style="margin-right: 10px;"></i>
        ${message}
    `;
    
    document.body.appendChild(errorDiv);
    
    // 3秒后自动移除
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.parentNode.removeChild(errorDiv);
        }
    }, 3000);
}

// 显示成功信息
function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #d4edda;
        color: #155724;
        padding: 15px 20px;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        z-index: 1000;
        max-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    `;
    successDiv.innerHTML = `
        <i class="fas fa-check-circle" style="margin-right: 10px;"></i>
        ${message}
    `;
    
    document.body.appendChild(successDiv);
    
    setTimeout(() => {
        if (successDiv.parentNode) {
            successDiv.parentNode.removeChild(successDiv);
        }
    }, 3000);
}

// 视频播放器控制
function setupVideoControls() {
    // 添加视频播放事件监听
    const videos = document.querySelectorAll('video');
    videos.forEach(video => {
        video.addEventListener('play', function() {
            // 暂停其他视频
            videos.forEach(otherVideo => {
                if (otherVideo !== video && !otherVideo.paused) {
                    otherVideo.pause();
                }
            });
        });
    });
}

// 页面加载完成后设置视频控制
window.addEventListener('load', setupVideoControls);

// 添加键盘快捷键支持
document.addEventListener('keydown', function(event) {
    // ESC键重置界面
    if (event.key === 'Escape') {
        resetInterface();
    }
    
    // 空格键切换播放/暂停（当焦点在视频上时）
    if (event.key === ' ' && event.target.tagName === 'VIDEO') {
        event.preventDefault();
        if (event.target.paused) {
            event.target.play();
        } else {
            event.target.pause();
        }
    }
});