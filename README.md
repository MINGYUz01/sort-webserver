# YOLO+DeepSORT 视频目标检测跟踪应用

一个基于FastAPI和YOLO+DeepSORT的智能视频处理应用，支持视频上传、目标检测跟踪、在线播放和下载功能。

## 功能特性

- 🎥 **智能视频上传**: 支持拖拽上传和点击选择MP4文件
- 🔍 **YOLO目标检测**: 基于YOLOv11的实时目标检测
- 🎯 **DeepSORT跟踪**: 多目标跟踪和ID保持
- 📺 **在线播放对比**: 支持原始视频和检测跟踪结果的在线播放
- 💾 **一键下载**: 提供原始和处理后视频的下载功能
- 🎨 **美观界面**: 现代化响应式设计，支持移动端
- 🔄 **实时进度**: 显示处理进度和状态

## 技术栈

### 后端
- **FastAPI**: 高性能Python Web框架
- **YOLOv11**: 实时目标检测模型
- **DeepSORT**: 多目标跟踪算法
- **OpenCV**: 计算机视觉处理
- **Uvicorn**: ASGI服务器

### AI/ML
- **Ultralytics YOLO**: 目标检测框架
- **DeepSORT**: 目标跟踪算法
- **PyTorch**: 深度学习框架

### 前端
- **HTML5/CSS3**: 现代化界面设计
- **JavaScript**: 交互逻辑和API调用
- **Font Awesome**: 图标库

## 安装和运行

### 前置要求

1. **Python 3.8+**
2. **CUDA支持** (可选，用于GPU加速)
3. **FFmpeg** (用于视频处理)

#### 安装FFmpeg

**Windows:**
```bash
# 使用chocolatey安装
choco install ffmpeg

# 或手动下载并添加到PATH
# 从 https://ffmpeg.org/download.html 下载
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

### 安装步骤

1. **克隆或下载项目**
```bash
git clone <repository-url>
cd sort-webserver
```

2. **安装Python依赖**
```bash
pip install -r requirements.txt
```

3. **下载预训练模型** (可选)
项目已包含预训练权重文件，如需使用自定义模型：
- 将YOLO模型文件(.pt)放入 `weights/` 目录
- 修改 `yolo_deepsort_processor.py` 中的模型路径

4. **运行应用**
```bash
python run.py
```

或者使用uvicorn直接运行：
```bash
uvicorn main:app --host 0.0.0.0 --port 8066 --reload
```

5. **访问应用**
打开浏览器访问: http://localhost:8066

## 项目结构

```
sort-webserver/
├── main.py                 # FastAPI主应用
├── run.py                 # 应用启动脚本
├── video_processor.py      # 视频处理逻辑
├── yolo_deepsort_processor.py # YOLO+DeepSORT处理器
├── requirements.txt        # Python依赖
├── README.md              # 项目说明
├── deep_sort/             # DeepSORT算法实现
│   ├── configs/
│   │   └── deep_sort.yaml  # 跟踪器配置
│   ├── deep_sort/         # DeepSORT核心代码
│   └── utils/             # 工具函数
├── weights/               # 模型权重文件
│   ├── best.pt           # 自定义训练模型
│   ├── yolo11l.pt        # YOLOv11大模型
│   └── yolo11s.pt        # YOLOv11小模型
├── templates/             # HTML模板
│   └── index.html         # 主页面
├── static/               # 静态文件
│   ├── css/
│   │   └── style.css      # 样式文件
│   └── js/
│       └── script.js      # JavaScript逻辑
├── uploads/              # 上传文件目录
├── processed/            # 处理结果目录
└── test_video/           # 测试视频目录
```

## API接口

### 上传视频
- **POST** `/upload/`
- 参数: `file` (MP4视频文件)
- 返回: 文件ID和处理结果

### 下载视频
- **GET** `/download/original/{file_id}` - 下载原始视频
- **GET** `/download/processed/{file_id}` - 下载处理后视频

### 状态查询
- **GET** `/api/status/{file_id}` - 查询处理状态

### 目标检测跟踪
- **内部处理**: 视频处理过程中自动执行YOLO+DeepSORT算法
- **输出结果**: 包含目标边界框和跟踪ID的处理后视频

## 使用说明

1. **上传视频**: 点击"选择文件"或拖拽视频文件到上传区域
2. **目标检测跟踪**: 系统自动执行YOLO目标检测和DeepSORT多目标跟踪
3. **查看结果**: 处理完成后可在线预览包含目标框和跟踪ID的视频
4. **下载视频**: 点击下载按钮获取原始或处理后的视频文件

### 功能特点
- **实时检测**: 支持多种目标类别检测（人、车、动物等）
- **多目标跟踪**: 为每个检测到的目标分配唯一跟踪ID
- **边界框显示**: 实时显示目标位置和类别信息
- **性能优化**: 支持CPU和GPU加速处理

## 视频处理说明

应用会对上传的视频进行以下优化处理：

- **视频编码**: 使用H.264编码，确保浏览器兼容性
- **音频编码**: 使用AAC编码，128kbps比特率
- **分辨率优化**: 确保分辨率为偶数
- **网络优化**: 添加faststart标志，优化在线播放
- **质量平衡**: CRF 23参数，平衡文件大小和质量

## 浏览器兼容性

- ✅ Chrome 60+
- ✅ Firefox 55+
- ✅ Edge 79+
- ✅ Safari 11+
- ✅ 移动端浏览器

<!-- ## 开发说明

### 添加新的视频处理功能

编辑 `video_processor.py` 文件：

```python
def new_processing_method(self, input_path, output_path):
    # 添加自定义处理逻辑
    pass
```

### 修改前端界面

编辑 `templates/index.html` 和 `static/css/style.css` 文件。

### 添加新的API端点

在 `main.py` 中添加新的路由：

```python
@app.get("/api/new-endpoint")
async def new_endpoint():
    return {"message": "新功能"}
``` -->

## 故障排除

### 常见问题

1. **FFmpeg未找到错误**
   - 确保FFmpeg已正确安装并添加到PATH
   - 在命令行运行 `ffmpeg -version` 验证安装

2. **视频上传失败**
   - 检查文件是否为MP4格式
   - 确保文件大小不超过100MB
   - 检查网络连接

3. **视频处理失败**
   - 检查FFmpeg是否支持输入视频的编码格式
   - 查看控制台错误信息

### 日志查看

应用运行时会输出详细的日志信息，包括：
- 文件上传状态
- 视频处理进度
- 错误信息

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- 创建GitHub Issue
- 发送邮件到项目维护者

---

**注意**: 这是一个演示项目，生产环境使用前请进行充分测试和安全评估。