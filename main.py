from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import os
import uuid
import shutil
from pathlib import Path
from video_processor import VideoProcessor

# 创建应用实例
app = FastAPI(title="视频处理应用", description="一个简单易用的视频上传、处理和下载应用")

# 创建必要的目录
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
STATIC_DIR = Path("static")

for directory in [UPLOAD_DIR, PROCESSED_DIR, STATIC_DIR]:
    directory.mkdir(exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 模板配置
templates = Jinja2Templates(directory="templates")

# 视频处理器实例
video_processor = VideoProcessor()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """主页面"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    """上传视频文件"""
    # 验证文件类型
    if not file.filename.lower().endswith('.mp4'):
        raise HTTPException(status_code=400, detail="只支持MP4格式文件")
    
    # 生成唯一文件名
    file_id = str(uuid.uuid4())
    original_filename = f"{file_id}_original.mp4"
    processed_filename = f"{file_id}_processed.mp4"
    
    # 保存原始文件到临时位置
    temp_path = UPLOAD_DIR / f"{file_id}_temp.mp4"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 初始化路径变量
    original_path = UPLOAD_DIR / original_filename
    processed_path = PROCESSED_DIR / processed_filename
    
    try:
        # 处理原始视频，确保Edge浏览器兼容性
        original_success = video_processor.process_original_video(temp_path, original_path)
        
        if not original_success:
            raise HTTPException(status_code=500, detail="原始视频处理失败")
        
        # 处理视频（优化版本）
        processed_success = video_processor.process_video(original_path, processed_path)
        
        if not processed_success:
            raise HTTPException(status_code=500, detail="视频处理失败")
        
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()
        
        return {
            "file_id": file_id,
            "original_url": f"/download/original/{file_id}",
            "processed_url": f"/download/processed/{file_id}",
            "message": "视频上传和处理成功"
        }
    
    except Exception as e:
        # 只清理临时文件，保留已成功处理的文件
        if temp_path.exists():
            temp_path.unlink()
        # 只有在原始视频处理失败时才删除原始文件
        if not original_path.exists() or original_path.stat().st_size == 0:
            if original_path.exists():
                original_path.unlink()
        # 只有在视频处理失败时才删除处理后的文件
        if not processed_path.exists() or processed_path.stat().st_size == 0:
            if processed_path.exists():
                processed_path.unlink()
        
        # 记录详细的错误信息
        print(f"视频处理过程中发生错误: {str(e)}")
        print(f"临时文件状态: {temp_path.exists()}")
        print(f"原始文件状态: {original_path.exists()}")
        print(f"处理后文件状态: {processed_path.exists()}")
        
        raise HTTPException(status_code=500, detail=f"处理错误: {str(e)}")

@app.get("/download/original/{file_id}")
async def download_original(file_id: str):
    """下载原始视频"""
    file_path = UPLOAD_DIR / f"{file_id}_original.mp4"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=file_path,
        filename=f"original_{file_id}.mp4",
        media_type='video/mp4'
    )

@app.get("/download/processed/{file_id}")
async def download_processed(file_id: str):
    """下载处理后的视频"""
    file_path = PROCESSED_DIR / f"{file_id}_processed.mp4"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=file_path,
        filename=f"processed_{file_id}.mp4",
        media_type='video/mp4'
    )

@app.get("/api/status/{file_id}")
async def get_status(file_id: str):
    """获取处理状态"""
    original_exists = (UPLOAD_DIR / f"{file_id}_original.mp4").exists()
    processed_exists = (PROCESSED_DIR / f"{file_id}_processed.mp4").exists()
    
    return {
        "file_id": file_id,
        "original_ready": original_exists,
        "processed_ready": processed_exists,
        "status": "completed" if processed_exists else "processing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
