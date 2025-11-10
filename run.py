#!/usr/bin/env python3
"""
视频处理应用启动脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def check_ffmpeg():
    """检查FFmpeg是否安装"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg已安装")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ FFmpeg未安装或未添加到PATH")
    print("请先安装FFmpeg：")
    print("Windows: choco install ffmpeg")
    print("macOS: brew install ffmpeg") 
    print("Linux: sudo apt install ffmpeg")
    return False

def check_dependencies():
    """检查Python依赖"""
    try:
        import fastapi
        import uvicorn
        import moviepy
        print("✅ Python依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def create_directories():
    """创建必要的目录"""
    directories = ['uploads', 'processed', 'static', 'templates']
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ 创建目录: {dir_name}")

def cleanup_directories():
    """清理uploads和processed文件夹中的文件"""
    directories_to_clean = ['uploads', 'processed']
    
    for dir_name in directories_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            # 删除目录中的所有文件，但保留目录本身
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        print(f"🗑️  删除文件: {file_path}")
                    except Exception as e:
                        print(f"⚠️  删除文件失败 {file_path}: {e}")
            print(f"✅ 清理完成: {dir_name}")
        else:
            print(f"ℹ️  目录不存在，无需清理: {dir_name}")

def main():
    """主函数"""
    print("🚀 视频处理应用启动检查")
    print("=" * 50)
    
    # 检查当前目录
    current_dir = Path(__file__).parent
    print(f"📁 工作目录: {current_dir}")
    
    # 检查必要文件
    required_files = ['main.py', 'video_processor.py', 'requirements.txt']
    for file in required_files:
        if (current_dir / file).exists():
            print(f"✅ 文件存在: {file}")
        else:
            print(f"❌ 文件缺失: {file}")
            return False
    
    # 创建目录
    create_directories()
    
    # 检查依赖
    if not check_dependencies():
        return False
    
    # 检查FFmpeg
    if not check_ffmpeg():
        print("⚠️  注意: 没有FFmpeg将使用简化处理模式")
    
    print("=" * 50)
    print("🎉 所有检查完成！")
    print("\n启动应用...")
    
    # 启动应用
    try:
        import uvicorn
        print("🌐 服务器启动中...")
        print("📱 访问地址: http://localhost:8066")
        print("⏹️  按 Ctrl+C 停止服务器")
        print("-" * 50)
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0", 
            port=8066,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    finally:
        # 无论应用如何退出，都执行清理操作
        print("🧹 正在清理临时文件...")
        cleanup_directories()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)