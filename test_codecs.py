import cv2
import numpy as np
import subprocess
import os

def comprehensive_codec_check():
    """全面的编解码器兼容性检查"""
    
    print("=== OpenCV 编解码器兼容性检查 ===")
    
    # 检查构建信息
    print(f"OpenCV 版本: {cv2.__version__}")
    
    # 测试可用的后端
    backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
    for backend in backends:
        out = cv2.VideoWriter()
        success = out.open('test_video/output/output.mp4', backend, 
                        cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
        if success:
            print(f"使用后端 {backend} 成功")
        out.release()

    # 测试各种编码器
    codecs_to_test = [
        ('avc1', 'mp4', 'H.264/AVC1'),
        ('h264', 'mp4', 'H.264'),
        ('x264', 'mp4', 'x264'),
        ('mp4v', 'mp4', 'MPEG-4'),
        ('XVID', 'avi', 'XVID'),
        ('MJPG', 'avi', 'Motion-JPEG'),
    ]
    
    working_codecs = []
    
    for codec, ext, desc in codecs_to_test:
        print(f'test: {codec}, {ext}, {desc}')
        try:
            filename = f'test_{codec}.{ext}'
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(filename, fourcc, 20, (640, 480))
            
            if out.isOpened():
                # 写入测试帧
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
                out.write(frame)
                out.release()
                
                # 检查文件是否创建成功
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    working_codecs.append((codec, ext, desc))
                    print(f"✓ {desc} ({codec}) - 支持")
                    # 清理测试文件
                    os.remove(filename)
                else:
                    print(f"✗ {desc} ({codec}) - 文件创建失败")
            else:
                print(f"✗ {desc} ({codec}) - 无法打开")
                
        except Exception as e:
            print(f"✗ {desc} ({codec}) - 错误: {e}")
    
    print(f"\n可用的编码器: {len(working_codecs)}")
    for codec, ext, desc in working_codecs:
        print(f"  - {desc} (.{ext}, fourcc: {codec})")
    
    return working_codecs

# 运行检查
available_codecs = comprehensive_codec_check()