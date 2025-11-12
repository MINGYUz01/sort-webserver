import os
import shutil
from pathlib import Path
from typing import Tuple, Optional
import cv2
from ultralytics import YOLO
from yolo_deepsort_processor import YOLODeepSORTProcessor

# 定义轨迹选项模型以避免循环导入
class TrackingOptions:
    def __init__(self, show_trajectory: bool = False, trajectory_length: int = 30, trajectory_color: str = "red"):
        self.show_trajectory = show_trajectory
        self.trajectory_length = trajectory_length
        self.trajectory_color = trajectory_color

class VideoProcessor:
    """视频处理类，负责视频格式转换和优化"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        self.processor = YOLODeepSORTProcessor('weights/best.pt')
    
    def process_video(self, input_path: Path, original_output_path: Path, processed_output_path: Path, 
                     tracking_options: Optional[TrackingOptions] = None) -> bool:
        """
        处理视频文件，同时保存原始视频（确保Edge浏览器兼容性）和处理后的视频（带目标检测和跟踪）
        
        Args:
            input_path: 输入视频路径
            original_output_path: 原始视频输出路径（仅转换格式）
            processed_output_path: 处理后视频输出路径（带检测和跟踪）
            tracking_options: 轨迹显示选项
            
        Returns:
            bool: 处理是否成功
        """
        # 清除轨迹历史
        self.processor.trajectory_history.clear()
            
        try:
            # 检查输入文件是否存在
            if not input_path.exists():
                print(f"输入文件不存在: {input_path}")
                return False
            
            # 检查是否安装了OpenCV
            try:
                import cv2
                # 使用OpenCV进行视频处理
                cap = cv2.VideoCapture(str(input_path))
                
                if not cap.isOpened():
                    print("无法打开视频文件")
                    return False
                
                # 获取视频信息
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # 创建视频写入器，使用更兼容的编码器
                # 尝试使用H.264编码器（如果可用），否则使用mp4v
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
                except:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 备用编码器
                
                # 创建两个输出视频写入器
                original_out = cv2.VideoWriter(str(original_output_path), fourcc, fps, (width, height))
                processed_out = cv2.VideoWriter(str(processed_output_path), fourcc, fps, (width, height))
                
                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # 读取并写入每一帧
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 写入原始帧到原始视频
                    original_out.write(frame)
                    
                    # 根据轨迹选项处理帧并写入处理后的视频
                    if tracking_options and tracking_options.show_trajectory:
                        # 使用轨迹显示功能
                        _, processed_frame = self.processor.process_detection_results(
                            frame, 
                            trajectory_length=tracking_options.trajectory_length,
                            trajectory_color=tracking_options.trajectory_color
                        )
                    else:
                        # 使用普通处理（不显示轨迹）
                        _, processed_frame = self.processor.process_detection_results(frame)
                    
                    processed_out.write(processed_frame)
                    
                    frame_count += 1
                    if frame_count % 10 == 0:
                        print(f"已处理 {frame_count}/{total_frames} 帧")
                
                # 释放资源
                cap.release()
                original_out.release()
                processed_out.release()
                
                # 检查输出文件是否创建成功
                original_success = original_output_path.exists() and original_output_path.stat().st_size > 0
                processed_success = processed_output_path.exists() and processed_output_path.stat().st_size > 0
                
                if original_success and processed_success:
                    print(f"原始视频处理完成: {original_output_path}")
                    print(f"处理后视频完成: {processed_output_path}")
                    return True
                else:
                    if not original_success:
                        print("原始视频输出文件创建失败")
                    if not processed_success:
                        print("处理后视频输出文件创建失败")
                    return False
                    
            except ImportError:
                # 如果没有OpenCV，使用简单的文件复制
                print("OpenCV未安装，使用简单文件复制")
                shutil.copy2(input_path, original_output_path)
                shutil.copy2(input_path, processed_output_path)
                return True
                
        except Exception as e:
            print(f"视频处理异常: {str(e)}")
            return False
    

    def get_video_info(self, video_path: Path) -> dict:
        """获取视频信息"""
        try:
            # 检查是否安装了OpenCV
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                
                if not cap.isOpened():
                    return {}
                
                # 获取基本信息
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # 计算时长
                duration = frame_count / fps if fps > 0 else 0
                
                # 获取文件大小
                size = video_path.stat().st_size
                
                video_info = {
                    'duration': duration,
                    'size': size,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'frame_count': frame_count,
                    'codec': 'Unknown',  # OpenCV无法直接获取编解码器信息
                    'audio_codec': 'Unknown'
                }
                
                cap.release()
                return video_info
                
            except ImportError:
                # 如果没有OpenCV，返回基本文件信息
                if video_path.exists():
                    return {
                        'size': video_path.stat().st_size,
                        'duration': 0,
                        'width': 0,
                        'height': 0,
                        'fps': 0,
                        'codec': 'Unknown',
                        'audio_codec': 'Unknown'
                    }
                
        except Exception as e:
            print(f"获取视频信息失败: {str(e)}")
        
        return {}
    


# 简单的视频处理函数，用于测试
def simple_process_video(input_path: Path, output_path: Path) -> bool:
    """简化版视频处理，用于没有FFmpeg的环境"""
    try:
        # 如果没有FFmpeg，直接复制文件（仅用于演示）
        import shutil
        shutil.copy2(input_path, output_path)
        return True
    except Exception as e:
        print(f"简单视频处理失败: {str(e)}")
        return False