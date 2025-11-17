#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
目标跟踪推理脚本

此脚本提供了一个简单的接口，用于使用YOLO-DeepSORT模型进行视频目标跟踪推理。
用户可以通过命令行参数指定输入视频路径、输出视频路径以及其他配置选项。

用法示例：
    python track_inference.py --input_video ./test_video.mp4 --output_video ./output_tracking.mp4 --show --save

依赖：
    - opencv-python
    - numpy
    - torch (YOLOv8所需)
    - yolov8 (ultralytics)
    - deepsort库
"""

import argparse
import os
import cv2
import time
import numpy as np
from yolo_deepsort_processor import YOLODeepSORTProcessor


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='YOLO-DeepSORT目标跟踪推理脚本')
    parser.add_argument('--input_video', type=str, default='./test_video/bottles.mp4', 
                        help='输入视频文件路径')
    parser.add_argument('--output_video', type=str, default='./output_tracking.mp4', 
                        help='输出视频文件保存路径')
    parser.add_argument('--show', action='store_true', default=True, 
                        help='是否实时显示处理结果')
    parser.add_argument('--save', action='store_true', default=False, 
                        help='是否保存处理后的视频')
    parser.add_argument('--confidence_threshold', type=float, default=0.25, 
                        help='检测置信度阈值')
    parser.add_argument('--display_tracks', action='store_true', default=True, 
                        help='是否显示跟踪轨迹')
    parser.add_argument('--track_history_length', type=int, default=100, 
                        help='轨迹历史长度')
    parser.add_argument('--fps', type=int, default=None, 
                        help='输出视频帧率，默认与输入视频相同')
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='运行设备 (cuda:0 或 cpu)')
    parser.add_argument('--class_ids', type=int, nargs='+', default=[0], 
                        help='要检测和跟踪的类别ID列表，默认只跟踪人(0)')
    return parser.parse_args()


def process_video(args):
    """
    处理视频并进行目标跟踪
    """
    # 检查输入视频文件是否存在
    if not os.path.exists(args.input_video):
        print(f"错误：找不到输入视频文件 {args.input_video}")
        return False
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化视频捕获
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {args.input_video}")
        return False
    
    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if args.fps is None else args.fps
    
    # 初始化视频写入器（如果需要保存）
    out = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output_video, fourcc, fps, (frame_width, frame_height))
    
    # 初始化YOLO-DeepSORT处理器
    use_cuda = 'cuda' in args.device.lower()
    processor = YOLODeepSORTProcessor(
        model_path='weights/best_2.pt',  # 使用默认模型路径
        use_cuda=use_cuda
    )
    
    # 用于计算FPS
    prev_time = 0
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"开始处理视频：{args.input_video}")
    print(f"视频信息：{frame_width}x{frame_height}, {fps} FPS, 总帧数：{total_frames}")
    
    # 处理每一帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 记录开始时间
        start_time = time.time()
        
        # 进行目标跟踪
        tracking_results, processed_frame = processor.process_detection_results(
            frame, 
            min_confidence=args.confidence_threshold,
            trajectory_length=args.track_history_length if args.display_tracks else 0,
            trajectory_color="red"  # 轨迹颜色可以根据需要修改
        )
        
        # 计算处理时间和FPS
        processing_time = time.time() - start_time
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        
        # 在画面上显示FPS
        cv2.putText(
            processed_frame, 
            f"FPS: {current_fps:.2f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # 显示进度
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"进度: {progress:.1f}% | FPS: {current_fps:.2f}", end='\r')
        
        # 显示结果（如果需要）
        if args.show:
            cv2.imshow('目标跟踪结果', processed_frame)
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 保存结果（如果需要）
        if args.save:
            out.write(processed_frame)
    
    print("\n视频处理完成！")
    
    # 释放资源
    cap.release()
    if args.save and out is not None:
        out.release()
        print(f"结果已保存到：{args.output_video}")
    if args.show:
        cv2.destroyAllWindows()
    
    return True


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    # args.input_video = './test_video/bottles_2.mp4'
    
    # 处理视频
    success = process_video(args)
    
    if success:
        print("\n目标跟踪推理成功完成！")
    else:
        print("\n目标跟踪推理过程中出现错误！")


if __name__ == "__main__":
    main()