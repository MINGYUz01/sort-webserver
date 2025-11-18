#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
姿态估计推理脚本

此脚本提供了一个简单的接口，用于使用YOLOv8姿态估计模型进行视频人体姿态分析。
用户可以通过命令行参数指定输入视频路径、输出视频路径以及其他配置选项。

用法示例：
    python pose_inference.py --input_video ./test_video.mp4 --output_video ./output_pose.mp4 --show --save

依赖：
    - opencv-python
    - numpy
    - torch (YOLOv8所需)
    - yolov8 (ultralytics)
"""

import argparse
import os
import cv2
import time
import numpy as np
from yolo_pose_processor import YOLOPoseProcessor


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='YOLOv8姿态估计推理脚本')
    parser.add_argument('--input_video', type=str, default='./test_video/pose_bottles.mp4', 
                        help='输入视频文件路径')
    parser.add_argument('--output_video', type=str, default='./output_pose.mp4', 
                        help='输出视频文件保存路径')
    parser.add_argument('--show', action='store_true', default=True, 
                        help='是否实时显示处理结果')
    parser.add_argument('--save', action='store_true', default=False, 
                        help='是否保存处理后的视频')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, 
                        help='姿态估计置信度阈值')
    parser.add_argument('--line_thickness', type=int, default=2, 
                        help='绘制骨架线条粗细')
    parser.add_argument('--keypoint_radius', type=int, default=3, 
                        help='关键点半径')
    parser.add_argument('--person_color', type=str, default='blue', 
                        help='人体框颜色 (blue, green, red, yellow, cyan, magenta, white)')
    parser.add_argument('--skeleton_color', type=str, default='green', 
                        help='骨架线条颜色 (blue, green, red, yellow, cyan, magenta, white)')
    parser.add_argument('--keypoint_color', type=str, default='yellow', 
                        help='关键点颜色 (blue, green, red, yellow, cyan, magenta, white)')
    parser.add_argument('--fps', type=int, default=None, 
                        help='输出视频帧率，默认与输入视频相同')
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='运行设备 (cuda:0 或 cpu)')
    parser.add_argument('--max_people', type=int, default=None, 
                        help='最大处理人数，默认不限制')
    return parser.parse_args()


def get_color_by_name(color_name):
    """
    根据颜色名称返回BGR颜色值
    """
    color_map = {
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255)
    }
    return color_map.get(color_name.lower(), (0, 255, 0))


def process_video(args):
    """
    处理视频并进行姿态估计
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
    
    # 初始化YOLOPose处理器
    use_cuda = 'cuda' in args.device.lower()
    processor = YOLOPoseProcessor(
        model_path='weights/yolo11s-pose.pt',  # 使用默认姿态估计模型
        use_cuda=use_cuda
    )
    
    # 用于计算FPS
    prev_time = 0
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"开始处理视频：{args.input_video}")
    print(f"视频信息：{frame_width}x{frame_height}, {fps} FPS, 总帧数：{total_frames}")
    print(f"姿态估计配置：置信度阈值={args.confidence_threshold}, 线条粗细={args.line_thickness}, 关键点半径={args.keypoint_radius}")
    
    # 处理每一帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 记录开始时间
        start_time = time.time()
        
        # 进行姿态估计
        pose_results, processed_frame = processor.process_pose_detection(
            frame, 
            min_confidence=args.confidence_threshold,
            mask_separation=False  # 使用原图背景
        )
        
        # 如果设置了最大人数限制，则只保留指定数量的结果
        if args.max_people is not None and pose_results is not None:
            pose_results = pose_results[:args.max_people]
        
        # 计算处理时间和FPS
        processing_time = time.time() - start_time
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        
        # 在画面上显示FPS和检测到的人数
        cv2.putText(
            processed_frame, 
            f"FPS: {current_fps:.2f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        if pose_results is not None and len(pose_results) > 0:
            cv2.putText(
                processed_frame, 
                f"检测到的人数: {len(pose_results)}", 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
        
        # 显示进度
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"进度: {progress:.1f}% | FPS: {current_fps:.2f} | 检测人数: {len(pose_results) if pose_results is not None else 0}", end='\r')
        
        # 显示结果（如果需要）
        if args.show:
            cv2.imshow('姿态估计结果', processed_frame)
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
    
    # 处理视频
    success = process_video(args)
    
    if success:
        print("\n姿态估计推理成功完成！")
    else:
        print("\n姿态估计推理过程中出现错误！")


if __name__ == "__main__":
    main()