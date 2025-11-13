import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

class YOLOPoseProcessor:
    """YOLO 姿态检测处理器"""
    
    # 人体关键点连接关系（COCO格式）
    POSE_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 躯干和手臂
        (5, 11), (6, 12), (11, 12),  # 躯干
        (11, 13), (13, 15), (12, 14), (14, 16)  # 腿部
    ]
    
    # 关键点名称（COCO格式）
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, model_path='weights/yolo11s-pose.pt', use_cuda=False):
        """初始化姿态检测处理器
        
        Args:
            model_path: YOLO姿态检测模型路径
            use_cuda: 是否使用GPU
        """
        self.model = YOLO(model_path)
        self.use_cuda = use_cuda
        
        # 设置设备
        if use_cuda and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        print(f"姿态检测模型已加载，使用设备: {self.device}")
    
    def process_pose_detection(self, image, mask_separation=False, min_confidence=0.25):
        """处理单帧图像的姿态检测
        
        Args:
            image: 输入图像 (BGR格式)
            mask_separation: 是否使用掩码分离（True: 黑背景，False: 叠加到原图）
            min_confidence: 最小置信度阈值
            
        Returns:
            tuple: (姿态检测结果列表, 处理后的图像)
        """
        # 复制图像用于绘制
        if mask_separation:
            # 创建黑色背景
            processed_image = np.zeros_like(image)
        else:
            # 使用原图背景
            processed_image = image.copy()
        
        # 使用YOLO进行姿态检测
        pose_results = self._yolo_pose_detection(image, min_confidence)
        
        # 在图像上绘制姿态检测结果
        self._draw_pose_detections(processed_image, pose_results)
        
        return pose_results, processed_image
    
    def _yolo_pose_detection(self, image, min_confidence):
        """YOLO姿态检测
        
        Args:
            image: 输入图像
            min_confidence: 最小置信度
            
        Returns:
            list: 姿态检测结果 [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'keypoints': [[x, y, conf], ...],  # 17个关键点
                    'class_id': int
                }, ...
            ]
        """
        results = self.model(image, conf=min_confidence, device=self.device)
        pose_detections = []
        
        for result in results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                for i, (box, keypoints) in enumerate(zip(result.boxes, result.keypoints)):
                    if box.conf[0] > min_confidence:
                        # 获取边界框
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # 获取关键点
                        keypoints_data = keypoints.xy.cpu().numpy()
                        keypoints_conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else np.ones(len(keypoints_data))
                        
                        # 组合关键点信息 - 修复数组转换问题
                        keypoints_list = []
                        
                        # 关键点数据形状为 (1, 17, 2)，需要处理第一个维度的数据
                        if keypoints_data.ndim == 3:
                            # 取第一个检测结果的关键点
                            person_keypoints = keypoints_data[0]  # 形状为 (17, 2)
                            person_confidences = keypoints_conf[0] if keypoints_conf.ndim == 2 else keypoints_conf
                            
                            for j in range(len(person_keypoints)):
                                point = person_keypoints[j]  # 单个关键点，形状为 (2,)
                                conf = person_confidences[j] if j < len(person_confidences) else 1.0
                                
                                # 安全地提取坐标和置信度
                                point_x = float(point[0]) if len(point) >= 1 else 0.0
                                point_y = float(point[1]) if len(point) >= 2 else 0.0
                                conf_val = float(conf) if isinstance(conf, (np.ndarray, np.generic)) else float(conf)
                                
                                keypoints_list.append([
                                    int(point_x), int(point_y), conf_val
                                ])
                        else:
                            # 备用处理逻辑
                            for j, (point, conf) in enumerate(zip(keypoints_data, keypoints_conf)):
                                if point.size > 0:
                                    point_x = float(point[0]) if point.size >= 1 else 0.0
                                    point_y = float(point[1]) if point.size >= 2 else 0.0
                                    conf_val = float(conf) if isinstance(conf, (np.ndarray, np.generic)) else float(conf)
                                    
                                    keypoints_list.append([
                                        int(point_x), int(point_y), conf_val
                                    ])
                        
                        pose_detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(box.conf[0]),
                            'keypoints': keypoints_list,
                            'class_id': int(box.cls[0]) if box.cls is not None else 0,
                            'track_id': i  # 简单的跟踪ID
                        })
        
        return pose_detections
    
    def _draw_pose_detections(self, image, pose_results):
        """在图像上绘制姿态检测结果
        
        Args:
            image: 要绘制的图像
            pose_results: 姿态检测结果
        """
        # 定义颜色
        bbox_color = (0, 255, 0)  # 绿色边界框
        skeleton_color = (255, 0, 0)  # 蓝色骨架
        keypoint_color = (0, 0, 255)  # 红色关键点
        
        for pose in pose_results:
            bbox = pose['bbox']
            keypoints = pose['keypoints']
            confidence = pose['confidence']
            track_id = pose.get('track_id', 0)
            
            # 绘制边界框
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)
            
            # 绘制置信度和ID标签
            label = f"Pose ID:{track_id} ({confidence:.2f})"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - text_size[1] - 4), 
                         (x1 + text_size[0], y1), bbox_color, -1)
            cv2.putText(image, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 绘制关键点
            valid_keypoints = []
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.1:  # 关键点置信度阈值
                    cv2.circle(image, (int(x), int(y)), 4, keypoint_color, -1)
                    # 显示关键点名称
                    if i < len(self.KEYPOINT_NAMES):
                        cv2.putText(image, self.KEYPOINT_NAMES[i][:3], 
                                  (int(x) + 5, int(y) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    valid_keypoints.append((int(x), int(y)))
            
            # 绘制骨架连接
            for connection in self.POSE_CONNECTIONS:
                if (connection[0] < len(valid_keypoints) and 
                    connection[1] < len(valid_keypoints) and
                    len(valid_keypoints[connection[0]]) == 2 and
                    len(valid_keypoints[connection[1]]) == 2):
                    
                    pt1 = valid_keypoints[connection[0]]
                    pt2 = valid_keypoints[connection[1]]
                    cv2.line(image, pt1, pt2, skeleton_color, 2)
    
    def process_video(self, video_path, output_path='output_pose.mp4', 
                     mask_separation=False, show_preview=False, save_result=True):
        """处理视频文件进行姿态检测
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            mask_separation: 是否使用掩码分离
            show_preview: 是否显示实时预览
            save_result: 是否保存处理结果
            
        Returns:
            str: 输出文件路径
        """
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建视频写入器
        if save_result:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
            except:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 备用编码器            
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"开始处理视频进行姿态检测: {video_path}")
        print(f"视频信息: {width}x{height}, {fps:.1f} FPS, 总帧数: {total_frames}")
        print(f"掩码分离模式: {'开启' if mask_separation else '关闭'}")
        
        frame_count = 0
        
        try:
            with tqdm(total=total_frames, desc="姿态检测进度") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 处理当前帧
                    pose_results, processed_frame = self.process_pose_detection(
                        frame, mask_separation=mask_separation
                    )
                    
                    # 保存处理结果
                    if save_result:
                        out.write(processed_frame)
                    
                    # 显示预览
                    if show_preview:
                        window_name = 'YOLO姿态检测 - 掩码分离' if mask_separation else 'YOLO姿态检测'
                        cv2.imshow(window_name, processed_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    frame_count += 1
                    pbar.update(1)
                    
                    # 每100帧打印一次统计信息
                    if frame_count % 100 == 0:
                        print(f"已处理 {frame_count}/{total_frames} 帧, 检测到姿态数: {len(pose_results)}")
                        
        finally:
            # 释放资源
            cap.release()
            if save_result:
                out.release()
            cv2.destroyAllWindows()
            
        print(f"姿态检测处理完成! 输出文件: {output_path}")
        return output_path

# 独立函数接口
def process_pose_detection(image, model=None, mask_separation=False, min_confidence=0.25):
    """独立函数：处理单帧图像的姿态检测
    
    Args:
        image: 输入图像
        model: YOLO模型
        mask_separation: 是否使用掩码分离
        min_confidence: 最小置信度
        
    Returns:
        tuple: (姿态检测结果列表, 处理后的图像)
    """
    processor = YOLOPoseProcessor()
    if model is not None:
        processor.model = model
    
    return processor.process_pose_detection(image, mask_separation, min_confidence)

def process_video_pose(video_path, output_path='output_pose.mp4', 
                      model_path='yolo11s-pose.pt', mask_separation=False, show_preview=True):
    """独立函数：处理视频文件进行姿态检测
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
        model_path: 模型路径
        mask_separation: 是否使用掩码分离
        show_preview: 是否显示预览
        
    Returns:
        str: 输出文件路径
    """
    processor = YOLOPoseProcessor(model_path)
    return processor.process_video(video_path, output_path, mask_separation, show_preview)

if __name__ == '__main__':
    # 测试示例
    video_path = 'test_video/test33.mp4'
    
    if os.path.exists(video_path):
        # 使用类的方式 - 掩码分离模式
        processor = YOLOPoseProcessor('weights/yolo11s-pose.pt')
        processor.process_video(video_path, 'test_video/output/pose_mask_output.mp4', 
                               mask_separation=True, show_preview=True)
        
        # 使用类的方式 - 叠加模式
        processor.process_video(video_path, 'test_video/output/pose_overlay_output.mp4', 
                               mask_separation=False, show_preview=False)
        
        # 使用独立函数的方式
        process_video_pose(video_path, 'test_video/output/pose_function_output.mp4', 
                          mask_separation=True, show_preview=False)
        
        print("姿态检测测试完成!")
    else:
        print(f"测试视频文件不存在: {video_path}")
        print("请确保 test_video/test33.mp4 文件存在于当前目录")