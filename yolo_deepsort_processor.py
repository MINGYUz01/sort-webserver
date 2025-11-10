import cv2
import os
import torch
from ultralytics import YOLO
from tqdm import tqdm
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import numpy as np

# DeepSort 跟踪器配置
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")

class YOLODeepSORTProcessor:
    """YOLO + DeepSORT 目标检测与跟踪处理器"""
    
    def __init__(self, model_path='weights/yolo11s.pt', use_cuda=False):
        """初始化处理器
        
        Args:
            model_path: YOLO模型路径
            use_cuda: 是否使用GPU
        """
        self.model = YOLO(model_path)
        self.deepsort = self._initialize_deepsort(use_cuda)
        self.track_history = {}
        
    def _initialize_deepsort(self, use_cuda):
        """初始化DeepSort跟踪器"""
        return DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=use_cuda
        )
    
    def process_detection_results(self, image, min_confidence=0.25):
        """处理单帧图像的目标检测和跟踪
        
        Args:
            image: 输入图像 (BGR格式)
            min_confidence: 最小置信度阈值
            
        Returns:
            tuple: (检测结果列表, 处理后的图像)
        """
        # 复制图像用于绘制
        processed_image = image.copy()
        
        # 使用YOLO进行目标检测
        detection_results = self._yolo_detection(image, min_confidence)
        
        # 使用DeepSort进行目标跟踪
        tracking_results = self._deepsort_tracking(detection_results, processed_image)
        
        # 在图像上绘制结果
        self._draw_detections(processed_image, tracking_results)
        
        return tracking_results, processed_image
    
    def _yolo_detection(self, image, min_confidence):
        """YOLO目标检测
        
        Args:
            image: 输入图像
            min_confidence: 最小置信度
            
        Returns:
            list: 检测结果 [(x1, y1, x2, y2, class_id, confidence), ...]
        """
        results = self.model(image)
        detections = []
        
        for result in results:
            for detection in result.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = detection
                
                if confidence > min_confidence:
                    detections.append((
                        int(x1), int(y1), int(x2), int(y2), 
                        int(class_id), float(confidence)
                    ))
        
        return detections
    
    def _deepsort_tracking(self, detections, image):
        """DeepSort目标跟踪
        
        Args:
            detections: YOLO检测结果
            image: 当前帧图像
            
        Returns:
            list: 跟踪结果 [(x1, y1, x2, y2, class_name, track_id), ...]
        """
        # 准备DeepSort输入格式
        bbox_xywh = []
        confidences = []
        class_ids = []
        
        for x1, y1, x2, y2, class_id, confidence in detections:
            # 转换为(x_center, y_center, width, height)格式
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            
            bbox_xywh.append([center_x, center_y, width, height])
            confidences.append(confidence)
            class_ids.append(class_id)
        
        # 转换为tensor
        bbox_tensor = torch.Tensor(bbox_xywh)
        conf_tensor = torch.Tensor(confidences)
        
        # 更新跟踪器
        outputs = self.deepsort.update(bbox_tensor, conf_tensor, class_ids, image)
        
        # 处理跟踪结果
        tracking_results = []
        for output in outputs:
            x1, y1, x2, y2, class_id, track_id = output
            
            # 获取类别名称
            class_name = self.model.names.get(int(class_id), "unknown")
            
            tracking_results.append((
                int(x1), int(y1), int(x2), int(y2), 
                class_name, int(track_id)
            ))
        
        return tracking_results
    
    def _draw_detections(self, image, tracking_results):
        """在图像上绘制检测和跟踪结果
        
        Args:
            image: 要绘制的图像
            tracking_results: 跟踪结果
        """
        colors = {
            'car': (0, 255, 0),      # 绿色
            'person': (255, 0, 0),   # 蓝色
            'bicycle': (0, 0, 255),  # 红色
            'default': (255, 255, 0) # 黄色
        }
        
        for x1, y1, x2, y2, class_name, track_id in tracking_results:
            # 根据类别选择颜色
            color = colors.get(class_name, colors['default'])
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            label = f"{class_name} ID:{track_id}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # 标签背景
            cv2.rectangle(image, (x1, y1 - text_size[1] - 4), 
                         (x1 + text_size[0], y1), color, -1)
            
            # 标签文字
            cv2.putText(image, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_video(self, video_path, output_path='output_video.mp4', 
                     show_preview=False, save_result=True):
        """处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
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
        
        print(f"开始处理视频: {video_path}")
        print(f"视频信息: {width}x{height}, {fps:.1f} FPS, 总帧数: {total_frames}")
        
        frame_count = 0
        
        try:
            with tqdm(total=total_frames, desc="处理进度") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 处理当前帧
                    tracking_results, processed_frame = self.process_detection_results(frame)
                    
                    # 保存处理结果
                    if save_result:
                        out.write(processed_frame)
                    
                    # 显示预览
                    if show_preview:
                        cv2.imshow('YOLO+DeepSORT 目标跟踪', processed_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    frame_count += 1
                    pbar.update(1)
                    
                    # 每100帧打印一次统计信息
                    if frame_count % 100 == 0:
                        print(f"已处理 {frame_count}/{total_frames} 帧, 当前跟踪目标数: {len(tracking_results)}")
                        
        finally:
            # 释放资源
            cap.release()
            if save_result:
                out.release()
            cv2.destroyAllWindows()
            
        print(f"视频处理完成! 输出文件: {output_path}")
        return output_path

def process_detection_results(image, model=None, deepsort=None, min_confidence=0.25):
    """独立函数：处理单帧图像的目标检测和跟踪
    
    Args:
        image: 输入图像
        model: YOLO模型
        deepsort: DeepSort跟踪器
        min_confidence: 最小置信度
        
    Returns:
        tuple: (检测结果列表, 处理后的图像)
    """
    processor = YOLODeepSORTProcessor()
    if model is not None:
        processor.model = model
    if deepsort is not None:
        processor.deepsort = deepsort
    
    return processor.process_detection_results(image, min_confidence)

def process_video(video_path, output_path='output_video.mp4', 
                 model_path='yolo11s.pt', show_preview=True):
    """独立函数：处理视频文件
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
        model_path: 模型路径
        show_preview: 是否显示预览
        
    Returns:
        str: 输出文件路径
    """
    processor = YOLODeepSORTProcessor(model_path)
    return processor.process_video(video_path, output_path, show_preview)

if __name__ == '__main__':
    # 测试示例
    video_path = 'test_video/test33.mp4'
    
    if os.path.exists(video_path):
        # 使用类的方式
        processor = YOLODeepSORTProcessor('weights/best.pt')
        processor.process_video(video_path, 'test_video/output/class_output.mp4', show_preview=True)
        
        # 使用独立函数的方式
        process_video(video_path, 'test_video/output/function_output.mp4', show_preview=False)
        
        print("测试完成!")
    else:
        print(f"测试视频文件不存在: {video_path}")
        print("请确保 test_video/test33.mp4 文件存在于当前目录")

