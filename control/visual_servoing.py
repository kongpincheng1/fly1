# visual_servoing.py

import cv2
from ultralytics import YOLO
from enum import Enum
import math
import numpy as np
import os
import time # 导入 time 模块

# 这个类中的状态只关心视觉任务本身
class VisualState(Enum):
    GLOBAL_SEARCH = 1
    CENTERING = 2
    TARGET_LOCKED = 3
    LOST = 4

class VisualServoingController:
    """
    一个独立的视觉伺服控制器类。
    它负责处理图像、运行YOLO模型，并根据其内部状态返回指令。
    新增功能：可以根据设置，在处理图像时定期保存照片。
    """
    def __init__(self, model_path, confidence_threshold=0.5, 
                 center_tolerance_px=25, 
                 # 拍照功能参数
                 enable_photo_capture: bool = False, 
                 photo_save_path: str = '/tmp/drone_captures',
                 photo_capture_interval: int = 30,
                 # <<< 新增：视频录制相关的参数 >>>
                 enable_video_recording: bool = False,
                 video_save_path: str = '/tmp/drone_videos',
                 video_filename: str = 'output.avi',
                 video_fps: float = 30.0):
        """
        初始化视觉控制器。
        :param enable_photo_capture: bool, 是否启用拍照功能。
        :param photo_save_path: str, 照片保存的目录路径。
        :param photo_capture_interval: int, 每隔多少帧拍一张照片。
        """
        print("视觉控制器：对象已创建，模型待加载。")
        self.model_path = model_path
        self.model = None
        self.is_model_loaded = False

        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.visual_state = VisualState.GLOBAL_SEARCH
        self.current_target_label = None
        self.initial_target_map = {}
        self.CENTER_TOLERANCE_PX = center_tolerance_px

        # === 新增：拍照功能相关的实例变量 ===
        self.enable_photo_capture = enable_photo_capture
        self.photo_save_path = photo_save_path
        self.photo_capture_interval = photo_capture_interval
        self.frame_counter = 0

        # 如果启用拍照，则创建保存目录
        if self.enable_photo_capture:
            os.makedirs(self.photo_save_path, exist_ok=True)
            print(f"拍照功能已启用，照片将保存到: {self.photo_save_path}")
        
        # <<< 新增：视频录制相关的实例变量 >>>
        self.enable_video_recording = enable_video_recording
        self.video_writer = None  # 先初始化为None
        self.video_fps = video_fps
        self.video_full_path = None
        if self.enable_video_recording:
            # 确保目录存在
            os.makedirs(video_save_path, exist_ok=True)
            self.video_full_path = os.path.join(video_save_path, video_filename)
            print(f"视频录制功能已启用，视频将保存为: {self.video_full_path}")
    
    # ... (load_model, reset_for_new_mission, set_target, _find_active_target, _get_drone_command 方法保持不变) ...
    def load_model(self):
        """
        一个独立的方法，专门用于加载模型。
        这个方法应该在ROS节点进入主循环后调用。
        """
        if self.is_model_loaded:
            print("视觉控制器：模型已经加载过了。")
            return
        
        try:
            print("视觉控制器：正在加载模型...")
            self.model = YOLO(self.model_path)
            # 在这里可以进行一次虚拟推理来预热GPU
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False) 
            self.is_model_loaded = True
            print("视觉控制器：模型加载并预热成功。")
            return True
        except Exception as e:
            print(f"视觉控制器：加载模型失败！错误: {e}")
            self.is_model_loaded = False
            return False
    
    def reset_for_new_mission(self):
        """重置整个视觉任务，回到最初的全局搜索状态。"""
        print("视觉控制器：任务重置，返回全局搜索。")
        self.visual_state = VisualState.GLOBAL_SEARCH
        self.current_target_label = None
        self.initial_target_map = {}
        self.last_detection_count = 0 # 同样重置

    # <<< 新增：一个方法，用于在两次投放之间重置状态 >>>
    def reset_to_search_mode(self):
        """
        将控制器重置回搜索模式，以便重新确认所有目标。
        这个方法不会清除 initial_target_map。
        """
        print("视觉控制器：重置为搜索模式，以重新确认所有目标。")
        self.visual_state = VisualState.GLOBAL_SEARCH
        self.current_target_label = None # 清除当前特定目标

    # <<< 新增：一个getter方法，用于获取检测到的目标数 >>>
    def get_current_detection_count(self) -> int:
        """返回上一帧处理中检测到的目标数量。"""
        return self.last_detection_count

    def set_target(self, target_label):
        """
        从外部（ROS节点）设置要追踪的目标。
        这将使视觉状态切换到CENTERING。
        """
        if not self.initial_target_map:
            print("错误：在设置目标前，必须先完成全局搜索！")
            return
        print(f"视觉控制器：已设置新目标 [{target_label}]，开始对准。")
        self.current_target_label = target_label
        self.visual_state = VisualState.CENTERING

    def _find_active_target(self, detections, image_width):
        """（内部方法）根据策略找到当前要追踪的目标。"""
        if not detections:
            return None
        
        target_label = self.current_target_label
        if target_label == "Middle":
            if len(detections) == 3:
                return sorted(detections, key=lambda d: d['center'][0])[1]
            else:
                return min(detections, key=lambda d: abs(d['center'][0] - image_width / 2))
        elif target_label == "Left":
            return min(detections, key=lambda d: d['center'][0])
        elif target_label == "Right":
            return max(detections, key=lambda d: d['center'][0])
        return None

    def _get_drone_command(self, target_center, image_center):
        """（内部方法）生成中文指令。"""
        tx, ty = target_center
        cx, cy = image_center
        dx = tx - cx
        dy = ty - cy

        if abs(dx) <= self.CENTER_TOLERANCE_PX and abs(dy) <= self.CENTER_TOLERANCE_PX:
            return "位置锁定，准备投放"
        
        command = []
        if dx > self.CENTER_TOLERANCE_PX: command.append("向右平移")
        elif dx < -self.CENTER_TOLERANCE_PX: command.append("向左平移")
        if dy > self.CENTER_TOLERANCE_PX: command.append("向后平移")
        elif dy < -self.CENTER_TOLERANCE_PX: command.append("向前平移")
        return " & ".join(command)
        
        # <<< 新增：清理方法，用于安全关闭视频文件 >>>
    def cleanup(self):
        """在程序结束时调用，用于释放资源，特别是关闭视频写入器。"""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"视频文件已成功保存并关闭: {self.video_full_path}")

    def process_frame(self, frame):
        """
        处理单帧图像的核心方法。
        返回: (visual_state, command, annotated_frame)
        """
        # 在处理第一帧前，确保模型已加载
        if not self.is_model_loaded:
            print("错误：在处理图像前，模型尚未加载！")
            return self.visual_state, None, frame
        
        # === 新增：帧计数和拍照逻辑 ===
        self.frame_counter += 1
        if self.enable_photo_capture and (self.frame_counter % self.photo_capture_interval == 0):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}_{self.frame_counter}.jpg"
            filepath = os.path.join(self.photo_save_path, filename)
            cv2.imwrite(filepath, frame)
            # 你可以在这里使用 self.get_logger().info() 如果这个类能访问到logger
            # 但作为一个独立的类，print更通用
            print(f"已保存照片: {filepath}")

        height, width, _ = frame.shape
        image_center = (width // 2, height // 2)
        
        # 1. 检测 (保持不变)
        results = self.model(frame, verbose=False)
        detections = []
        for box in results[0].boxes:
            if box.conf[0] > self.CONFIDENCE_THRESHOLD :
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                detections.append({'center': (cx, cy), 'box': [x1, y1, x2, y2]})
        # <<< 新增：更新检测到的目标数量 >>>
        self.last_detection_count = len(detections)
        # 2. 视觉状态机逻辑 (保持不变)
        command = None
        
        if self.visual_state == VisualState.GLOBAL_SEARCH:
            if not self.initial_target_map and len(detections) == 3:
                print("视觉控制器：全局搜索成功，已识别3个目标。")
                detections.sort(key=lambda d: d['center'][0])
                self.initial_target_map = {"Left": detections[0], "Middle": detections[1], "Right": detections[2]}
            
        elif self.visual_state == VisualState.CENTERING:
            if not detections:
                self.visual_state = VisualState.LOST
                command = "丢失目标"
            else:
                active_target = self._find_active_target(detections, width)
                if active_target:
                    command = self._get_drone_command(active_target['center'], image_center)
                    if "位置锁定" in command:
                        self.visual_state = VisualState.TARGET_LOCKED
                else:
                    self.visual_state = VisualState.LOST
                    command = "丢失目标"

        # 3. 可视化 (保持不变)
        annotated_frame = frame.copy() # 复制一份以进行绘制，避免影响原始图像保存
        for det in detections:
            cv2.rectangle(annotated_frame, (det['box'][0], det['box'][1]), (det['box'][2], det['box'][3]), (0, 255, 0), 2)
        
        if self.visual_state == VisualState.CENTERING and detections:
             active_target = self._find_active_target(detections, width)
             if active_target:
                cv2.rectangle(annotated_frame, (active_target['box'][0], active_target['box'][1]), (active_target['box'][2], active_target['box'][3]), (0, 255, 255), 3)
                cv2.putText(annotated_frame, f"Tracking: {self.current_target_label}", (active_target['box'][0], active_target['box'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        cv2.rectangle(annotated_frame, (image_center[0] - self.CENTER_TOLERANCE_PX, image_center[1] - self.CENTER_TOLERANCE_PX), (image_center[0] + self.CENTER_TOLERANCE_PX, image_center[1] + self.CENTER_TOLERANCE_PX), (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Visual State: {self.visual_state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # <<< 新增：视频写入逻辑 >>>        
        if self.enable_video_recording:
            # 如果是第一帧，则初始化VideoWriter
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG') # MJPG适用于.avi格式
                frame_size = (width, height)
                self.video_writer = cv2.VideoWriter(self.video_full_path, fourcc, self.video_fps, frame_size)
                print(f"视频录制已开始... 尺寸:{frame_size}, FPS:{self.video_fps}")
            
            # 将不带有标注的帧写入视频文件
            self.video_writer.write(frame)

        return self.visual_state, command, annotated_frame
