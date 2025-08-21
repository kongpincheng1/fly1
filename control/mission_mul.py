#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleOdometry
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from collections import deque
import time
from control.DronePositionChecker import DronePositionChecker
from control.AlignmentChecker import AlignmentChecker
from control.ServoControl import ServoControl
from control.visual_servoing import VisualServoingController, VisualState # 从你的包中导入视觉控制器
import cv2
from enum import Enum
import subprocess
import re
import os
import csv
CAMERA_NAME_HINT = "imx577"


class MissionState(Enum):
    START = 0
    TAKING_OFF = 1
    GLOBAL_SEARCH = 2
    TARGETING_CYCLE = 3
    RECONFIRMING_TARGETS = 3.5  # <<< 新增的状态
    LANDING = 4
    MISSION_COMPLETE = 5

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.target_position_subscriber = self.create_subscription(Point, '/target_position',
                                                                   self.target_position_callback, 10)

        base_photo_path = '/home/image_recodes'
        # Create a folder name based on the current date and time (e.g., 'run_20230727_153000')
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_photo_path = os.path.join(base_photo_path, f"run_{run_timestamp}")
        self.get_logger().info(f"This run's photos will be saved to: {unique_photo_path}")
        
        # <<< 新增：视频路径和文件名 >>>
        base_video_path = '/home/video_recodes' # 你可以指定一个新的文件夹
        unique_video_filename = f"mission_{run_timestamp}.avi" # AVI格式与MJPG编码器配合良好        
        
        # === 初始化视觉部分 (带视频录制功能) ===
        self.vision_controller = VisualServoingController(
            model_path='/home/weights/0711.engine',
            # 拍照功能
            enable_photo_capture=False,
            photo_save_path=unique_photo_path, 
            photo_capture_interval=30,
            # <<< 新增：启用并配置视频录制 >>>
            enable_video_recording=True,           # 设置为 True 来开启录制
            video_save_path=base_video_path,       # 视频保存的目录
            video_filename=unique_video_filename,  # 带有时间戳的唯一文件名
            video_fps=30.0                         # 视频帧率 (与你的timer频率匹配)
        )
        
        device_path = self.find_video_device_by_name(CAMERA_NAME_HINT)
        self.cap = cv2.VideoCapture(device_path if device_path else 0)        
        if not self.cap.isOpened():
            self.get_logger().error("无法打开摄像头！")
            rclpy.shutdown()
        
        self.is_vision_ready = False

        # === 新增：任务流程管理变量 ===
        self.mission_state = MissionState.GLOBAL_SEARCH
        self.target_priority = ["Middle", "Left", "Right"]
        self.current_target_index = 0
        self.visited_targets_count = 0
        #=========================================================

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        

        self.target_position = None
        self.CurrentHeightFromCamera = 0.0
        
        self.already_reached = False
        self.first_aligned = False
        self.second_aligned = False
        self.last_found_x_NED = None
        self.last_found_y_NED = None
        self.last_found_z_NED = None


        #起飞高度
        self.takeoff_height = -2.3
        #向前飞行的距离
        self.forward_x = 32.5
        #最大步长
        self.align_maxstep = 0.2
        #对准之后下降的高度
        self.afterAlign_descentHeight = 0.7
        #GLOBAL_SEARCH高度
        self.global_search_height = -5.0
        self.global_search_target_z = None

        self.initial_z = None  # 初始高度
        self.initial_x = None  #
        self.initial_y = None
        self.init_yaw = None

        self.DropArea_x = None
        self.DropArea_y = None

        self.takeoff_target_height = None
        self.is_ReadyToTakeoff = False
        self.is_AtTakeoffHeight = False
        self.is_AtDropArea = False
        self.is_FinishDrop = False
        self.Is_Finish_1st_Drop = False
        self.Is_Finish_2nd_Drop = False

        self.Is_Descending_to_depth_camera_height = False

        self.droping_x = None
        self.droping_y = None
        self.droping_z = None

        # 新增日志计数器，用于减少日志输出频率
        self.log_counter = 0

        self.first_alignment_complete = False
        self.second_alignment_complete = False

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.03, self.timer_callback)
        
        #初始化位置判断器
        self.initPositionChecker = DronePositionChecker(
            logger_func=self.get_logger().info,
            tolerance=0.17, 
            duration=5.0
        )

          # 初始化 AlignmentChecker
        self.first_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,  # 传递日志记录函数
            threshold=0.15,
            time_window=2.0,
            check_frequency=5
        )
        self.second_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,  # 传递日志记录函数
            threshold=0.10,
            time_window=2.0,
            check_frequency=5
        )
        # 初始化舵机控制器
        self.servo_control = ServoControl()
        
        # ========== PID控制参数设置区域 ==========
        # 📌 饱和P控制参数（大误差阶段）
        self.align_maxstep = 0.2  # 最大步长限制 (米) - 可调参数
        
        # 📌 细调阶段PID参数（小误差阶段）
        self.epsilon = self.align_maxstep  # 切换阈值 (0.2m) - 可调参数
        self.Kp_fine = 0.9911  # P增益 - 可调参数 (建议范围: 1.0-2.5)
        self.Ki = 0.1021       # I增益 - 可调参数 (建议范围: 0.1-0.8)
        self.Kd = 0.0009      # D增益 - 可调参数 (建议范围: 0.2-0.8)
        
        # 📌 PID状态变量
        self.integral_x = 0.0      # X方向积分项
        self.integral_y = 0.0      # Y方向积分项
        self.last_error_x = 0.0    # 上次X误差 (用于微分计算)
        self.last_error_y = 0.0    # 上次Y误差 (用于微分计算)
        self.dt = 0.03             # 控制周期 (秒) - 与timer频率一致
        
        # 📌 积分限幅参数
        self.max_integral = self.epsilon  # 积分限幅值 - 可调参数
        # =========================================

        # ========== 目标像素坐标日志自定义路径 ==========
        custom_dir = '/Users/jihaobi/cqufly/fly3/mylog'  # <--- 你可以自定义
        custom_filename = 'my_bucket_log_20240601.csv'   # <--- 你可以自定义
        os.makedirs(custom_dir, exist_ok=True)
        self.pixel_log_path = os.path.join(custom_dir, custom_filename)
        # 如果文件不存在，写入表头
        if not os.path.exists(self.pixel_log_path):
            with open(self.pixel_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'target_x', 'target_y'])
        # =========================================


    def target_position_callback(self, msg: Point):
        """Callback function for receiving target position."""
        self.target_position = msg  


    def current_height_callback(self, msg):
        self.CurrentHeightFromCamera = msg.data

    def fly_to_position(self, x, y, z):
        """Fly to the specified position."""
        self.publish_position_setpoint(x, y, z)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = self.init_yaw  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

        # <<< 新增：重写 destroy_node 方法以进行清理 >>>
    def destroy_node(self):
        """在节点关闭前，执行必要的清理工作。"""
        self.get_logger().info("节点正在关闭，执行清理程序...")
        # 清理视觉控制器（保存视频）
        if self.vision_controller:
            self.vision_controller.cleanup()
        # 清理摄像头
        if self.cap and self.cap.isOpened():
            self.cap.release()
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        # 调用父类的方法完成ROS节点的销毁
        super().destroy_node()
        self.get_logger().info("清理完成，节点已关闭。")
    
    def find_video_device_by_name(self,name_hint="USB Camera"):
    # (This function remains unchanged)
        try:
            result = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError): return None
        lines = result.stdout.splitlines()
        matched_device_name = False
        for line in lines:
            if name_hint in line: matched_device_name = True
            elif matched_device_name and "/dev/video" in line:
                match = re.search(r"(/dev/video\d+)", line)
                if match: return match.group(1)
        return None

    def execute_visual_command(self, command):
        """根据视觉指令来控制无人机"""
        if command is None:
            return

        # 简单的比例控制，将指令转换为小的位置增量
        step_size_xy = 0.3  # 水平移动步长
        step_size_z = 0.0   # 这里我们只做水平调整

        current_x, current_y = self.coordinate_NED2FRD(self.vehicle_local_position.x, self.vehicle_local_position.y)
        
        delta_x, delta_y = 0.0, 0.0
        if "向右平移" in command: delta_y = step_size_xy  #??????
        if "向左平移" in command: delta_y = -step_size_xy
        if "向前平移" in command: delta_x = step_size_xy
        if "向后平移" in command: delta_x = -step_size_xy

        # 计算新的FRD目标点
        target_x_frd = current_x + delta_x
        target_y_frd = current_y + delta_y
        
        # 转换回NED并发布
        target_x_ned, target_y_ned = self.coordinate_FRD2NED(target_x_frd, target_y_frd)
        self.publish_position_setpoint(target_x_ned, target_y_ned, self.global_search_target_z)    
    
    def drop_payload(self,servo_1,servo_2):
        self.servo_control.open_servo(servo_1,servo_2)

        self.get_logger().info("---------------Payload dropped.-------------------")

    def takeoff_relative(self): # 不再需要 relative_height 参数
        """
        飞向预先计算好的目标起飞高度。
        这个函数假定 self.takeoff_target_height 和 self.init_yaw 等已经被设置。
        """
        if self.takeoff_target_height is None:
            self.get_logger().error("takeoff_relative 被调用，但目标起飞高度未设置！")
            return
        
        # 直接命令无人机飞到（初始x, 初始y, 目标z）
        # fly_to_position_FRD2NED 会自动使用 self.initial_x, self.initial_y, self.init_yaw
        self.fly_to_position_FRD2NED(0.0, 0.0, self.takeoff_target_height)

    def takeoff_height_check(self, threshold=0.22):
        """
        检查是否到达相对目标高度
        :param threshold: 高度误差阈值
        :return: True 如果到达目标高度，否则 False
        """
        if self.takeoff_target_height is None:
            self.get_logger().warn("目标高度尚未设置！")
            return False
        current_height = self.vehicle_local_position.z
        height_error = abs(current_height - self.takeoff_target_height)
        # 为了减少日志输出，只有每隔一定周期时才打印此日志
        if self.log_counter % 10 == 0:
            self.get_logger().info(f"当前高度：{current_height:.2f} 米，目标高度：{self.takeoff_target_height:.2f} 米，高度误差：{height_error:.2f} 米")
        if height_error < threshold:
            self.is_AtTakeoffHeight = True

    def fly_forward(self, x):
        """Fly forward to the drop area."""
        self.DropArea_x, self.DropArea_y = self.fly_to_position_FRD2NED(x, 0, self.takeoff_target_height)
        
    def fly_forward_check(self, threshold=0.2):
        """Check if the drone has reached the drop area."""
        current_x = self.vehicle_local_position.x
        current_y = self.vehicle_local_position.y
        error = math.sqrt((current_x - self.DropArea_x)**2 + (current_y - self.DropArea_y)**2)
        if self.log_counter % 10 == 0:
            self.get_logger().info(f"--当前x：{current_x:.2f},y:{current_y:.2f}米，--目标x：{self.DropArea_x:.2f} 米，y:{self.DropArea_y:.2f},--误差：{error:.2f} 米")
        if error < threshold:
            self.is_AtDropArea = True
    
    def first_alignment_check(self, target_x, target_y):
        """Check first alignment with the target."""
        current_x = self.vehicle_local_position.x
        current_y = self.vehicle_local_position.y
        is_align_now = self.first_alignment_checker.check(
            current_x,
            current_y,
            target_x=target_x,
            target_y=target_y
)       
        if is_align_now:
            self.first_alignment_complete = True
            self.second_alignment_checker.reset()
            self.get_logger().info("------------------------first对准完成！------------------------")

    def second_alignment_check(self, target_x, target_y):
        """Check second alignment with the target."""
        is_align_now = self.second_alignment_checker.check(
    current_x=self.vehicle_local_position.x,
    current_y=self.vehicle_local_position.y,
            target_x=target_x,
            target_y=target_y
        )
        if is_align_now:
            self.second_alignment_complete = True
            self.get_logger().info("-------------------------second对准完成！------------------------")

    def fly_to_position_FRD2NED(self,x,y,z):
        '''
        通过旋转矩阵, 将FRD坐标系转换为NED坐标系。再根据初始误差增加平移矩阵。

        '''
        x_target = x*math.cos(self.init_yaw)-y*math.sin(self.init_yaw) + self.initial_x
        y_target = x*math.sin(self.init_yaw)+y*math.cos(self.init_yaw) + self.initial_y
        z_target = z
        self.publish_position_setpoint(x_target, y_target, z_target)
        if self.log_counter % 10 == 0:
            self.get_logger().info(f"Flying to FRDposition: x={x}, y={y}, z={z}")
        return x_target, y_target

    def coordinate_NED2FRD(self,x_NED,y_NED):
        '''
        将NED坐标转换为FRD坐标。
        '''
        x_FRD = (x_NED-self.initial_x)*math.cos(self.init_yaw)+(y_NED-self.initial_y)*math.sin(self.init_yaw)
        y_FRD = -(x_NED-self.initial_x)*math.sin(self.init_yaw)+(y_NED-self.initial_y)*math.cos(self.init_yaw)
        return x_FRD, y_FRD

    def coordinate_FRD2NED(self,x,y):
        '''
        将FRD坐标转换为NED坐标。
        '''
        x_target = x*math.cos(self.init_yaw)-y*math.sin(self.init_yaw) + self.initial_x
        y_target = x*math.sin(self.init_yaw)+y*math.cos(self.init_yaw) + self.initial_y

        return x_target, y_target

    def adjust_to_target(self):
        """Adjust drone position towards the current target using PID control."""
        if self.target_position:
            # ========== pid控制实现，记录目标像素坐标 ========== 
            import time
            with open(self.pixel_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(),
                    self.target_position.x,
                    self.target_position.y
                ])
            # ========== 原有控制逻辑 ==========
            # 获取当前位置
            current_xned, current_yned = self.vehicle_local_position.x, self.vehicle_local_position.y
            current_x, current_y = self.coordinate_NED2FRD(current_xned, current_yned)
            
            # 📌 计算相机坐标系误差（考虑相机中心偏移）
            dx_cam = self.target_position.y - 0.065  # 相机中心相对投放中心的Y偏差
            dy_cam = -self.target_position.x + 0.033        # 相机中心相对投放中心的X偏差
            distance = math.hypot(dx_cam, dy_cam)
            
            # 📌 根据误差大小选择控制策略
            if distance < self.epsilon:
                # ——————— 细调阶段：PID控制 ———————
                if self.log_counter % 10 == 0:
                    self.get_logger().info(f"PID细调阶段 - 误差:{distance:.3f}m < 阈值:{self.epsilon:.3f}m")
                
                # 计算误差项
                error_x = dx_cam
                error_y = dy_cam
                
                # 📌 积分项计算（带限幅防饱和）
                self.integral_x += error_x * self.dt
                self.integral_y += error_y * self.dt
                # 积分限幅
                self.integral_x = max(min(self.integral_x, self.max_integral), -self.max_integral)
                self.integral_y = max(min(self.integral_y, self.max_integral), -self.max_integral)
                
                # 📌 微分项计算
                derivative_x = (error_x - self.last_error_x) / self.dt
                derivative_y = (error_y - self.last_error_y) / self.dt
                
                # 📌 PID控制量计算
                control_x = (self.Kp_fine * error_x + 
                           self.Ki * self.integral_x + 
                           self.Kd * derivative_x)
                control_y = (self.Kp_fine * error_y + 
                           self.Ki * self.integral_y + 
                           self.Kd * derivative_y)
                
                # 保存本次误差用于下次微分计算
                self.last_error_x = error_x
                self.last_error_y = error_y
                
                if self.log_counter % 10 == 0:
                    self.get_logger().info(f"PID输出: P={self.Kp_fine * error_x:.3f}, I={self.Ki * self.integral_x:.3f}, D={self.Kd * derivative_x:.3f}")
                
            else:
                # ——————— 大误差阶段：饱和P控制 ———————
                if self.log_counter % 10 == 0:
                    self.get_logger().info(f"饱和P控制阶段 - 误差:{distance:.3f}m >= 阈值:{self.epsilon:.3f}m")
                
                # 📌 饱和比例控制
                scale = self.align_maxstep / distance
                control_x = dx_cam * scale
                control_y = dy_cam * scale
                
                # 清零PID状态，避免积累
                self.integral_x = 0.0
                self.integral_y = 0.0
                self.last_error_x = 0.0
                self.last_error_y = 0.0
                
                if self.log_counter % 10 == 0:
                    self.get_logger().info(f"饱和P输出: scale={scale:.3f}, 最大步长={self.align_maxstep:.3f}m")
            
            # ============== 目标位置计算 ==============
            # 计算FRD目标位置
            target_x_FRD = current_x + control_x
            target_y_FRD = current_y + control_y
            
            # 转换为NED坐标
            target_x_NED, target_y_NED = self.coordinate_FRD2NED(target_x_FRD, target_y_FRD)
            
            # 精确目标位置（用于对准检查）
            precise_target_x_FRD = current_x + dx_cam
            precise_target_y_FRD = current_y + dy_cam
            precise_target_x_NED, precise_target_y_NED = self.coordinate_FRD2NED(precise_target_x_FRD, precise_target_y_FRD)
            
            # ============== 两次对准逻辑 ==============
            # First alignment
            if not self.first_alignment_complete:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("执行第一次对准")
                self.fly_to_position(target_x_NED, target_y_NED, self.takeoff_target_height)
                self.first_alignment_check(precise_target_x_NED, precise_target_y_NED)
                self.last_found_x_NED = target_x_NED
                self.last_found_y_NED = target_y_NED
                self.last_found_z_NED = self.takeoff_target_height

            elif self.first_alignment_complete and not self.second_alignment_complete:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("执行第二次精确对准")
                self.fly_to_position(target_x_NED, target_y_NED, self.takeoff_target_height + self.afterAlign_descentHeight)
                self.second_alignment_check(precise_target_x_NED, precise_target_y_NED)
                self.last_found_x_NED = target_x_NED
                self.last_found_y_NED = target_y_NED
                self.last_found_z_NED = self.takeoff_target_height + self.afterAlign_descentHeight

            self.target_position = None
            
            # ============== 投水逻辑 ==============
            if self.first_alignment_complete and self.second_alignment_complete:
                if not self.Is_Finish_1st_Drop:
                    self.drop_payload(-1.0,1.0)
                    self.get_logger().info("——————————————————————第一次投水————————————————————————")
                    self.Is_Finish_1st_Drop = True
                elif not self.Is_Finish_2nd_Drop:
                    self.drop_payload(1.0,-1.0)
                    self.get_logger().info("——————————————————————第二次投水————————————————————————")
                    self.Is_Finish_2nd_Drop = True
                self.droping_x = self.vehicle_local_position.x
                self.droping_y = self.vehicle_local_position.y
                self.droping_z = self.vehicle_local_position.z

                
        else:
            # ============== 无目标时的处理 ==============
            if self.last_found_x_NED and self.last_found_y_NED and self.last_found_z_NED:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("无新目标，使用上次记录位置")
                self.fly_to_position(self.last_found_x_NED, self.last_found_y_NED, self.last_found_z_NED)
            else:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("无目标记录，返回投水区域")
                self.fly_to_position(self.DropArea_x, self.DropArea_y, self.takeoff_target_height)


    #定时器
    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal()
        
        if not self.is_vision_ready:
            # 只有在第一次进入timer_callback时执行
            if self.vision_controller.load_model():
                self.is_vision_ready = True
                self.get_logger().info("视觉系统准备就绪，开始执行任务逻辑。")
            else:
                self.get_logger().error("视觉系统初始化失败，节点将不执行任务。")
                return # 如果模型加载失败，直接返回，不执行后续逻辑          
        
        # 更新日志计数器
        self.log_counter += 1
        
        # --- 视觉处理部分 ---
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("无法捕获图像")
            return
        
        # 调用视觉控制器处理图像
        visual_state, visual_command, annotated_frame = self.vision_controller.process_frame(frame)
        cv2.imshow("Drone View", annotated_frame)
        cv2.waitKey(1)
                
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()  
            # 仅在日志计数满足条件时打印
            if self.log_counter % 10 == 0:
                self.get_logger().info("try offboard")

        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if not self.is_ReadyToTakeoff:
                if self.initial_x is None:
                    # 第一次进入此状态，记录当前位置为目标保持位置
                    self.initial_x = self.vehicle_local_position.x
                    self.initial_y = self.vehicle_local_position.y
                    self.initial_z = self.vehicle_local_position.z
                    self.init_yaw = self.vehicle_local_position.heading 
                    self.get_logger().info(f"进入Offboard模式，锁定初始位置: x={self.initial_x:.2f}, y={self.initial_y:.2f}, z={self.initial_z:.2f}")

                # 持续发布保持初始位置的指令
                self.publish_position_setpoint(self.initial_x, self.initial_y, self.initial_z)

                # 更新并检查位置稳定性
                current_pos = (
                    self.vehicle_local_position.x,
                    self.vehicle_local_position.y,
                    self.vehicle_local_position.z
                )
                self.initPositionChecker.update_position(current_pos)

                if self.initPositionChecker.is_stable():
                    self.is_ReadyToTakeoff = True
                    self.arm()
                    self.initial_z = self.vehicle_local_position.z
                    self.takeoff_target_height = float(self.initial_z + self.takeoff_height)
                    self.get_logger().info(f"起飞基准高度: {self.initial_z:.2f} m, 目标起飞高度: {self.takeoff_target_height:.2f} m")

            if self.is_ReadyToTakeoff and not self.is_AtTakeoffHeight:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("执行步骤2,上升到指定高度")
                self.takeoff_relative()
                self.takeoff_height_check()
                # self.is_AtTakeoffHeight = False#  测试用

            if self.is_AtTakeoffHeight and not self.is_AtDropArea:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("执行步骤3,飞向投水区")
                self.fly_forward(self.forward_x)
                self.fly_forward_check()
                # self.is_AtDropArea = False #测试用

            if self.is_AtDropArea and not self.is_FinishDrop:
                if self.mission_state == MissionState.GLOBAL_SEARCH:
                    #上升到global——search高度
                    self.global_search_target_z = float(self.initial_z+self.global_search_height)
                    self.publish_position_setpoint(self.DropArea_x, self.DropArea_y, self.global_search_target_z)
                    if self.log_counter % 10 == 0:
                        self.get_logger().info("进行全局搜索")

                    if self.vision_controller.initial_target_map:
                        self.get_logger().info("全局搜索完成，进入目标打击循环。")
                        self.mission_state = MissionState.TARGETING_CYCLE
                
                elif self.mission_state == MissionState.RECONFIRMING_TARGETS: # <<< 新增的处理块
                    self.get_logger().info("正在爬升并重新确认目标位置...")
                    # 命令无人机飞到全局搜索高度
                    self.publish_position_setpoint(self.DropArea_x, self.DropArea_y, self.global_search_target_z)
                    
                    # 检查视觉控制器是否看到了3个目标
                    num_targets_seen = self.vision_controller.get_current_detection_count()
                    if self.log_counter % 10 == 0:
                        self.get_logger().info(f"重新确认中... 当前看到 {num_targets_seen} / 3 个目标")

                    # 当再次看到3个目标时，才真正进入下一个目标的打击流程
                    if num_targets_seen == 3:
                        self.get_logger().info("重新确认成功！已找到所有3个目标。准备攻击下一个目标。")
                        
                        # 重置对准相关的状态，为下一个目标做准备
                        self.first_alignment_complete = False
                        self.second_alignment_complete = False
                        self.Is_Descending_to_depth_camera_height = False
                        self.first_alignment_checker.reset()
                        self.second_alignment_checker.reset()
                        
                        # 转换回目标打击循环状态
                        self.mission_state = MissionState.TARGETING_CYCLE

                # GLOBAL_SEARCH执行一次之后，mission_state状态都为TARGETING_CYCLE
                elif self.mission_state == MissionState.TARGETING_CYCLE:
                    if self.visited_targets_count >= len(self.target_priority):
                        self.mission_state = MissionState.LANDING
                        return
                    current_target_name = self.target_priority[self.current_target_index]

                    if self.vision_controller.visual_state not in [VisualState.CENTERING, VisualState.TARGET_LOCKED]:
                        self.get_logger().info(f"设置新目标: [{current_target_name}]")
                        self.vision_controller.set_target(current_target_name)

                    if visual_state == VisualState.CENTERING:
                        self.execute_visual_command(visual_command)

                    elif visual_state == VisualState.TARGET_LOCKED:
            
                        # 在这里执行下降和投放逻辑
                        if not self.Is_Descending_to_depth_camera_height:
                            self.get_logger().info(f"目标 [{current_target_name}] 已锁定，准备下降。")
                            self.publish_position_setpoint(self.vehicle_local_position.x, self.vehicle_local_position.y, self.takeoff_target_height)                        
                            if abs(self.vehicle_local_position.z - self.takeoff_target_height) < 0.2:
                                self.Is_Descending_to_depth_camera_height = True
                                self.get_logger().info(f"目标 [{current_target_name}] 已锁定，下降完成。")
                        
                        if self.Is_Descending_to_depth_camera_height == True :
                            self.adjust_to_target() 
                            if self.Is_Finish_1st_Drop and self.visited_targets_count == 0:                        
                                # 更新任务进度
                                self.visited_targets_count += 1
                                self.current_target_index += 1
                                
                                if self.visited_targets_count < 2:
                                    # 投放完成，不要直接设置下一个目标！
                                    # 而是进入“重新确认”状态
                                    self.get_logger().info("第一次投放完成。进入目标重新确认阶段。")
                                    self.mission_state = MissionState.RECONFIRMING_TARGETS                                    
                                    # 重置视觉控制器到通用搜索模式
                                    self.vision_controller.reset_to_search_mode()
                                else:
                                    pass
                            if self.Is_Finish_1st_Drop and self.Is_Finish_2nd_Drop:
                                self.is_FinishDrop = True

            if self.is_FinishDrop: 
                self.fly_to_position(float(self.droping_x), float(self.droping_y), float(self.droping_z))
        else:
            self.get_logger().info("启动offboard模式失败")
            
        if self.offboard_setpoint_counter < 30:
            self.offboard_setpoint_counter += 1

def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    try:
        rclpy.spin(offboard_control)
    except KeyboardInterrupt:
        print("程序被用户中断 (Ctrl+C)")
    finally:
        # 确保节点在退出时被正确销毁，从而触发我们的清理逻辑
        offboard_control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
