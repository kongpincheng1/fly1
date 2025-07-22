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
import argparse # <<< 新增
import sys      # <<< 新增

class MissionState(Enum):
    START = 0
    TAKING_OFF = 1
    GLOBAL_SEARCH = 2
    TARGETING_CYCLE = 3
    RECONFIRMING_TARGETS = 3.5
    PROACTIVE_SEARCH = 7
    TIMEOUT_DROP = 8  # <<< 新增的状态
    INMISSION = 4
    MISSION_COMPLETE = 5

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self,args) -> None:
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
                                                                   self.target_position_callback, qos_profile)

        base_photo_path = args.photo_path
        base_video_path = args.video_path
        
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_photo_path = os.path.join(base_photo_path, f"run_{run_timestamp}")
        self.get_logger().info(f"This run's photos will be saved to: {unique_photo_path}")
        unique_video_filename = f"mission_{run_timestamp}.avi" # AVI格式与MJPG编码器配合良好        
        
        # === 初始化视觉部分 (带视频录制功能) ===
        self.vision_controller = VisualServoingController(
            model_path=args.model_path,
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
        
        device_path = self.find_video_device_by_name(args.camera_hint)
        self.cap = cv2.VideoCapture(device_path if device_path else 0)        
        if not self.cap.isOpened():
            self.get_logger().error("无法打开摄像头！")
            rclpy.shutdown()
        
        self.is_vision_ready = False

        # === 新增：任务流程管理变量 ===
        self.mission_state = MissionState.GLOBAL_SEARCH
        self.target_priority = args.target_order 
        self.current_target_index = 0
        self.visited_targets_count = 0
        #=========================================================

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        

        self.target_position = None
        self.last_found_x_NED = None
        self.last_found_y_NED = None
        self.last_found_z_NED = None


        #起飞高度
        self.takeoff_height = args.takeoff_height
        #向前飞行的距离
        self.forward_x = args.forward_x
        # <<< 修改：从命令行参数初始化任务参数 >>>
        self.align_maxstep = args.align_maxstep
        self.afterAlign_descentHeight = args.descent_height
        self.global_search_height = args.search_height
        self.proactive_search_distance = args.proactive_search_dist

        

        # <<< 新增：从命令行参数获取超时和延迟设置 >>>
        self.drop_phase_timeout = args.drop_phase_timeout
        self.search_timeout = args.search_timeout
        self.second_align_maxtime = args.second_align_maxtime

        self.depthcam_xoffset = args.depthcam_xoffset
        self.depthcam_yoffset = args.depthcam_yoffset

        self.trigger_distance = args.trigger_distance
        self.position_threshold = args.position_threshold

        self.global_search_target_z = None

        self.initial_z = None  # 初始高度
        self.initial_x = None  #
        self.initial_y = None
        self.init_yaw = None

        self.DropArea_x = None
        self.DropArea_y = None
        
        # <<< 新增：用于计时超时的状态变量 >>>
        self.drop_phase_start_time = None
        self.second_align_start_timestamp = None
        self.search1_phase_start_time = None
        self.search2_phase_start_time = None
        self.timeout_drop_start_time = None
        self.switch_to_offboard_start_time = None
        self.prepare_offboard_start_time = None
        self.first_drop_delay = None

        self.takeoff_target_height = None
        self.is_ReadyToTakeoff = False
        self.is_AtTakeoffHeight = False
        self.is_AtDropArea = False
        self.is_FinishDrop = False


        self.Is_Descending_to_depth_camera_height = False

        self.droping_x = None
        self.droping_y = None
        self.droping_z = None

        # 新增日志计数器，用于减少日志输出频率
        self.log_counter = 0
        self.timeout_drop_count = 0
        self.timeout_drop_delay = 1.0

        self.first_alignment_complete = False
        self.second_alignment_complete = False

        self.Is_Finish_1st_Drop = False
        self.Is_Finish_2nd_Drop = False

        

        # 为主动搜索阶段设置的状态变量
        self.proactive_target_x = None
        self.proactive_target_y = None
        self.is_proactive_target_set = False # <<< 新增：用于确保目标点只计算一次

        # Create a timer to publish control commands
        self.dt = args.timer_period             # 控制周期 (秒) - 与timer频率一致
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        #初始化位置判断器
        self.initPositionChecker = DronePositionChecker(
            logger_func=self.get_logger().info,
            tolerance=0.17, 
            duration=5.0
        )

          # 初始化 AlignmentChecker
        # <<< 修改：使用命令行参数来初始化 AlignmentChecker >>>
        self.first_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,
            threshold=args.first_align_threshold,
            time_window=args.first_align_time_window,
            check_frequency=args.first_align_check_freq
        )
        self.second_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,
            threshold=args.second_align_threshold,
            time_window=args.second_align_time_window,
            check_frequency=args.second_align_check_freq
        )
        # 初始化舵机控制器
        self.servo_control = ServoControl()
        
        # ========== PID控制参数设置区域 ==========
        # 📌 饱和P控制参数（大误差阶段）
        
        # 📌 细调阶段PID参数（小误差阶段）
        self.epsilon = self.align_maxstep  # 切换阈值 (0.2m) - 可调参数
        self.Kp_fine = args.kp  # P增益 - 可调参数 (建议范围: 1.0-2.5)
        self.Ki = args.ki       # I增益 - 可调参数 (建议范围: 0.1-0.8)
        self.Kd = args.kd
        
        # 📌 PID状态变量
        self.integral_x = 0.0      # X方向积分项
        self.integral_y = 0.0      # Y方向积分项
        self.last_error_x = 0.0    # 上次X误差 (用于微分计算)
        self.last_error_y = 0.0    # 上次Y误差 (用于微分计算)
        
        
        # 📌 积分限幅参数
        self.max_integral = self.epsilon  # 积分限幅值 - 可调参数
        # =========================================

        # ========== 目标像素坐标日志自动生成带时间戳的文件 ==========
        # 生成带时间戳的日志目录和文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = '/Users/jihaobi/cqufly/fly3/mylog'  # 日志目录
        log_filename = f'bucket_pixel_log_{timestamp}.csv'  # 带时间戳的文件名
        os.makedirs(log_dir, exist_ok=True)
        self.pixel_log_path = os.path.join(log_dir, log_filename)
        
        # 创建CSV文件并写入表头
        with open(self.pixel_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'target_x', 'target_y', 'bucket_type', 'alignment_stage'])
        
        self.get_logger().info(f"日志文件已创建: {self.pixel_log_path}")
        # =========================================


    def target_position_callback(self, msg: Point):
        """Callback function for receiving target position."""
        self.target_position = msg  

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

    def start_mission(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=4.0, param2=3.0)
        self.get_logger().info("Switching to Mission mode")

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
        if self.init_yaw is None:
            msg.yaw = 0.00
        else:
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
        if "向右平移" in command: delta_y = step_size_xy  
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
        """Adjust drone position towards the current target."""   
        is_in_second_alignment = self.first_alignment_complete and not self.second_alignment_complete
        
        # 如果正处于第二次对准阶段，无条件检查超时
        if is_in_second_alignment:
            # 启动计时器 (如果尚未启动)
            if self.second_align_start_timestamp is None:
                self.second_align_start_timestamp = self.get_clock().now()
                self.get_logger().info(f"第二次对准开始，启动 {self.second_align_maxtime} 秒超时计时器。")

            # 计算已过时间
            elapsed_drop_time = (self.get_clock().now() - self.second_align_start_timestamp).nanoseconds / 1e9

            # 检查是否超时
            if elapsed_drop_time > self.second_align_maxtime:
                self.get_logger().warn(f"第二次对准超时 ({elapsed_drop_time:.1f}s > {self.second_align_maxtime}s)，强制执行投放！")
                
                # <<< 开始投放逻辑 (从原代码中移动至此) >>>
                if not self.Is_Finish_1st_Drop:
                    self.drop_payload(-1.0, 1.0)
                    self.get_logger().info("——————————————————————DROP (TIMEOUT)————————————————————————")
                    self.Is_Finish_1st_Drop = True
                    self.second_align_start_timestamp = None # 重置计时器
                elif not self.Is_Finish_2nd_Drop:
                    self.drop_payload(1.0, -1.0)
                    self.get_logger().info("——————————————————————DROP (TIMEOUT)————————————————————————")
                    self.Is_Finish_2nd_Drop = True
                    self.second_align_start_timestamp = None # 重置计时器
                
                self.droping_x = self.vehicle_local_position.x
                self.droping_y = self.vehicle_local_position.y
                self.droping_z = self.vehicle_local_position.z
                # <<< 投放逻辑结束 >>>

                return # 既然已经超时投放，直接结束本次函数调用
            
      
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
            dx_cam = self.target_position.y + self.depthcam_xoffset  # 相机中心相对投放中心的Y偏差
            dy_cam = -self.target_position.x + self.depthcam_yoffset  # 相机中心相对投放中心的X偏差
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
                    self.second_align_start_timestamp = None
                elif self.visited_targets_count == 1 and not self.Is_Finish_2nd_Drop:
                    self.drop_payload(1.0,-1.0)
                    self.get_logger().info("——————————————————————第二次投水————————————————————————")
                    self.Is_Finish_2nd_Drop = True
                    self.second_align_start_timestamp = None
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
        #进入offboard前发布位置控制点
        
                
        if self.offboard_setpoint_counter < 10:
            self.publish_position_setpoint(self.vehicle_local_position.x, self.vehicle_local_position.y, self.vehicle_local_position.z)
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
                #======增加限时模块========
                if self.drop_phase_start_time is None:
                    self.get_logger().info(f"已到达投水区域，启动 {self.drop_phase_timeout} 秒投放任务倒计时。")
                    self.drop_phase_start_time = self.get_clock().now()
                
                elapsed_drop_time = (self.get_clock().now() - self.drop_phase_start_time).nanoseconds / 1e9
                if elapsed_drop_time > self.drop_phase_timeout:
                    self.get_logger().warn(f"投放阶段整体超时（超过 {self.drop_phase_timeout} 秒），进入强制投放流程。")
                    # <<< 修改：不再直接投放，而是切换到专用状态 >>>
                    self.mission_state = MissionState.TIMEOUT_DROP
                    return # 立刻返回，让下一个循环处理新状态
                #======增加限时模块========
                
                if self.mission_state == MissionState.GLOBAL_SEARCH:
                    
                    #全局搜索限时10s
                    if self.search1_phase_start_time is None:
                        self.get_logger().info(f"开始全局搜索，限时 {self.search_timeout} 秒。")
                        self.search1_phase_start_time = self.get_clock().now()
                                        
                    #上升到global——search高度
                    self.global_search_target_z = float(self.initial_z+self.global_search_height)
                    self.publish_position_setpoint(self.DropArea_x, self.DropArea_y, self.global_search_target_z)
                    if self.log_counter % 10 == 0:
                        self.get_logger().info("进行全局搜索")

                    if self.vision_controller.initial_target_map:
                        self.get_logger().info("全局搜索完成，进入目标打击循环。")
                        self.mission_state = MissionState.TARGETING_CYCLE
                    if self.search1_phase_start_time and (self.get_clock().now() - self.search1_phase_start_time).nanoseconds / 1e9 > self.search_timeout:
                        self.get_logger().warn(f"第一次全局搜索超时，进入强制投放流程。")
                        # <<< 修改：切换到专用状态 >>>
                        self.mission_state = MissionState.TIMEOUT_DROP
                        return
                
                elif self.mission_state == MissionState.RECONFIRMING_TARGETS:
                    if self.search2_phase_start_time is None:
                        self.get_logger().info(f"开始第二次全局搜索，限时 {self.search_timeout} 秒。")
                        self.search2_phase_start_time = self.get_clock().now()
                    if self.search2_phase_start_time and (self.get_clock().now() - self.search2_phase_start_time).nanoseconds / 1e9 > self.search_timeout:
                        self.get_logger().error(f"第二次全局搜索超时，进入强制投放流程。")
                        # <<< 修改：切换到专用状态 >>>
                        self.mission_state = MissionState.TIMEOUT_DROP
                        return
                
                    self.get_logger().info("正在爬升并重新确认目标位置...")
                    # 命令无人机飞到全局搜索高度
                    self.publish_position_setpoint(self.proactive_target_x, self.proactive_target_y, self.global_search_target_z)
                    
                    # 检查视觉控制器是否看到了3个目标
                    num_targets_seen = self.vision_controller.get_current_detection_count()
                    if self.log_counter % 10 == 0:
                        self.get_logger().info(f"重新确认中... 当前看到 {num_targets_seen} / 3 个目标")

                    # 当再次看到大于2个目标时，才真正进入下一个目标的打击流程
                    if num_targets_seen > 1:
                        self.get_logger().info("重新确认成功！已找到至少两个目标。准备攻击下一个目标。")
                        
                        # 重置对准相关的状态，为下一个目标做准备
                        self.first_alignment_complete = False
                        self.second_alignment_complete = False
                        self.Is_Descending_to_depth_camera_height = False
                        self.first_alignment_checker.reset()
                        self.second_alignment_checker.reset()
                        
                        # 转换回目标打击循环状态
                        self.mission_state = MissionState.TARGETING_CYCLE

                elif self.mission_state == MissionState.PROACTIVE_SEARCH:
                    # --- 在这个状态下，无人机爬升并向下一个目标的大致方向移动 ---
                    # 1. 计算主动搜索的目标点 (只在第一次进入时计算)
                    if not self.is_proactive_target_set:
                        self.get_logger().info("计算主动搜索的目标点...")
                        
                        # 确定下一个目标是左还是右
                        next_target_name = self.target_priority[self.current_target_index]
                        previous_target_name = self.target_priority[self.current_target_index-1]

                        y_offset_frd = 0.0
                        if "Middle" in previous_target_name:
                            if "Left" in next_target_name:
                                y_offset_frd = -self.proactive_search_distance # FRD坐标系中，Y轴负方向是左
                                self.get_logger().info(f"下一个目标在左侧，向左移动 {self.proactive_search_distance} 米。")
                            elif "Right" in next_target_name:
                                y_offset_frd = self.proactive_search_distance # FRD坐标系中，Y轴正方向是右
                                self.get_logger().info(f"下一个目标在右侧，向右移动 {self.proactive_search_distance} 米。")
                        elif "Left" in previous_target_name :
                            y_offset_frd = self.proactive_search_distance # FRD坐标系中，Y轴正方向是右
                            self.get_logger().info(f"下一个目标在右侧，向右移动 {self.proactive_search_distance} 米。")
                        elif "Right" in previous_target_name :
                            y_offset_frd = -self.proactive_search_distance # FRD坐标系中，Y轴负方向是左
                            self.get_logger().info(f"下一个目标在左侧，向左移动 {self.proactive_search_distance} 米。")

                        # 基于投水区的中心点，计算偏移后的NED坐标
                        # 注意：这里我们使用 self.DropArea_x 和 self.DropArea_y 作为基准点
                        # 这样可以避免从有微小误差的投放点开始计算
                        base_x, base_y = self.DropArea_x, self.DropArea_y
                        
                        # 将FRD的偏移量转换为NED的偏移量
                        delta_x_ned = 0 * math.cos(self.init_yaw) - y_offset_frd * math.sin(self.init_yaw)
                        delta_y_ned = 0 * math.sin(self.init_yaw) + y_offset_frd * math.cos(self.init_yaw)

                        # 计算最终的NED目标点
                        self.proactive_target_x = base_x + delta_x_ned
                        self.proactive_target_y = base_y + delta_y_ned
                        
                        self.is_proactive_target_set = True
                        self.get_logger().info(f"主动搜索目标点(NED): x={self.proactive_target_x:.2f}, y={self.proactive_target_y:.2f}")

                    # 2. 命令无人机飞向目标点，并爬升到全局搜索高度
                    self.publish_position_setpoint(self.proactive_target_x, self.proactive_target_y, self.global_search_target_z)

                    # 3. 检查是否已到达目标点
                    current_x = self.vehicle_local_position.x
                    current_y = self.vehicle_local_position.y
                    height_error = abs(self.vehicle_local_position.z - self.global_search_target_z)
                    distance_error = math.sqrt((current_x - self.proactive_target_x)**2 + (current_y - self.proactive_target_y)**2)
                    
                    if self.log_counter % 10 == 0:
                        self.get_logger().info(f"主动搜索中... 距离目标点: {distance_error:.2f}米, 高度误差: {height_error:.2f}米")

                    # 如果水平和垂直都接近目标位置，则认为此阶段完成
                    if distance_error < 0.3 and height_error < 0.3:
                        self.get_logger().info("主动搜索阶段完成，已到达预定搜索区域。")
                        self.get_logger().info("现在切换到悬停确认阶段(RECONFIRMING_TARGETS)。")
                        
                        # 状态切换到确认阶段
                        self.mission_state = MissionState.RECONFIRMING_TARGETS
                        
                        # 重置标志位，以便下次（如果还有第三个目标）可以再次使用
                        self.is_proactive_target_set = False
                
                # GLOBAL_SEARCH执行一次之后，mission_state状态都为TARGETING_CYCLE
                elif self.mission_state == MissionState.TARGETING_CYCLE:
                    if self.visited_targets_count >= len(self.target_priority):
                        self.mission_state = MissionState.TIMEOUT_DROP
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
                            height_error = abs(self.vehicle_local_position.z - self.takeoff_target_height)
                            if height_error < 0.3:
                                self.Is_Descending_to_depth_camera_height = True
                                self.get_logger().info(f"目标 [{current_target_name}] 已锁定，下降完成。")
                            else:
                                self.get_logger().info(f"正在下降，误差{height_error}")

                        
                        if self.Is_Descending_to_depth_camera_height == True :
                            is_first_drop_transitioning = self.Is_Finish_1st_Drop and self.visited_targets_count == 0
                                
                            if is_first_drop_transitioning:
                                # 如果正在过渡，则只处理过渡逻辑，不执行对准
                                if self.first_drop_delay is None:
                                    self.first_drop_delay = self.get_clock().now()
                                
                                # 检查1秒延迟是否结束
                                if (self.get_clock().now() - self.first_drop_delay).nanoseconds / 1e9 > 1.0:
                                    self.get_logger().info("第一次投放完成。进入主动搜索阶段。")
                                    self.visited_targets_count += 1
                                    self.current_target_index += 1

                                    # --- 为下一个目标重置所有相关状态 ---
                                    self.first_alignment_complete = False
                                    self.second_alignment_complete = False
                                    self.Is_Descending_to_depth_camera_height = False
                                    self.first_alignment_checker.reset()
                                    self.second_alignment_checker.reset()
                                    # 关键：在这里也重置计时器，双重保险
                                    self.second_align_start_timestamp = None 
                                    self.first_drop_delay = None # 重置延迟计时器

                                    self.mission_state = MissionState.PROACTIVE_SEARCH                                    
                                    self.vision_controller.reset_to_search_mode()
                                else:
                                    if self.log_counter % 10 == 0:
                                        self.get_logger().info("第一次投放完成后等待...")

                            else:
                                # 如果不处于过渡期，才执行正常的对准和投放逻辑
                                self.adjust_to_target()

                            # 检查整个任务是否完成（两个都投完）
                            if self.Is_Finish_1st_Drop and self.Is_Finish_2nd_Drop:
                                self.is_FinishDrop = True
                
                elif self.mission_state == MissionState.TIMEOUT_DROP:
                    self.get_logger().info("正在执行超时强制投放流程...")
                    
                    # 1. 检查是否需要进行第一次投放
                    if not self.Is_Finish_1st_Drop:
                        self.get_logger().info("强制投放第一个载荷。")
                        self.drop_payload(-1.0, 1.0)
                        self.Is_Finish_1st_Drop = True
                        
                        # 记录投放时间，用于计算延迟
                        self.timeout_drop_start_time = self.get_clock().now()
                        return # 返回，等待下个循环来检查延迟

                    # 2. 如果第一个已投放，检查是否需要投放第二个
                    if not self.Is_Finish_2nd_Drop:
                        # 计算自第一次投放以来的时间
                        elapsed_time = (self.get_clock().now() - self.timeout_drop_start_time).nanoseconds / 1e9
                        
                        if elapsed_time > self.timeout_drop_delay:
                            self.get_logger().info("强制投放第二个载荷。")
                            self.drop_payload(1.0, -1.0)
                            self.Is_Finish_2nd_Drop = True
                        else:
                            if self.log_counter % 10 == 0:
                                self.get_logger().info(f"等待 {self.timeout_drop_delay}s 投放延迟... ({elapsed_time:.1f}s)")
                    
                    # 3. 检查是否全部投放完毕
                    if self.Is_Finish_1st_Drop and self.Is_Finish_2nd_Drop:
                        self.get_logger().info("所有载荷均已强制投放，任务完成。")
                        self.is_FinishDrop = True # 触发外部状态机进入 DROP_COMPLETE

            if self.is_FinishDrop: 
                self.get_logger().info("投放阶段任务完成。")
                self.start_mission()
                self.mission_state = MissionState.INMISSION
                if self.mission_state == MissionState.INMISSION:
                    if self.log_counter% 10 == 0:
                        self.get_logger().info(f"任务模式。")

        else:
            self.get_logger().info("启动offboard模式失败")
            
        if self.offboard_setpoint_counter < 30:
            self.offboard_setpoint_counter += 1

def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)

    # 2. 设置我们自己的命令行参数解析器
    parser = argparse.ArgumentParser(description="Offboard control script for PX4 drone mission.")
    
    # 添加你想通过命令行配置的参数
    parser.add_argument('--model-path', type=str, default='/home/weights/0711.engine',
                        help='Path to the object detection model file.')
    parser.add_argument('--photo-path', type=str, default='/home/image_recodes',
                        help='Base directory to save captured photos.')
    parser.add_argument('--video-path', type=str, default='/home/video_recodes',
                        help='Base directory to save recorded mission videos.')
    parser.add_argument('--camera-hint', type=str, default='imx577',
                        help='Hint to find the camera device name (e.g., "USB", "C920").')
    
    parser.add_argument('--takeoff-height', type=float, default=-2.3,
                        help='Takeoff height in meters (negative value for altitude).')
    parser.add_argument('--descent-height', type=float, default=1.0,
                        help='Descent height after first alignment in meters (positive value).')
    
    parser.add_argument('--forward-x', type=float, default=2.3,
                        help='Forward distance to fly to the drop area in meters.')
    parser.add_argument('--search-height', type=float, default=-5.0,
                        help='Global search height in meters (negative value for altitude).')
   
    parser.add_argument('--align-maxstep', type=float, default=0.2,
                        help='Maximum step size for each alignment adjustment.')
    parser.add_argument('--proactive-search-dist', type=float, default=0.6,
                        help='Distance to move sideways for proactive search.')
    
    # <<< 新增：在这里为 AlignmentChecker 添加参数 >>>
    parser.add_argument('--first-align-threshold', type=float, default=0.25,
                        help='Threshold (distance in meters) for the first alignment.')
    parser.add_argument('--first-align-time-window', type=float, default=2.0,
                        help='Time window (seconds) to maintain stability for the first alignment.')
    parser.add_argument('--first-align-check-freq', type=int, default=5,
                        help='Check frequency (how many timer calls per check) for the first alignment.')
    parser.add_argument('--second-align-threshold', type=float, default=0.15,
                        help='Threshold (distance in meters) for the second alignment.')
    parser.add_argument('--second-align-time-window', type=float, default=3.0,
                        help='Time window (seconds) to maintain stability for the second alignment.')
    parser.add_argument('--second-align-check-freq', type=int, default=5,
                        help='Check frequency (how many timer calls per check) for the second alignment.')    
    
    parser.add_argument('--drop-phase-timeout', type=float, default=90.0,
                        help='Maximum time in seconds for the entire dropping phase.')
    parser.add_argument('--search-timeout', type=float, default=10.0,
                        help='Maximum time in seconds for each search attempt.')
    parser.add_argument('--second-align-maxtime', type=float, default=8.0,
                        help='Maximum time in seconds for each search attempt.')
    
    
    parser.add_argument('--depthcam_xoffset', type=float, default=-0.065,
                        help='深度相机的x方向误差.')
    parser.add_argument('--depthcam_yoffset', type=float, default=0.033,
                        help='深度相机的y方向误差.')
    
    
    parser.add_argument('--trigger-distance', type=float, default=32.5,
                    help='切换到offboard的触发距离.')
    parser.add_argument('--position-threshold', type=float, default=0.5,
                    help='触发距离的阈值.')
    
     # --- 定时器参数 ---
    parser.add_argument('--timer-period', type=float, default=0.03,
                        help='定时器周期 (秒), 这也决定了PID控制中的 dt。默认: 0.03s (约33Hz).')

    # --- PID 核心参数 ---
    parser.add_argument('--kp', type=float, default=0.9911,
                        help='PID控制器 - 精细调节阶段的P增益 (Kp)。默认: 0.9911.')
    parser.add_argument('--ki', type=float, default=0.1021,
                        help='PID控制器 - 积分增益 (Ki)。默认: 0.1021.')
    parser.add_argument('--kd', type=float, default=0.0009,
                        help='PID控制器 - 微分增益 (Kd)。默认: 0.0009.')

    # --- PID 行为阈值和限制参数 ---
    parser.add_argument('--max-integral', type=float, default=0.2, # 这个值默认等于 align_maxstep
                        help='PID控制器 - 积分项的最大限制值 (防止积分饱和)。默认: 0.2.')
    
    # --- 选择投放桶 --- 
    parser.add_argument('--target-order', 
                        type=int,  # 关键：将类型改为整数
                        nargs='+', # 接收一个或多个值
                        default=[2, 1, 3], # 默认顺序: 中(2), 左(1), 右(3)
                        help='设置目标的投放顺序。使用数字: 1=左, 2=中, 3=右。 '
                             '例如: --target-order 3 1 2')
    # 3. 解析参数
    # 使用 rclpy.utilities.remove_ros_args 来确保我们只解析自己的参数，
    # 这样可以安全地与 ROS2 的参数（如 --ros-args）一起使用。
    custom_args = parser.parse_args(args=rclpy.utilities.remove_ros_args(args=sys.argv)[1:])

    TARGET_MAP = {
        1: "Left",
        2: "Middle",
        3: "Right"
    }
    VALID_INPUTS = set(TARGET_MAP.keys()) # {1, 2, 3}

    user_order_nums = custom_args.target_order

    # 验证1：检查用户输入的数字是否都在允许的范围内
    for num in user_order_nums:
        if num not in VALID_INPUTS:
            print(f"错误：无效的顺序编号 '{num}'。请从 {list(VALID_INPUTS)} 中选择。")
            sys.exit(1) # 退出程序

    # 验证2：确保没有重复的编号，并且数量正确 (正好是3个)
    if len(set(user_order_nums)) != len(VALID_INPUTS):
        print(f"错误：投放顺序必须包含且仅包含 {list(VALID_INPUTS)} 各一次。")
        print(f"您提供的顺序是: {user_order_nums}")
        sys.exit(1) # 退出程序

    # 翻译：将数字列表 [3, 1, 2] 转换为字符串列表 ["Right", "Left", "Middle"]
    try:
        translated_order_strings = [TARGET_MAP[num] for num in user_order_nums]
    except KeyError as e:
        # 这一步理论上不会出错，因为上面已经验证过了，但作为健壮性代码保留
        print(f"内部错误：无法翻译编号 {e}。")
        sys.exit(1)

    # 关键：用翻译好的字符串列表，覆盖掉原来的数字列表
    custom_args.target_order = translated_order_strings
    
    # =================================================================

    print(f"任务将按照以下顺序执行投放: {custom_args.target_order}")
    print('Starting offboard control node with custom parameters...')
    
    # 4. 将解析后的参数传入节点
    offboard_control = OffboardControl(args=custom_args)

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
