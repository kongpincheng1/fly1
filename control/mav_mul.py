#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Point
# from std_msgs.msg import Float32
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav_msgs.msg import Odometry # 如果你需要速度信息，可以用这个
from collections import deque
import time
from control.DronePositionChecker import DronePositionChecker
from control.AlignmentChecker import AlignmentChecker
#from control.ServoControl import ServoControl
from control.visual_servoing import VisualServoingController, VisualState # 从你的包中导入视觉控制器
import cv2
from enum import Enum
import subprocess
import re
import os
CAMERA_NAME_HINT = "USB"


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
    # --- Mission Parameters ---
    TAKEOFF_HEIGHT = 2.1
    FORWARD_FLIGHT_DISTANCE = 2.3
    GLOBAL_SEARCH_ALTITUDE = 5.0
    ALIGNMENT_MAX_STEP = 0.2
    POST_ALIGNMENT_DESCENT = -0.7  # Use positive for descent if z is up, negative if z is down
    CAMERA_DROPPER_OFFSET_Y = 0.052
    
    # --- Tolerance ------
    POSITION_STABILITY_TOLERANCE = 0.17
    POSITION_STABILITY_DURATION = 4
    FLYING_TO_POINT_TOLERANCE = 0.20
    FIRST_ALIGNMENT_TOLERANCE = 0.15
    SECOND_ALIGNMENT_TOLERANCE = 0.10
    
    
    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        # Configure QoS profile for publishing and subscribing
        control_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )


        status_qos_profile = QoSProfile(
            # 关键修改：根据`ros2 topic info`的输出，必须使用 BEST_EFFORT
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            # 同样根据`ros2 topic info`的输出，使用 VOLATILE
            durability=DurabilityPolicy.VOLATILE,
            depth=1  # MAVROS通常使用depth=5，但depth=1通常是兼容的
        )

                # +++ 新增的诊断日志 +++
        self.get_logger().info(
            f"--- [DIAGNOSIS] ATTEMPTING TO USE QOS: "
            f"RELIABILITY={status_qos_profile.reliability}, "
            f"DURABILITY={status_qos_profile.durability} ---"
        )
        # +++ 诊断日志结束 +++
        # Create publishers
        self.trajectory_setpoint_publisher = self.create_publisher(
    PositionTarget, '/mavros/setpoint_raw/local', control_qos_profile)
        
        # 创建订阅者
        self.vehicle_local_position_subscriber = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.vehicle_local_position_callback, status_qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            State, '/mavros/state', self.vehicle_status_callback, status_qos_profile)        
                
        self.target_position_subscriber = self.create_subscription(Point, '/target_position',
                                                                   self.target_position_callback, status_qos_profile)
                # --- 新增：创建服务客户端 ---
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # 检查服务是否可用，这是很好的实践
        while not self.arming_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Arming service not available, waiting again...')
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set mode service not available, waiting again...')
        self.get_logger().info('MAVROS services are available.')

        
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
        
        self.current_pose = PoseStamped()
        self.current_state = State() # 我们重命名一下以示区分
        

        self.target_position = None        
        self.already_reached = False
        self.first_aligned = False
        self.second_aligned = False
        self.last_found_x_enu = None
        self.last_found_y_enu = None
        self.last_found_z_enu = None


        #起飞高度
        self.takeoff_height = self.TAKEOFF_HEIGHT
        #向前飞行的距离
        self.forward_x = self.FORWARD_FLIGHT_DISTANCE
        #最大步长
        self.align_maxstep = self.ALIGNMENT_MAX_STEP
        #对准之后下降的高度
        self.afterAlign_descentHeight = self.POST_ALIGNMENT_DESCENT
        #GLOBAL_SEARCH高度
        self.global_search_height = self.GLOBAL_SEARCH_ALTITUDE
    

        self.global_search_target_z = None
        self.initial_z = None  
        self.initial_x = None  
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
        self.current_yaw = 0.0

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.03, self.timer_callback)
        
        #初始化位置判断器
        self.initPositionChecker = DronePositionChecker(
            logger_func=self.get_logger().info,
            tolerance=self.POSITION_STABILITY_TOLERANCE, 
            duration=self.POSITION_STABILITY_DURATION
        )

          # 初始化 AlignmentChecker
        self.first_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,  # 传递日志记录函数
            threshold=self.FIRST_ALIGNMENT_TOLERANCE,
            time_window=2.0,
            check_frequency=5
        )
        self.second_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,  # 传递日志记录函数
            threshold=self.SECOND_ALIGNMENT_TOLERANCE,
            time_window=2.0,
            check_frequency=5
        )
        # 初始化舵机控制器
        # self.servo_control = ServoControl()


    def target_position_callback(self, msg: Point):
        """Callback function for receiving target position."""
        self.target_position = msg  

    def fly_to_position(self, x, y, z):
        """Fly to the specified position."""
        self.publish_position_setpoint(x, y, z, self.init_yaw)

    def vehicle_local_position_callback(self, msg): # msg 类型现在是 PoseStamped
        """Callback function for vehicle_local_position topic subscriber."""
        self.current_pose = msg

        # 从四元数计算偏航角，因为 self.init_yaw 很重要
        # 您需要在 __init__ 中添加 self.current_yaw = 0.0
        q = msg.pose.orientation
        self.current_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def vehicle_status_callback(self, msg): # msg 类型现在是 State
        """Callback function for vehicle_status topic subscriber."""
        self.current_state = msg

    def arm(self):
        """Send an arm command using MAVROS service."""
        req = CommandBool.Request()
        req.value = True
        future = self.arming_client.call_async(req)
        self.get_logger().info('Arm command sent')
        # rclpy.spin_until_future_complete(self, future) # 可以选择等待结果

    def disarm(self):
        """Send a disarm command using MAVROS service."""
        req = CommandBool.Request()
        req.value = False
        self.arming_client.call_async(req)
        self.get_logger().info('Disarm command sent')

    def set_flight_mode(self, mode):
        """Set flight mode using MAVROS service."""
        req = SetMode.Request()
        # MAVROS 使用字符串来定义自定义模式
        # PX4 主模式码 6 对应 'OFFBOARD'
        req.custom_mode = mode # e.g., 'OFFBOARD', 'AUTO.LAND'
        self.set_mode_client.call_async(req)
        self.get_logger().info(f"Request to switch to {mode} mode sent")

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.set_flight_mode('OFFBOARD')

    def land(self):
        """Switch to land mode."""
        self.set_flight_mode('AUTO.LAND')

    def publish_position_setpoint(self, x: float, y: float, z: float, yaw: float = None):
        """Publish the trajectory setpoint using PositionTarget."""
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map' # 或者 'odom'，取决于您的设置

        # MAVROS 使用 type_mask 来决定哪些字段是有效的。
        # 我们只想控制位置和偏航角。
        # IGNORE_VX, VY, VZ, AX, AY, AZ, YAW_RATE
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = (PositionTarget.IGNORE_VX |
                        PositionTarget.IGNORE_VY |
                        PositionTarget.IGNORE_VZ |
                        PositionTarget.IGNORE_AFX |
                        PositionTarget.IGNORE_AFY |
                        PositionTarget.IGNORE_AFZ |
                        PositionTarget.IGNORE_YAW_RATE)
        
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z
        
        # 如果提供了yaw，就使用它。否则，忽略yaw。
        if yaw is not None:
            msg.yaw = yaw
        else:
            msg.type_mask |= PositionTarget.IGNORE_YAW # 在掩码中添加忽略YAW

        self.trajectory_setpoint_publisher.publish(msg)

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

        current_x, current_y = self.coordinate_ENU2FRD(self.current_pose.pose.position.x, self.current_pose.pose.position.y)
        
        delta_x, delta_y = 0.0, 0.0
        if "向右平移" in command: delta_y = step_size_xy  
        if "向左平移" in command: delta_y = -step_size_xy
        if "向前平移" in command: delta_x = step_size_xy
        if "向后平移" in command: delta_x = -step_size_xy

        # 计算新的FRD目标点
        target_x_frd = current_x + delta_x
        target_y_frd = current_y + delta_y
        
        # 转换回NED并发布
        target_x_enu, target_y_enu = self.coordinate_FRD2ENU(target_x_frd, target_y_frd)
        self.publish_position_setpoint(target_x_enu, target_y_enu, self.global_search_target_z, self.init_yaw)    
    
    def drop_payload(self,servo_1,servo_2):
        # self.servo_control.open_servo(servo_1,servo_2)

        self.get_logger().info("---------------Payload dropped.-------------------")

    def takeoff_relative(self):
        # ...
        # 1. 先用纯函数计算出目标点的 x, y 坐标
        target_x, target_y = self.coordinate_FRD2ENU(0.0, 0.0)
        # 2. 再连同目标高度 z，一起发布出去
        self.publish_position_setpoint(target_x, target_y, self.takeoff_target_height, self.init_yaw)

    def takeoff_height_check(self, threshold=0.22):
        """
        检查是否到达相对目标高度
        :param threshold: 高度误差阈值
        :return: True 如果到达目标高度，否则 False
        """
        if self.takeoff_target_height is None:
            self.get_logger().warn("目标高度尚未设置！")
            return False
        current_height = self.current_pose.pose.position.z
        height_error = abs(current_height - self.takeoff_target_height)
        # 为了减少日志输出，只有每隔一定周期时才打印此日志
        if self.log_counter % 10 == 0:
            self.get_logger().info(f"当前高度：{current_height:.2f} 米，目标高度：{self.takeoff_target_height:.2f} 米，高度误差：{height_error:.2f} 米")
        if height_error < threshold:
            self.is_AtTakeoffHeight = True

    def fly_forward(self, x):
        """Fly forward to the drop area."""
        # 1. 计算出目标点的 x, y 坐标
        target_x, target_y = self.coordinate_FRD2ENU(x, 0.0)
        # 2. 记录下来用于检查
        self.DropArea_x = target_x
        self.DropArea_y = target_y
        # 3. 发布指令
        self.publish_position_setpoint(target_x, target_y, self.takeoff_target_height, self.init_yaw)
    
    def fly_forward_check(self, threshold=None):
        """Check if the drone has reached the drop area."""
        # 如果调用时没有提供threshold，就使用实例中定义的默认值
        if threshold is None:
            threshold = self.FLYING_TO_POINT_TOLERANCE

        current_x = self.current_pose.pose.position.x
        current_y = self.current_pose.pose.position.y
        error = math.sqrt((current_x - self.DropArea_x)**2 + (current_y - self.DropArea_y)**2)
        if self.log_counter % 10 == 0:
            self.get_logger().info(f"--当前x：{current_x:.2f},y:{current_y:.2f}米，--目标x：{self.DropArea_x:.2f} 米，y:{self.DropArea_y:.2f},--误差：{error:.2f} 米")
        if error < threshold:
            self.is_AtDropArea = True
    
    def first_alignment_check(self, target_x, target_y):
        """Check first alignment with the target."""
        current_x = self.current_pose.pose.position.x
        current_y = self.current_pose.pose.position.y
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
    current_x=self.current_pose.pose.position.x,
    current_y=self.current_pose.pose.position.y,
            target_x=target_x,
            target_y=target_y
        )
        if is_align_now:
            self.second_alignment_complete = True
            self.get_logger().info("-------------------------second对准完成！------------------------")


    def coordinate_ENU2FRD(self, x_enu: float, y_enu: float) -> tuple[float, float]:
        """
        将世界坐标系(ENU)下的一个绝对点，转换为相对于无人机初始位置的机体坐标(FRD)。
        这个函数用于回答 "某个世界坐标点，在我的机头前方/右侧多远？" 这样的问题。

        :param x_enu: 世界坐标系下的 X 坐标 (East)
        :param y_enu: 世界坐标系下的 Y 坐标 (North)
        :return: (x_frd, y_frd) - 相对于无人机初始位置的机体坐标
        """
        # 使用在Offboard模式启动时记录的初始偏航角作为旋转基准
        yaw = self.init_yaw

        # 1. 先从绝对坐标中减去初始位置，得到一个纯粹的、从起点出发的向量
        delta_x_enu = x_enu - self.initial_x
        delta_y_enu = y_enu - self.initial_y

        # 2. --- 应用反向旋转矩阵 ---
        #    将世界坐标系下的向量，反向旋转 yaw 度 (或者说，旋转 -yaw 度)
        #    从而得到它在机体坐标系下的表示
        #    cos(-yaw) = cos(yaw)
        #    sin(-yaw) = -sin(yaw)
        x_frd =  delta_x_enu * math.cos(yaw) + delta_y_enu * math.sin(yaw)
        y_frd = -delta_x_enu * math.sin(yaw) + delta_y_enu * math.cos(yaw)
        
        return x_frd, y_frd

    def coordinate_FRD2ENU(self, x_frd: float, y_frd: float) -> tuple[float, float]:
        """
        将相对于无人机机体(FRD)的运动增量，转换为世界坐标系(ENU)下的绝对目标点。
        这是最常用的函数，用于将 "向前飞"、"向右飞" 这种指令转换成世界坐标。

        :param x_frd: 机体前方为正的距离 (Front)
        :param y_frd: 机体右侧为正的距离 (Right)
        :return: (x_target_enu, y_target_enu) - 在世界坐标系(ENU)下的绝对目标点
        """
        # 使用在Offboard模式启动时记录的初始偏航角作为旋转基准
        yaw = self.init_yaw 

        # --- 标准二维旋转矩阵 ---
        # 将机体坐标系下的运动向量 (x_frd, y_frd) 旋转 yaw 度
        # 得到在世界坐标系下的运动增量 (delta_x_enu, delta_y_enu)
        delta_x_enu = x_frd * math.cos(yaw) - y_frd * math.sin(yaw)
        delta_y_enu = x_frd * math.sin(yaw) + y_frd * math.cos(yaw)
        
        # 将世界坐标系下的运动增量，叠加到无人机的初始位置上
        # 从而得到最终的绝对目标点
        x_target_enu = self.initial_x + delta_x_enu
        y_target_enu = self.initial_y + delta_y_enu

        return x_target_enu, y_target_enu
    
    def adjust_to_target(self):
        """Adjust drone position towards the current target."""
        if self.target_position:
            # Example logic: Adjust position incrementally based on target position
            current_x_enu,current_y_enu = self.current_pose.pose.position.x, self.current_pose.pose.position.y
            current_x, current_y =self.coordinate_ENU2FRD(current_x_enu,current_y_enu)
            distance = math.sqrt((self.target_position.x)**2+(self.target_position.y)**2)
            scale = self.align_maxstep/distance 
            target_x_FRD = current_x + self.target_position.y - 0.052  # 0.05 为相机中心相对投放中心的误差。
            target_y_FRD = current_y - self.target_position.x 

            target_x_enu, target_y_enu = self.coordinate_FRD2ENU(target_x_FRD, target_y_FRD)
            if distance < self.align_maxstep:
                target_x_FRD_f = current_x + self.target_position.y
                target_y_FRD_f = current_y - self.target_position.x
            else:
                target_x_FRD_f = current_x + self.target_position.y*scale
                target_y_FRD_f = current_y - self.target_position.x*scale
                if self.log_counter % 10 == 0:
                    self.get_logger().info("超过最大步长，")

            target_x_enu_f,target_y_enu_f = self.coordinate_FRD2ENU(target_x_FRD_f, target_y_FRD_f)
            # First alignment
            if not self.first_alignment_complete:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("Performing first alignment")
                self.fly_to_position(target_x_enu_f, target_y_enu_f, self.takeoff_target_height)
                self.first_alignment_check(target_x_enu, target_y_enu)
                self.last_found_x_enu = target_x_enu_f
                self.last_found_y_enu = target_y_enu_f
                self.last_found_z_enu = self.takeoff_target_height

            elif self.first_alignment_complete and not self.second_alignment_complete:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("Performing second alignment")
                self.fly_to_position(target_x_enu_f, target_y_enu_f, self.takeoff_target_height + self.afterAlign_descentHeight)
                self.second_alignment_check(target_x_enu, target_y_enu)
                self.last_found_x_enu = target_x_enu_f
                self.last_found_y_enu = target_y_enu_f
                self.last_found_z_enu = self.takeoff_target_height + self.afterAlign_descentHeight

            self.target_position = None
            
            if self.first_alignment_complete and self.second_alignment_complete:
                if not self.Is_Finish_1st_Drop:
                    self.drop_payload(-1.0,1.0)
                    self.get_logger().info("——————————————————————DROP————————————————————————")
                    self.Is_Finish_1st_Drop = True
                elif not self.Is_Finish_2nd_Drop:
                    self.drop_payload(1.0,-1.0)
                    self.get_logger().info("——————————————————————DROP————————————————————————")
                    self.Is_Finish_2nd_Drop = True
                self.droping_x = self.current_pose.pose.position.x
                self.droping_y = self.current_pose.pose.position.y
                self.droping_z = self.current_pose.pose.position.z

                
        else:
            if self.last_found_x_enu and self.last_found_y_enu and self.last_found_z_enu:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("使用上次记录")
                self.fly_to_position(self.last_found_x_enu, self.last_found_y_enu, self.last_found_z_enu)
            else:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("上次记录不存在")
                self.fly_to_position(self.DropArea_x, self.DropArea_y, self.takeoff_target_height)


    #定时器
    def timer_callback(self) -> None:
        """Callback function for the timer."""

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
                
        if self.current_state.connected and self.offboard_setpoint_counter < 30: # 稍微增加计数器上限
            # 持续发送当前位置作为设定点，这是让飞控信任我们的关键
            self.publish_position_setpoint(
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
                self.current_pose.pose.position.z,
                self.current_yaw
            )
            
            # 在发送了一段时间设定点后 (例如 counter > 10)，再请求切换模式
            if self.current_state.mode != "OFFBOARD" and self.offboard_setpoint_counter > 10:
                self.get_logger().info("尝试切换到 OFFBOARD 模式...")
                self.engage_offboard_mode()

        if self.current_state.mode == "OFFBOARD":
            if not self.is_ReadyToTakeoff:
                if self.initial_x is None:
                    # 第一次进入此状态，记录当前位置为目标保持位置
                    self.initial_x = self.current_pose.pose.position.x
                    self.initial_y = self.current_pose.pose.position.y
                    self.initial_z = self.current_pose.pose.position.z
                    self.init_yaw = self.current_yaw 
                    self.get_logger().info(f"进入Offboard模式，锁定初始位置: x={self.initial_x:.2f}, y={self.initial_y:.2f}, z={self.initial_z:.2f}")

                # 持续发布保持初始位置的指令
                self.publish_position_setpoint(self.initial_x, self.initial_y, self.initial_z, self.init_yaw)

                # 更新并检查位置稳定性
                current_pos = (
                    self.current_pose.pose.position.x,
                    self.current_pose.pose.position.y,
                    self.current_pose.pose.position.z
                )
                self.initPositionChecker.update_position(current_pos)

                if self.initPositionChecker.is_stable():
                    self.is_ReadyToTakeoff = True
                    self.arm()
                    self.get_logger().info("位置已稳定，发送解锁指令...")
            
            elif self.is_ReadyToTakeoff and self.current_state.armed and not self.is_AtTakeoffHeight:
                if self.takeoff_target_height is None:
                    # 在第一次确认解锁后，再设置目标高度，确保万无一失
                    self.initial_z = self.current_pose.pose.position.z
                    self.takeoff_target_height = float(self.initial_z + self.takeoff_height)
                    self.get_logger().info(f"已解锁！起飞基准高度: {self.initial_z:.2f} m, 目标起飞高度: {self.takeoff_target_height:.2f} m")
                
                if self.log_counter % 10 == 0:
                    self.get_logger().info("执行步骤2,上升到指定高度")
                self.takeoff_relative()
                self.takeoff_height_check()
                # self.is_AtTakeoffHeight = False#  测试用

            elif self.is_AtTakeoffHeight and not self.is_AtDropArea:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("执行步骤3,飞向投水区")
                self.fly_forward(self.forward_x)
                self.fly_forward_check()
                # self.is_AtDropArea = False #测试用

            elif self.is_AtDropArea and not self.is_FinishDrop:
                if self.mission_state == MissionState.GLOBAL_SEARCH:
                    #上升到global——search高度
                    self.global_search_target_z = float(self.initial_z+self.global_search_height)
                    self.publish_position_setpoint(self.DropArea_x, self.DropArea_y, self.global_search_target_z, self.init_yaw)
                    if self.log_counter % 10 == 0:
                        self.get_logger().info("进行全局搜索")

                    if self.vision_controller.initial_target_map:
                        self.get_logger().info("全局搜索完成，进入目标打击循环。")
                        self.mission_state = MissionState.TARGETING_CYCLE
                
                elif self.mission_state == MissionState.RECONFIRMING_TARGETS: # <<< 新增的处理块
                    self.get_logger().info("正在爬升并重新确认目标位置...")
                    # 命令无人机飞到全局搜索高度
                    self.publish_position_setpoint(self.DropArea_x, self.DropArea_y, self.global_search_target_z, self.init_yaw)
                    
                    # 检查视觉控制器是否看到了3个目标
                    num_targets_seen = self.vision_controller.get_current_detection_count()
                    if self.log_counter % 10 == 0:
                        self.get_logger().info(f"重新确认中... 当前看到 {num_targets_seen} / 3 个目标")

                    # 当再次看到3个目标时，才真正进入下一个目标的打击流程
                    if num_targets_seen > 1:
                        self.get_logger().info("重新确认成功！已找到至少2个目标。准备攻击下一个目标。")
                        
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
                            self.publish_position_setpoint(self.current_pose.pose.position.x, self.current_pose.pose.position.y, self.takeoff_target_height, self.init_yaw)                        
                            if abs(self.current_pose.pose.position.z - self.takeoff_target_height) < 0.2:
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

            elif self.is_FinishDrop: 
                self.fly_to_position(float(self.droping_x), float(self.droping_y), float(self.droping_z))
        else:
            self.get_logger().info("启动offboard模式失败")
            
        
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
