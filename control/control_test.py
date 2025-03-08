#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import time
import atexit
import signal
import traceback
import sys
from control.DronePositionChecker import DronePositionChecker
from control.AlignmentChecker import AlignmentChecker

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
        self.center_height_subscriber = self.create_subscription(Float32, '/current_height',
                                                                 self.current_height_callback, 10)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        
        # 新增：用于聚类和目标管理的变量
        self.bucket_positions = []  # 存储接收到的桶位置数据
        self.bucket_centers = None  # 存储聚类后的桶中心位置
        self.dropped_buckets = set()  # 记录已投放的桶
        self.current_target_idx = None  # 当前投放目标的索引
        self.is_aligning = False  # 是否正在对准某个目标
        self.target_position = None
        self.CurrentHeightFromCamera = 0.0
        
        self.already_reached = False
        self.first_aligned = False
        self.second_aligned = False
        self.last_found_x_NED = None
        self.last_found_y_NED = None
        self.last_found_z_NED = None

        self.takeoff_height = -1.7
        self.forward_x = 2.3
        self.align_maxstep = 0.2
        self.afterAlign_descentHeight = 0.5
        self.rise_height_after_drop = -1.0  # 投放后上升的高度

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
        self.is_rising_after_drop = False
        self.drop_count = 0

        self.droping_x = None
        self.droping_y = None
        self.droping_z = None

        # 新增日志计数器，用于减少日志输出频率
        self.log_counter = 0

        # 修改：将定时器周期从 0.03 秒调为 0.1 秒
        self.timer = self.create_timer(0.03, self.timer_callback)
        
        # 初始化位置判断器
        self.initPositionChecker = DronePositionChecker(
            logger_func=self.get_logger().info,
            tolerance=0.17, 
            duration=3.0
        )

        # 初始化 AlignmentChecker
        self.first_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,
            threshold=0.18,
            time_window=2.0,
            check_frequency=5
        )
        self.second_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,
            threshold=0.15,
            time_window=2.0,
            check_frequency=10
        )

    def target_position_callback(self, msg: Point):
        """Callback function for receiving target position."""
        self.bucket_positions.append([msg.x, msg.y])  # 收集桶位置数据
        self.target_position = msg  # 保留最新目标位置用于对准

    def current_height_callback(self, msg):
        self.CurrentHeightFromCamera = msg.data

    def initialize_bucket_centers(self):
        """Initialize bucket centers using clustering."""
        if len(self.bucket_positions) < 30 or self.bucket_centers is not None:
            return
        positions = np.array(self.bucket_positions[-30:])  # 使用最后30个位置数据
        kmeans = MiniBatchKMeans(n_clusters=3, batch_size=10).fit(positions)
        self.bucket_centers = kmeans.cluster_centers_
        self.get_logger().info(f"Bucket centers initialized: {self.bucket_centers}")

    def select_next_target(self):
        """Select the next target from un-dropped buckets."""
        if self.bucket_centers is None:
            return
        available_buckets = [(i, center) for i, center in enumerate(self.bucket_centers) 
                             if tuple(center) not in self.dropped_buckets]
        if not available_buckets:
            self.get_logger().warn("All buckets have been dropped.")
            return
        current_pos = np.array([self.vehicle_local_position.x, self.vehicle_local_position.y])
        distances = [np.linalg.norm(center - current_pos) for _, center in available_buckets]
        self.current_target_idx, _ = available_buckets[np.argmin(distances)]
        self.is_aligning = True
        self.get_logger().info(f"Selected target at index {self.current_target_idx}: {self.bucket_centers[self.current_target_idx]}")

    def update_bucket_centers(self):
        """Update the bucket centers based on the latest camera data."""
        if not self.is_aligning or self.current_target_idx is None or not self.target_position:
            return
        
        camera_pos = np.array([self.target_position.x, self.target_position.y])
        distances = [np.linalg.norm(camera_pos - center) for center in self.bucket_centers]
        
        if distances[self.current_target_idx] < 0.5:  # 阈值可调整
            alpha = 0.7  # 相机数据的权重
            self.bucket_centers[self.current_target_idx] = (
                alpha * camera_pos + (1 - alpha) * self.bucket_centers[self.current_target_idx]
            )
            if self.log_counter % 10==0:
                self.get_logger().info(f"Updated bucket center {self.current_target_idx}: {self.bucket_centers[self.current_target_idx]}")

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
        msg.yaw = self.init_yaw
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
    
    def drop_payload(self):
        """Drop the payload and record the dropped bucket."""
        self.get_logger().info("---------------Payload dropped.-------------------")
        if self.current_target_idx is not None:
            self.dropped_buckets.add(tuple(self.bucket_centers[self.current_target_idx]))
            self.droping_x = self.vehicle_local_position.x
            self.droping_y = self.vehicle_local_position.y
            self.droping_z = self.vehicle_local_position.z

    def takeoff_relative(self, relative_height):
        """Take off to a height relative to the current height."""
        if self.initial_z is None:
            self.initial_z = self.vehicle_local_position.z
            self.initial_x = self.vehicle_local_position.x
            self.initial_y = self.vehicle_local_position.y
            self.init_yaw = self.vehicle_local_position.heading
            self.takeoff_target_height = float(self.initial_z + relative_height)
            self.get_logger().info(f"初始稳定位置记录为：x:{self.initial_x:.2f},y:{self.initial_y:.2f},z:{self.initial_z:.2f} 米, init_yaw:{self.init_yaw:.2f}")
            self.get_logger().info(f"target_height:{self.takeoff_target_height}")
        self.fly_to_position_FRD2NED(0.0, 0.0, self.takeoff_target_height)

    def takeoff_height_check(self, threshold=0.1):
        """Check if the target takeoff height is reached."""
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

    def fly_forward_check(self, threshold=0.1):
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
        aligned = self.first_alignment_checker.alignment_check(
            current_x,
            current_y,
            target_x=target_x,
            target_y=target_y
        )
        if aligned:
            self.first_alignment_checker.first_aligned = True
            self.get_logger().info("------------------------first对准完成！------------------------")

    def second_alignment_check(self, target_x, target_y):
        """Check second alignment with the target."""
        aligned = self.second_alignment_checker.alignment_check(
            current_x=self.vehicle_local_position.x,
            current_y=self.vehicle_local_position.y,
            target_x=target_x,
            target_y=target_y
        )
        if aligned:
            self.second_alignment_checker.first_aligned = True
            self.get_logger().info("-------------------------second对准完成！------------------------")

    def fly_to_position_FRD2NED(self, x, y, z):
        """Convert FRD coordinates to NED and fly to the position."""
        x_target = x * math.cos(self.init_yaw) - y * math.sin(self.init_yaw) + self.initial_x
        y_target = x * math.sin(self.init_yaw) + y * math.cos(self.init_yaw) + self.initial_y
        z_target = z
        self.publish_position_setpoint(x_target, y_target, z_target)
        if self.log_counter % 10 == 0:
            self.get_logger().info(f"Flying to FRDposition: x={x}, y={y}, z={z}")
        return x_target, y_target

    def coordinate_NED2FRD(self, x_NED, y_NED):
        """Convert NED coordinates to FRD."""
        x_FRD = (x_NED - self.initial_x) * math.cos(self.init_yaw) + (y_NED - self.initial_y) * math.sin(self.init_yaw)
        y_FRD = -(x_NED - self.initial_x) * math.sin(self.init_yaw) + (y_NED - self.initial_y) * math.cos(self.init_yaw)
        return x_FRD, y_FRD

    def coordinate_FRD2NED(self, x, y):
        """Convert FRD coordinates to NED."""
        x_target = x * math.cos(self.init_yaw) - y * math.sin(self.init_yaw) + self.initial_x
        y_target = x * math.sin(self.init_yaw) + y * math.cos(self.init_yaw) + self.initial_y
        return x_target, y_target

    def check_isaround_fly(self, target_x_NED, target_y_NED, z):
        """Adjust height based on camera feedback and fly to position."""
        if self.CurrentHeightFromCamera != 0.0:
            if not self.first_alignment_checker.first_aligned:
                if self.CurrentHeightFromCamera - abs(self.takeoff_height) > 0.17:
                    z = z + 0.14
                    if self.log_counter % 10 == 0:
                        self.get_logger().info(f"根据深度相机，高度过高，下降,相机回传高度：{self.CurrentHeightFromCamera:.3f}")
                if abs(self.takeoff_height) - self.CurrentHeightFromCamera > 0.17:
                    z = z - 0.14
                    if self.log_counter % 10 == 0:
                        self.get_logger().info(f"根据深度相机，高度过低，上升,相机回传高度：{self.CurrentHeightFromCamera}")
                self.CurrentHeightFromCamera = 0.0 
            else:
                if self.CurrentHeightFromCamera - abs(self.takeoff_height + self.afterAlign_descentHeight) > 0.13:
                    z = z + 0.09
                    if self.log_counter % 10 == 0:
                        self.get_logger().info(f"根据深度相机，高度过高，下降,相机回传高度：{self.CurrentHeightFromCamera}")
                if abs(self.takeoff_height + self.afterAlign_descentHeight) - self.CurrentHeightFromCamera > 0.13:
                    z = z - 0.09
                    if self.log_counter % 10 == 0:
                        self.get_logger().info(f"根据深度相机，高度过低，上升,相机回传高度：{self.CurrentHeightFromCamera}")
                self.CurrentHeightFromCamera = 0.0  
        self.fly_to_position(target_x_NED, target_y_NED, z)

    def adjust_to_target(self):
        """Adjust drone position towards the current target."""
        if not self.is_aligning and self.current_target_idx is None:
            self.select_next_target()

        if self.current_target_idx is None:
            if self.log_counter % 10==0:
                self.get_logger().info("No target available, flying to drop area")
            self.fly_to_position(self.DropArea_x, self.DropArea_y, self.takeoff_target_height)
            return

        # 更新聚类中心
        self.update_bucket_centers()
        if self.target_position:
        # 获取当前目标的最新位置
            target_x_NED, target_y_NED = self.bucket_centers[self.current_target_idx]
            current_x, current_y = self.vehicle_local_position.x, self.vehicle_local_position.y
            distance = math.sqrt((current_x - target_x_NED)**2 + (current_y - target_y_NED)**2)
            scale = self.align_maxstep/distance
            if distance < self.align_maxstep:
                target_x_NED_f = target_x_NED
                target_y_NED_f = target_y_NED
            else:
                target_x_NED_f = current_x + (target_x_NED-current_x)*scale
                target_y_NED_f = current_y + (target_y_NED-current_y)*scale
                if self.log_counter % 10 == 0:
                    self.get_logger().info("超过最大步长，")        
            if self.log_counter % 10 == 0:
                self.get_logger().info(f"Distance to target: {distance:.4f}")

            if not self.first_alignment_checker.first_aligned:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("Performing first alignment")
                self.check_isaround_fly(target_x_NED_f, target_y_NED_f, self.takeoff_target_height)
                self.first_alignment_check(target_x_NED, target_y_NED)
                self.last_found_x_NED = target_x_NED_f
                self.last_found_y_NED = target_y_NED_f
                self.last_found_z_NED = self.takeoff_target_height

            elif self.first_alignment_checker.first_aligned and not self.second_alignment_checker.first_aligned:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("Performing second alignment")
                self.check_isaround_fly(target_x_NED_f, target_y_NED_f, self.takeoff_target_height + self.afterAlign_descentHeight)
                self.second_alignment_check(target_x_NED, target_y_NED)
                self.last_found_x_NED = target_x_NED_f
                self.last_found_y_NED = target_y_NED_f
                self.last_found_z_NED = self.takeoff_target_height + self.afterAlign_descentHeight
            
            self.target_position = None  
            
            if self.first_alignment_checker.first_aligned and self.second_alignment_checker.first_aligned:
                self.drop_payload()
                self.is_FinishDrop = True
                self.is_aligning = False
        else:
            if self.last_found_x_NED and self.last_found_y_NED and self.last_found_z_NED:
                self.fly_to_position(self.last_found_x_NED, self.last_found_y_NED, self.last_found_z_NED)
            else:
                self.get_logger().info("上次记录不存在")
                self.fly_to_position(self.DropArea_x, self.DropArea_y, self.takeoff_target_height)

    def rise_after_drop(self):
        """Rise to a certain height after dropping the payload."""
        rise_target_height = self.takeoff_target_height
        x, y = self.vehicle_local_position.x, self.vehicle_local_position.y
        self.fly_to_position(x, y, rise_target_height)
        self.get_logger().info(f"Rising to height: {rise_target_height}")

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal()
        
        # 更新日志计数器
        self.log_counter += 1

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            # 仅在日志计数满足条件时打印
            if self.log_counter % 10 == 0:
                self.get_logger().info("try offboard")

        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if not self.is_ReadyToTakeoff:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("执行步骤1,判断飞机起飞前位置是否稳定")
                    Position = (
                        self.vehicle_local_position.x,
                        self.vehicle_local_position.y,
                        self.vehicle_local_position.z
                    )
                    self.get_logger().info(f"positon:{Position}")
                self.initPositionChecker.update_position((
                    self.vehicle_local_position.x,
                    self.vehicle_local_position.y,
                    self.vehicle_local_position.z
                ))
                if self.initPositionChecker.is_stable():
                    self.is_ReadyToTakeoff = True
                    self.arm()
                    if self.log_counter % 10 == 0:
                        self.get_logger().info("起飞前位置稳定。")

            if self.is_ReadyToTakeoff and not self.is_AtTakeoffHeight:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("执行步骤2,上升到指定高度")
                self.takeoff_relative(self.takeoff_height)
                self.takeoff_height_check()

            if self.is_AtTakeoffHeight and not self.is_AtDropArea:
                if self.log_counter % 10 == 0:
                    self.get_logger().info("执行步骤3,飞向投水区")
                self.fly_forward(self.forward_x)
                self.fly_forward_check()

            if self.is_AtDropArea and not self.is_FinishDrop:
                self.initialize_bucket_centers()  # 初始化桶位置
                self.adjust_to_target()

            if self.is_FinishDrop:
                if self.drop_count < 1:  # 检查是否还需要投放
                    if not self.is_rising_after_drop:
                        self.rise_after_drop()
                        self.is_rising_after_drop = True
                    else:
                        target_height = self.takeoff_target_height
                        if abs(self.vehicle_local_position.z - target_height) < 0.07:
                            self.is_rising_after_drop = False
                            self.is_FinishDrop = False
                            self.first_alignment_checker.first_aligned = False
                            self.second_alignment_checker.first_aligned = False
                            self.drop_count += 1
                            if self.log_counter % 10 == 0:
                                self.get_logger().info(f"准备进行第 {self.drop_count + 1} 次投放")
                            self.select_next_target()  # 选择下一个目标
                else:
                    self.fly_to_position(float(self.droping_x), float(self.droping_y), float(self.droping_z))

        if self.offboard_setpoint_counter < 30:
            self.offboard_setpoint_counter += 1

def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
