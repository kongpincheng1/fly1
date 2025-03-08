#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleOdometry
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

import time
import atexit
import signal
import traceback
import sys
# from control.visualize import Visualize
from control.DronePositionChecker import DronePositionChecker
from control.AlignmentChecker import AlignmentChecker
# import RPi.GPIO as GPIO

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
        self.align_maxstep = 0.08
        self.afterAlign_descentHeight = 0.5
        self.rise_height_after_drop = -1.0  # 新增：投放后上升的高度（负值表示上升）

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
        self.is_rising_after_drop = False # 新增：高度二次上升判断
        self.drop_count = 0  # 新增：投放次数计数器

        self.droping_x = None
        self.droping_y = None
        self.droping_z = None


        # Create a timer to publish control commands
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
        """Drop the payload (example logic)."""
        # 这里假设使用GPIO控制投放，实际实现可能需要调整
        self.get_logger().info("---------------Payload dropped.-------------------")

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
        self.get_logger().info(f"--当前x：{current_x:.2f},y:{current_y:.2f}米，--目标x：{self.DropArea_x:.2f} 米，y:{self.DropArea_y:.2f},--误差：{error:.2f} 米")
        if error < threshold:
            self.is_AtDropArea = True
    
    def first_alignment_check(self, target_x1, target_y1):
        """Check first alignment with the target."""
        aligned = self.first_alignment_checker.alignment_check(
            current_x=self.vehicle_local_position.x,
            current_y=self.vehicle_local_position.y,
            target_x=target_x1,
            target_y=target_y1
        )
        if aligned:
            self.first_alignment_checker.first_aligned = True
            self.get_logger().info("------------------------first对准完成！------------------------")

    def second_alignment_check(self, target_x1, target_y1):
        """Check second alignment with the target."""
        aligned = self.second_alignment_checker.alignment_check(
            current_x=self.vehicle_local_position.x,
            current_y=self.vehicle_local_position.y,
            target_x=target_x1,
            target_y=target_y1
        )
        if aligned:
            self.second_aligned = True
            self.get_logger().info("-------------------------second对准完成！------------------------")

    def fly_to_position_FRD2NED(self, x, y, z):
        """Convert FRD coordinates to NED and fly to the position."""
        x_target = x * math.cos(self.init_yaw) - y * math.sin(self.init_yaw) + self.initial_x
        y_target = x * math.sin(self.init_yaw) + y * math.cos(self.init_yaw) + self.initial_y
        z_target = z
        self.publish_position_setpoint(x_target, y_target, z_target)
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
                    self.get_logger().info(f"根据深度相机，高度过高，下降,相机回传高度：{self.CurrentHeightFromCamera:.3f},初始高度：{self.initial_z:.3f},修改后飞行高度：{z:.3f},当前高度：{self.vehicle_local_position.z:.3f}")
                if abs(self.takeoff_height) - self.CurrentHeightFromCamera > 0.17:
                    z = z - 0.14
                    self.get_logger().info(f"根据深度相机，高度过低，上升,相机回传高度：{self.CurrentHeightFromCamera}")
                self.CurrentHeightFromCamera = 0.0 
            else:
                if self.CurrentHeightFromCamera - abs(self.takeoff_height + self.afterAlign_descentHeight) > 0.13:
                    z = z + 0.09
                    self.get_logger().info(f"根据深度相机，高度过高，下降,相机回传高度：{self.CurrentHeightFromCamera}")
                if abs(self.takeoff_height + self.afterAlign_descentHeight) - self.CurrentHeightFromCamera > 0.13:
                    z = z - 0.09
                    self.get_logger().info(f"根据深度相机，高度过低，上升,相机回传高度：{self.CurrentHeightFromCamera}")
                self.CurrentHeightFromCamera = 0.0  
        self.fly_to_position(target_x_NED, target_y_NED, z)
        self.get_logger().info(f"targetx:{target_x_NED:.4f},targety:{target_y_NED:.4f}")

    def adjust_to_target(self):
        """Adjust drone position towards the target."""
        if self.target_position:
            target_x_NED_f, target_y_NED_f = self.target_position.x, self.target_position.y
            target_x_NED, target_y_NED = target_x_NED_f, target_y_NED_f
            current_x, current_y = self.vehicle_local_position.x, self.vehicle_local_position.y
            distance = math.sqrt((current_x-target_x_NED)**2+(current_y-target_y_NED)**2)
            self.get_logger().info(f"距离目标距离：{distance}")
            
            if not self.first_alignment_checker.first_aligned:
                self.get_logger().info("第一次对准操作")
                self.check_isaround_fly(target_x_NED_f, target_y_NED_f, self.takeoff_target_height)  
                self.first_alignment_check(target_x_NED, target_y_NED)
                self.last_found_x_NED = target_x_NED_f
                self.last_found_y_NED = target_y_NED_f
                self.last_found_z_NED = self.takeoff_target_height

            elif self.first_alignment_checker.first_aligned and not self.second_alignment_checker.first_aligned:
                self.get_logger().info("第二次对准操作")
                self.check_isaround_fly(target_x_NED_f, target_y_NED_f, self.takeoff_target_height + self.afterAlign_descentHeight)
                self.second_alignment_check(target_x_NED, target_y_NED)
                self.last_found_x_NED = target_x_NED_f
                self.last_found_y_NED = target_y_NED_f
                self.last_found_z_NED = self.takeoff_target_height + self.afterAlign_descentHeight

            self.target_position = None
            if self.first_alignment_checker.first_aligned and self.second_alignment_checker.first_aligned:
                self.drop_payload()
                self.droping_x = self.vehicle_local_position.x
                self.droping_y = self.vehicle_local_position.y
                self.droping_z = self.vehicle_local_position.z
                self.is_FinishDrop = True
        else:
            if self.last_found_x_NED and self.last_found_y_NED and self.last_found_z_NED:
                self.fly_to_position(self.last_found_x_NED, self.last_found_y_NED, self.last_found_z_NED)
            else:
                self.get_logger().info("上次记录不存在")
                self.fly_to_position(self.DropArea_x, self.DropArea_y, self.takeoff_target_height)

    def rise_after_drop(self):
        """Rise to a certain height after dropping the payload."""
        # rise_target_height = self.vehicle_local_position.z + self.rise_height_after_drop
        rise_target_height = self.takeoff_target_height
        x,y=self.vehicle_local_position.x, self.vehicle_local_position.y
        self.fly_to_position(x, y, rise_target_height)
        self.get_logger().info(f"Rising to height: {rise_target_height}")

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal()
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.get_logger().info("try offboard")

        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if not self.is_ReadyToTakeoff:
                self.get_logger().info("执行步骤1,判断飞机起飞前位置是否稳定")
                Position = (
                    self.vehicle_local_position.x,
                    self.vehicle_local_position.y,
                    self.vehicle_local_position.z
                )
                self.get_logger().info(f"positon:{Position}")
                self.initPositionChecker.update_position(Position)
                if self.initPositionChecker.is_stable():
                    self.is_ReadyToTakeoff = True
                    self.arm()
                    self.get_logger().info("起飞前位置稳定。")

            if self.is_ReadyToTakeoff and not self.is_AtTakeoffHeight:
                self.get_logger().info("执行步骤2,上升到指定高度")
                self.takeoff_relative(self.takeoff_height)
                self.takeoff_height_check()

            if self.is_AtTakeoffHeight and not self.is_AtDropArea:
                self.get_logger().info("执行步骤3,飞向投水区")
                self.fly_forward(self.forward_x)
                self.fly_forward_check()

            if self.is_AtDropArea and not self.is_FinishDrop:
                self.adjust_to_target()

            if self.is_FinishDrop:
                if self.drop_count < 1:  # 检查是否还需要投放
                    if not self.is_rising_after_drop:
                        self.rise_after_drop()
                        self.is_rising_after_drop = True
                    else:
                        # 检查是否到达上升目标高度
                        target_height = self.takeoff_target_height  # 与rise_after_drop一致
                        if abs(self.vehicle_local_position.z - target_height) < 0.07:
                            self.is_rising_after_drop = False
                            self.is_FinishDrop = False
                            self.first_alignment_checker.first_aligned = False
                            self.second_alignment_checker.first_aligned = False
                            self.drop_count += 1
                            self.get_logger().info(f"准备进行第 {self.drop_count + 1} 次投放")
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