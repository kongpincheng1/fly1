import time
from enum import Enum
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleOdometry
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from control.ServoControl import ServoControl

# =============================================================================
# 1. 模拟 ROS 2 环境的辅助类
# 这些类让我们可以在不依赖 ROS 2 的情况下测试逻辑
# =============================================================================

class MockLogger:
    """模拟ROS 2的 get_logger() 功能，只使用 print。"""
    def info(self, msg):
        print(f"[INFO] {msg}")

class MockServoControl:
    """模拟舵机控制器，只打印将要发送的指令。"""
    def open_servo(self, servo_1: float, servo_2: float):
        print(f"--- [SERVO COMMAND] ---> 发送指令: ({servo_1}, {servo_2}) ---")

class MockClock:
    """模拟ROS 2的 get_clock()，使用Python标准的 time 模块。"""
    def now(self) -> float:
        return time.time()

# =============================================================================
# 2. 从你的项目中复制的核心逻辑代码
# 这些代码与你整合到ROS节点中的代码是完全一样的
# =============================================================================

class DroppingState(Enum):
    """描述投水过程各个阶段的枚举。"""
    IDLE = 0              # 空闲，未在投水
    STEP_1_COMMANDED = 1  # 已发送指令(0, 0)，等待延迟
    STEP_2_COMMANDED = 2  # 已发送指令(0, 1)，等待延迟
    STEP_3_COMMANDED = 3
    STEP_4_COMMANDED = 4  # 已发送指令(1, 0)，等待最后延迟
    COMPLETED = 5         # 投水序列完成

class ServoTester:
    """
    一个封装了投水逻辑的测试类。
    它包含了所有需要的状态变量和函数。
    """
    def __init__(self, step_delay=0.1):
        # 初始化模拟环境
        self.logger = MockLogger()
        self.servo_control = ServoControl()
        self.clock = MockClock()

        # 初始化状态变量
        self.servo_step_delay = step_delay  # 每个舵机动作之间的延迟（秒）
        self.current_dropping_state = {1: DroppingState.IDLE, 2: DroppingState.IDLE}
        self.last_servo_command_time = {1: None, 2: None}

    def initiate_dropping_sequence(self, drop_number: int):
        """启动指定编号的投水序列的第一步。"""
        if self.current_dropping_state[drop_number] == DroppingState.IDLE:
            self.logger.info(f"启动第 {drop_number} 次投水序列...")
            self.logger.info(f"第 {drop_number} 次投水 - 步骤 1: (0, 0)")
            if drop_number == 1 :
                self.servo_control.open_servo(0.0, 1.0)
            elif drop_number ==2 :
                self.servo_control.open_servo(0.0, -1.0)
            self.current_dropping_state[drop_number] = DroppingState.STEP_1_COMMANDED
            self.last_servo_command_time[drop_number] = self.clock.now()

    def manage_dropping_sequence(self, drop_number: int) -> bool:
        """
        非阻塞地管理多步骤投水序列。
        返回: 如果本次投水序列已完成，则返回 True，否则返回 False。
        """
        state = self.current_dropping_state[drop_number]
        
        if state == DroppingState.IDLE:
            return False
        if state == DroppingState.COMPLETED:
            return True

        # 检查自上次命令以来是否经过了足够的时间
        elapsed_time = self.clock.now() - self.last_servo_command_time[drop_number]
        if elapsed_time < self.servo_step_delay:
            # 等待时间还不够，直接返回
            return False

        # --- 时间到了，推进到下一步 ---
        self.logger.info(f"已等待 {elapsed_time:.2f} 秒, 执行下一步...")

        if state == DroppingState.STEP_1_COMMANDED:
            
            if drop_number == 1:
                self.servo_control.open_servo(0.0, 1.0)
                self.logger.info(f"第 {drop_number} 次投水 - 步骤 2: (0, 1)")
            elif drop_number ==2:
                self.servo_control.open_servo(0.0, -1.0)
                self.logger.info(f"第 {drop_number} 次投水 - 步骤 2: (0, 1)")
            self.current_dropping_state[drop_number] = DroppingState.STEP_2_COMMANDED
            self.last_servo_command_time[drop_number] = self.clock.now()

        elif state == DroppingState.STEP_2_COMMANDED:
            self.logger.info(f"第 {drop_number} 次投水 - 步骤 3: (1, 0)")
            self.servo_control.open_servo(0.0, 0.0)
            self.current_dropping_state[drop_number] = DroppingState.STEP_3_COMMANDED
            self.last_servo_command_time[drop_number] = self.clock.now()
        
        elif state == DroppingState.STEP_3_COMMANDED:
            if drop_number == 1:
                self.servo_control.open_servo(1.0, 0.0)
                self.logger.info(f"第 {drop_number} 次投水 - 步骤 2: (0, 1)")
            elif drop_number ==2:
                self.servo_control.open_servo(-1.0, 0.0)
                self.logger.info(f"第 {drop_number} 次投水 - 步骤 2: (0, 1)")
            
            self.current_dropping_state[drop_number] = DroppingState.STEP_4_COMMANDED
            self.last_servo_command_time[drop_number] = self.clock.now()
        
        elif state == DroppingState.STEP_4_COMMANDED:
            self.servo_control.open_servo(0.0, 0.0)
            self.logger.info(f"第 {drop_number} 次投水序列完成。")
            self.current_dropping_state[drop_number] = DroppingState.COMPLETED
            return True
            
        return False

# =============================================================================
# 3. 测试执行器
# 这部分代码模拟了你的无人机主循环
# =============================================================================

def main(args=None): # ROS 2 节点通常接受 args 参数
    # 初始化测试器，设置每步之间延迟2秒，方便观察
    rclpy.init(args=args)
    tester = ServoTester(step_delay=0.3)

    # 模拟无人机任务的状态
    is_finish_1st_drop = False
    is_finish_2nd_drop = False

    print("\n" + "="*50)
    print("      开始模拟多步骤舵机投水测试")
    print("="*50 + "\n")

    # 启动第一次投水
    print("\n>>> [测试流程] >>> 满足条件，准备开始第一次投水...\n")
    tester.initiate_dropping_sequence(1)

    # 主循环，模拟 timer_callback
    while not is_finish_2nd_drop:

        # --- 管理第一次投水 ---
        if not is_finish_1st_drop:
            is_first_drop_done = tester.manage_dropping_sequence(1)
            if is_first_drop_done:
                print("\n>>> [测试流程] >>> 第一次投水确认完成！等待3秒后模拟第二次投水...\n")
                is_finish_1st_drop = True
                time.sleep(3) # 等待一下，让输出更清晰
                # 启动第二次投水
                tester.initiate_dropping_sequence(2)

        # --- 管理第二次投水 ---
        elif is_finish_1st_drop and not is_finish_2nd_drop:
            is_second_drop_done = tester.manage_dropping_sequence(2)
            if is_second_drop_done:
                print("\n>>> [测试流程] >>> 第二次投水确认完成！\n")
                is_finish_2nd_drop = True

        # 模拟定时器频率 (例如 10Hz)
        time.sleep(0.03)

    print("="*50)
    print("      所有投水流程已完成，测试结束。")
    print("="*50 + "\n")

if __name__ == "__main__":
    # 如果直接运行此文件，则调用 main()
    main()
