#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import numpy as np
import time

class TargetPositionPublisher(Node):
    """Node to simulate publishing target positions of three buckets."""

    def __init__(self):
        super().__init__('target_position_publisher')

        # 创建发布者，发布到 /target_position 话题
        self.publisher = self.create_publisher(Point, '/target_position', 10)

        # 定义三个桶的固定坐标（NED 坐标系，单位：米）
        self.bucket_positions = [
            [1.0, 1.0],  # 桶 1
            [2.0, 2.5],  # 桶 2
            [3.5, 1.5]   # 桶 3
        ]

        # 创建定时器，每 0.1 秒发布一次数据（10 Hz）
        self.timer = self.create_timer(0.5, self.timer_callback)

        # 用于循环发布的计数器
        self.bucket_index = 0

    def timer_callback(self):
        """Callback function to publish target positions."""
        # 创建 Point 消息
        msg = Point()

        # 选择当前要发布的桶坐标
        current_bucket = self.bucket_positions[self.bucket_index]
        
        # 添加轻微噪声，模拟相机检测的实际情况
        noise_x = np.random.normal(0, 0.05)  # 均值为 0，标准差为 0.05 米的噪声
        noise_y = np.random.normal(0, 0.05)
        
        # 设置消息的 x, y 坐标（z 设为 0，因为桶在地面）
        msg.x = current_bucket[0] + noise_x
        msg.y = current_bucket[1] + noise_y
        msg.z = 0.0

        # 发布消息
        self.publisher.publish(msg)
        self.get_logger().info(f"Published target position: x={msg.x:.3f}, y={msg.y:.3f}, z={msg.z:.3f}")

        # 切换到下一个桶
        self.bucket_index = (self.bucket_index + 1) % 3  # 循环 0, 1, 2

def main(args=None):
    rclpy.init(args=args)
    target_position_publisher = TargetPositionPublisher()
    try:
        rclpy.spin(target_position_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        target_position_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()