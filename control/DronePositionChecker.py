import math
from collections import deque
import time

class DronePositionChecker:
    def __init__(self, logger_func,tolerance=0.25, duration=3.0):
        """
        初始化无人机位置检查器。
        :param tolerance: 位置变化容差（米）。
        :param duration: 判断稳定所需的时间（秒）。
        """
        self.logger_func = logger_func
        self.tolerance = tolerance
        self.duration = duration
        self.positions = deque()  # 用于存储最近位置和时间戳

    def update_position(self, position):
        """
        更新无人机的当前位置。
        :param position: 位置元组 (x, y, z)。
        """
        current_time = time.time()
        self.positions.append((position, current_time))

        # 移除超出时间窗口的旧数据
        while self.positions and current_time - self.positions[0][1] > self.duration:
            self.positions.popleft()

    def is_stable(self):
        """
        判断无人机当前位置是否稳定。
        :return: 如果稳定返回 True，否则返回 False。
        """
        if len(self.positions) < 100:
            return False  # 数据不足时无法判断

        # 计算所有位置的最大距离
        max_distance = 0
        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                dist = self._distance(self.positions[i][0], self.positions[j][0])
                max_distance = max(max_distance, dist)

        self.logger_func(f"最大误差为{max_distance:.4f},m.")
        self.logger_func(f"{self.tolerance},return:{max_distance<=self.tolerance}")
        return max_distance <= self.tolerance

    @staticmethod
    def _distance(pos1, pos2):
        """
        计算两点之间的欧几里得距离。
        :param pos1: 点1 (x, y, z)。
        :param pos2: 点2 (x, y, z)。
        :return: 两点之间的距离。
        """
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))