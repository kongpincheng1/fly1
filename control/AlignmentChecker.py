from collections import deque
import time
import math
class AlignmentChecker:
    def __init__(self, logger_func, threshold=0.15, time_window=2.0, check_frequency=10):
        """
        :param logger_func: 日志记录函数（例如 Node.get_logger().info）
        :param threshold: 误差阈值 (米)
        :param time_window: 时间窗口 (秒)
        :param check_frequency: 误差检查频率 (每秒次数)
        """
        self.logger_func = logger_func
        self.threshold = threshold
        self.time_window = time_window
        self.check_frequency = check_frequency
        self.error_deque = deque(maxlen=int(time_window * check_frequency))  # 固定大小的队列
        self.first_aligned = False

    def alignment_check(self, current_x, current_y, target_x, target_y):
        # 计算当前位置与目标点的误差
        result = math.sqrt(
            (current_x - target_x)**2 + (current_y - target_y)**2
        )

        # 将误差记录到队列中
        self.error_deque.append(result)

        # 检查队列是否已满
        if len(self.error_deque) == self.error_deque.maxlen:
            # 判断队列内所有误差是否小于阈值
            if all(error < self.threshold for error in self.error_deque):
                self.first_aligned = True
                self.logger_func(f"目标对准成功：误差连续 {len(self.error_deque)} 次小于阈值 {self.threshold}")
                self.error_deque.clear()
                return True

        # 打印当前误差和队列状态
        # self.logger_func(f"当前误差： {result:.3f},")
        return False
