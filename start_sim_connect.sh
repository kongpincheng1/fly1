#!/bin/bash

# 清理函数，终止所有相关进程
cleanup() {
    echo "收到 Ctrl+C，正在终止所有进程..."
    # 杀死记录的 PID
    kill -9 $SIM_PID 2>/dev/null
    kill -9 $CONNECT_PID 2>/dev/null
    # 确保清理所有可能的子进程
    pkill -9 -f "px4|gz_x500|gz_bridge|MicroXRCEAgent" 2>/dev/null
    exit 0
}

# 捕获 Ctrl+C 信号
trap cleanup SIGINT

echo "start sim"
cd "$HOME" || { echo "无法切换到主目录"; exit 1; }
cd PX4-Autopilot || { echo "PX4-Autopilot 目录不存在"; exit 1; }
PX4_GZ_SIM_RENDER_ENGINE=ogre make px4_sitl gz_x500 &
SIM_PID=$!
echo "仿真 PID: $SIM_PID"

sleep 1

echo "start connect"
cd "$HOME" || { echo "无法切换到主目录"; exit 1; }
MicroXRCEAgent udp4 -p 8888 &
CONNECT_PID=$!
echo "连接 PID: $CONNECT_PID"

# 等待后台进程完成
wait $SIM_PID
wait $CONNECT_PID

echo "所有进程已正常结束"