#!/usr/bin/env python3
"""Step-8: 目标检测 + 跟踪 —— HSV 检测找到目标, CSRT 跟踪器持续跟着它 (不发布任何话题)。

检测和跟踪分工:
  检测 (每帧 HSV 分割): 告诉我们"目标在哪", 但光照突变/遮挡时会失手
  跟踪 (CSRT 相关滤波): 记住目标外观持续跟, 检测短暂丢失时顶上

  检测命中   -> 用检测框, 并重新初始化跟踪器 (绿框, "detect")
  检测丢失   -> 用跟踪器输出继续跟 (蓝框, "track")
  两者都丢   -> 丢弃跟踪器, 回到全图重新搜索 (画面提示 "searching")

前置: Gazebo 仿真运行中, 场景里放一个目标色块 (默认红色)
运行: python3 step8_detect_track.py
改颜色: python3 step8_detect_track.py --color blue
提示: CSRT 精但慢 (小画面没问题); 想快可以把 TrackerCSRT 换成 TrackerKCF

知识点:
  1. cv2.TrackerCSRT: init() 用第一帧+框初始化, update() 每帧返回新框
  2. 检测/跟踪互补的朴素状态机: detect -> track -> searching
  3. 轨迹 (deque 存最近 N 个质心) 和帧间速度估计
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# 与 step2/项目 color_detector.py 同款参数
HSV_RANGES = {
    'red': [(0, 80, 60, 10, 255, 255), (170, 80, 60, 179, 255, 255)],
    'blue': [(100, 80, 60, 130, 255, 255)],
    'green': [(35, 60, 60, 85, 255, 255)],
    'yellow': [(20, 80, 60, 40, 255, 255)],
}
MIN_AREA = 150
TRAJECTORY_LEN = 60  # 轨迹保留多少帧


def detect_box(bgr, color):
    """返回 (mask, box); box = (x, y, w, h), 没找到为 None。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in HSV_RANGES[color]:
        lo, hi = np.array(r[:3], np.uint8), np.array(r[3:], np.uint8)
        mask |= cv2.inRange(hsv, lo, hi)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    if not contours:
        return mask, None
    return mask, cv2.boundingRect(max(contours, key=cv2.contourArea))


def box_center(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


class DetectTrack(Node):

    def __init__(self, color):
        super().__init__('demo_detect_track')
        self.color = color
        self.bridge = CvBridge()

        self.tracker = None       # 没初始化时为 None
        self.traj = deque(maxlen=TRAJECTORY_LEN)   # 最近 N 帧质心
        self.last_center = None   # 上一帧 (center, 时间), 用于算速度

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Image, '/camera_head/color/image_raw', self.on_image, qos)
        self.get_logger().info(f"detect + track '{color}', no publishing")

    def on_image(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        now = time.perf_counter()

        mask, det = detect_box(bgr, self.color)

        box, label, color_rgb = None, '', (0, 255, 255)
        if det is not None:
            # ---- 检测命中: 用检测框, (重)初始化跟踪器 ----
            box, label, color_rgb = det, 'detect', (0, 255, 0)
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(bgr, det)
        elif self.tracker is not None:
            # ---- 检测丢失: 让跟踪器顶上 ----
            ok, trk = self.tracker.update(bgr)
            if ok:
                box, label, color_rgb = tuple(map(int, trk)), 'track', (255, 128, 0)
            else:
                # 跟踪器也丢了: 丢弃, 下一帧开始全图重新搜索
                self.tracker = None

        speed = 0.0
        if box is not None:
            cx, cy = box_center(box)

            # 轨迹和速度 (与上一有效帧的位移 / 时间)
            self.traj.append((int(cx), int(cy)))
            if self.last_center is not None:
                (px, py), pt = self.last_center
                dt = now - pt
                if dt > 1e-3:
                    speed = np.hypot(cx - px, cy - py) / dt
            self.last_center = ((cx, cy), now)

            x, y, w, h = box
            cv2.rectangle(bgr, (x, y), (x + w, y + h), color_rgb, 2)
            cv2.putText(bgr, f'{label} ({cx:.0f},{cy:.0f}) {speed:.0f}px/s',
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color_rgb, 2)
            self.get_logger().info(
                f'{label}: center=({cx:.0f},{cy:.0f}) speed={speed:.0f}px/s',
                throttle_duration_sec=1.0)
        else:
            self.last_center = None
            cv2.putText(bgr, 'searching...', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if len(self.traj) >= 2:
            cv2.polylines(bgr, [np.array(self.traj, np.int32)],
                          False, (255, 0, 255), 2)

        cv2.imshow('step8 track', bgr)
        cv2.imshow('step8 mask', mask)
        cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--color', choices=list(HSV_RANGES), default='red')
    args = parser.parse_args()

    rclpy.init()
    node = DetectTrack(args.color)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
