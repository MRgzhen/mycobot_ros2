#!/usr/bin/env python3
"""Step-6: 加深度图 + 相机内参, 把像素反投影成光学系下的 3D 坐标。

前置: Gazebo 仿真运行中
运行: python3 step6_depth_3d.py

三个新知识点:
  1. message_filters 时间同步: 彩色图和深度图必须来自同一时刻才配对
  2. camera_info 里的内参 K 矩阵: fx, fy, cx, cy
  3. 针孔模型反投影 (结果在光学系, 单位米):
       x = (u - cx) * z / fx
       y = (v - cy) * z / fy
"""

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

HSV_RANGES = [(0, 80, 60, 10, 255, 255), (170, 80, 60, 179, 255, 255)]  # red
MIN_AREA = 150
DEPTH_MAX = 1.5   # 米, 超过视为无效
WINDOW = 5        # 质心周围取中值的窗口大小 (像素)


def detect_blob(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in HSV_RANGES:
        mask |= cv2.inRange(hsv, np.array(r[:3], np.uint8),
                            np.array(r[3:], np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    if not contours:
        return None
    m = cv2.moments(max(contours, key=cv2.contourArea))
    if m['m00'] == 0:
        return None
    return m['m10'] / m['m00'], m['m01'] / m['m00']


def read_depth(depth, cx, cy):
    """质心周围小窗口内取深度中值 (比单点更抗噪); 无有效值返回 None。"""
    h, w = depth.shape
    r = max(1, WINDOW // 2)
    win = depth[max(0, int(cy) - r):int(cy) + r + 1,
                max(0, int(cx) - r):int(cx) + r + 1].ravel()
    valid = win[np.isfinite(win) & (win > 0.01) & (win < DEPTH_MAX)]
    return float(np.median(valid)) if valid.size else None


class Depth3D(Node):

    def __init__(self):
        super().__init__('demo_depth_3d')
        self.bridge = CvBridge()

        # 内参: 收到第一条 camera_info 后填充
        self.fx = self.fy = self.cx0 = self.cy0 = None
        self.create_subscription(
            CameraInfo, '/camera_head/depth/camera_info', self.info_cb, 10)

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        color_sub = message_filters.Subscriber(
            self, Image, '/camera_head/color/image_raw',
            qos_profile=sensor_qos)
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera_head/depth/image_rect_raw',
            qos_profile=sensor_qos)
        # slop=0.1: 两帧时间戳相差 0.1 秒内就算"同一时刻"
        sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.1)
        sync.registerCallback(self.on_images)

    def info_cb(self, msg):
        if self.fx is None:
            k = msg.k  # 3x3 行优先内参矩阵
            self.fx, self.fy, self.cx0, self.cy0 = k[0], k[4], k[2], k[5]
            self.get_logger().info(
                f'intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} '
                f'cx={self.cx0:.1f} cy={self.cy0:.1f}')

    def on_images(self, color_msg, depth_msg):
        if self.fx is None:  # 还没拿到内参, 无法反投影
            return

        bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg,
                                          desired_encoding='passthrough')
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)

        found = detect_blob(bgr)
        if found is None:
            return
        cx, cy = found

        z = read_depth(depth, cx, cy)
        if z is None:
            self.get_logger().warning('no valid depth', throttle_duration_sec=2.0)
            return

        # ---- 核心: 针孔模型反投影 ----
        x = (cx - self.cx0) * z / self.fx
        y = (cy - self.cy0) * z / self.fy

        self.get_logger().info(
            f'px=({cx:.0f},{cy:.0f}) -> optical: '
            f'x={x:.3f} y={y:.3f} z={z:.3f} m', throttle_duration_sec=1.0)


def main():
    rclpy.init()
    node = Depth3D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
