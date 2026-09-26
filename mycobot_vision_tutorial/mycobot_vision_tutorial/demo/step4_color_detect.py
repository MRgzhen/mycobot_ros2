#!/usr/bin/env python3
"""Step-4: 在图像回调里做颜色检测 (HSV 阈值 -> 轮廓 -> 质心)。

前置: Gazebo 仿真运行中
运行: python3 step4_color_detect.py
改颜色: python3 step4_color_detect.py --color blue

核心流程 (与原版 color_detector.detect_blob 相同, 只是抽出成函数):
  BGR -> HSV -> inRange 生成掩码 -> 开/闭运算去噪 -> 轮廓
  -> 过滤小面积 -> 取最大轮廓 -> moments 求质心
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# OpenCV 色相范围 0-179, 红色跨 0 所以要两段
HSV_RANGES = {
    'red': [(0, 80, 60, 10, 255, 255), (170, 80, 60, 179, 255, 255)],
    'blue': [(100, 80, 60, 130, 255, 255)],
    'green': [(35, 60, 60, 85, 255, 255)],
    'yellow': [(20, 80, 60, 40, 255, 255)],
}
MIN_AREA = 150  # 小于这个面积(像素)的轮廓直接忽略


def detect_blob(bgr):
    """返回 (掩码, 最大色块质心 (cx, cy)); 没找到返回 (mask, None)。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in HSV_RANGES['red']:
        lo, hi = np.array(r[:3], np.uint8), np.array(r[3:], np.uint8)
        mask |= cv2.inRange(hsv, lo, hi)   # 多段范围用 |= 叠加

    # 开运算去掉零星噪点, 闭运算填补内部空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    if not contours:
        return mask, None

    largest = max(contours, key=cv2.contourArea)
    m = cv2.moments(largest)
    if m['m00'] == 0:
        return mask, None
    return mask, (m['m10'] / m['m00'], m['m01'] / m['m00'])


class ColorDetect(Node):

    def __init__(self):
        super().__init__('demo_color_detect')
        self.bridge = CvBridge()
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Image, '/camera_head/color/image_raw', self.on_image, qos)

    def on_image(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        mask, center = detect_blob(bgr)

        if center is not None:
            cx, cy = center
            cv2.drawMarker(bgr, (int(cx), int(cy)), (0, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)
            self.get_logger().info(
                f'blob at ({cx:.0f},{cy:.0f})', throttle_duration_sec=1.0)

        cv2.imshow('step4 color', bgr)
        cv2.imshow('step4 mask', mask)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ColorDetect()
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
