#!/usr/bin/env python3
"""Step-5: 订阅 -> 处理 -> 发布 完整闭环。

订阅彩色图 -> 检测红色色块 -> 把标注后的调试图发布到 /vision/debug_image

前置: Gazebo 仿真运行中
运行: python3 step5_publish_debug.py
查看: ros2 run rqt_image_view rqt_image_view
      然后在下拉框选 /vision/debug_image

知识点: 图像发布 (cv2_to_imgmsg), 订阅和处理串在同一个回调里
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

HSV_RANGES = {
    'red': [(0, 80, 60, 10, 255, 255), (170, 80, 60, 179, 255, 255)],
    'blue': [(100, 80, 60, 130, 255, 255)],
    'green': [(35, 60, 60, 85, 255, 255)],
    'yellow': [(20, 80, 60, 40, 255, 255)],
}
MIN_AREA = 150


def detect_blob(bgr):
    """返回 (最大色块质心, 带轮廓标注的图); 没找到返回 (None, bgr)。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in HSV_RANGES['red']:
        mask |= cv2.inRange(hsv, np.array(r[:3], np.uint8),
                            np.array(r[3:], np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    if not contours:
        return None, bgr

    largest = max(contours, key=cv2.contourArea)
    m = cv2.moments(largest)
    if m['m00'] == 0:
        return None, bgr

    cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
    cv2.drawContours(bgr, [largest], -1, (0, 255, 0), 2)   # 绿色描边
    cv2.drawMarker(bgr, (int(cx), int(cy)), (0, 255, 255),
                   cv2.MARKER_CROSS, 20, 2)
    cv2.putText(bgr, f'px=({cx:.0f},{cy:.0f})', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return (cx, cy), bgr


class DebugPublisher(Node):

    def __init__(self):
        super().__init__('demo_debug_publisher')
        self.bridge = CvBridge()

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Image, '/camera_head/color/image_raw', self.on_image, sensor_qos)

        # 发布端用默认 RELIABLE 即可 (rqt 能收)
        self.pub_debug = self.create_publisher(Image, '/vision/debug_image', 10)

    def on_image(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        center, debug = detect_blob(bgr)

        if center is None:
            self.get_logger().warning('no blob', throttle_duration_sec=2.0)

        # OpenCV 图 -> ROS 图像消息, 发出去
        self.pub_debug.publish(
            self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))


def main():
    rclpy.init()
    node = DebugPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
