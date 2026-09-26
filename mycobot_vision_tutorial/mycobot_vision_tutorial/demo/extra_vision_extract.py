#!/usr/bin/env python3
"""选做练习: 视觉提取 —— 在实时相机流上把目标物体"抠"出来 (纯本地, 不发布任何话题)。

step2 (纯 OpenCV) 和 step4 (颜色检测) 的综合练习: 订阅仿真相机彩色图,
用 HSV 分割找到最大色块, 画外接框并裁剪出目标区域 (ROI) 单独显示。
全程只有 imshow, 没有 publisher。

前置: 先启动 Gazebo 仿真 (带相机)
  ros2 launch mycobot_bringup mycobot.gazebo.launch.py use_camera:=true
运行: python3 extra_vision_extract.py
改颜色: python3 extra_vision_extract.py --color blue

知识点:
  1. 传感器数据 QoS 要用 BEST_EFFORT (宁可丢帧也不要旧帧)
  2. cv_bridge: sensor_msgs/Image <-> OpenCV numpy 数组
  3. HSV 阈值分割 -> 轮廓 -> boundingRect 外接框 -> numpy 切片裁 ROI
"""

import argparse

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# 与项目 color_detector.py 的 COLOR_PRESETS 保持一致
HSV_RANGES = {
    'red': [(0, 80, 60, 10, 255, 255), (170, 80, 60, 179, 255, 255)],
    'blue': [(100, 80, 60, 130, 255, 255)],
    'green': [(35, 60, 60, 85, 255, 255)],
    'yellow': [(20, 80, 60, 40, 255, 255)],
}
MIN_AREA = 150  # 小于这个面积(像素)的轮廓直接忽略


def extract_target(bgr, color):
    """提取指定颜色的最大目标。

    返回 (mask, box, center, area):
      mask   二值掩码 (目标=白)
      box    外接框 (x, y, w, h), 没找到为 None
      center 质心 (cx, cy), 没找到为 None
      area   轮廓面积(像素), 没找到为 0
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 红色跨 0 度, 预设里有两段范围, 用 |= 叠加成一个掩码
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in HSV_RANGES[color]:
        lo, hi = np.array(r[:3], np.uint8), np.array(r[3:], np.uint8)
        mask |= cv2.inRange(hsv, lo, hi)

    # 开运算去掉零星噪点, 闭运算填补内部空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    if not contours:
        return mask, None, None, 0

    largest = max(contours, key=cv2.contourArea)
    m = cv2.moments(largest)
    if m['m00'] == 0:
        return mask, None, None, 0

    box = cv2.boundingRect(largest)
    center = (m['m10'] / m['m00'], m['m01'] / m['m00'])
    return mask, box, center, cv2.contourArea(largest)


class VisionExtract(Node):

    def __init__(self, color):
        super().__init__('demo_vision_extract')
        self.color = color
        self.bridge = CvBridge()
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Image, '/camera_head/color/image_raw', self.on_image, qos)
        self.get_logger().info(f"extracting '{color}' target, no publishing")

    def on_image(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        mask, box, center, area = extract_target(bgr, self.color)

        roi = None
        if box is not None:
            x, y, w, h = box
            cx, cy = center

            # 外接框 + 质心十字 + 信息文字
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawMarker(bgr, (int(cx), int(cy)), (0, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(bgr, f'{self.color} area={area:.0f}px', (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # numpy 切片裁出目标本身 —— "提取"的结果
            roi = bgr[y:y + h, x:x + w]

            self.get_logger().info(
                f'box=({x},{y},{w},{h}) center=({cx:.0f},{cy:.0f}) '
                f'area={area:.0f}px', throttle_duration_sec=1.0)

        # ROI 窗口尺寸随目标变化, 没目标时给一块黑图占位
        if roi is None:
            roi = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.putText(roi, 'no target', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('extract image', bgr)
        cv2.imshow('extract mask', mask)
        cv2.imshow('extract roi', roi)
        cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--color', choices=list(HSV_RANGES), default='red')
    args = parser.parse_args()

    rclpy.init()
    node = VisionExtract(args.color)
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
