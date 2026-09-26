#!/usr/bin/env python3
"""Step-3: 订阅仿真相机彩色图并显示。

前置: 先启动 Gazebo 仿真 (带相机)
  ros2 launch mycobot_bringup mycobot.gazebo.launch.py use_camera:=true
运行: python3 step3_image_listener.py

知识点:
  1. 传感器数据 QoS 要用 BEST_EFFORT (宁可丢帧也不要旧帧)
  2. cv_bridge: sensor_msgs/Image <-> OpenCV numpy 数组
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class ImageListener(Node):

    def __init__(self):
        super().__init__('demo_image_listener')
        self.bridge = CvBridge()

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Image, '/camera_head/color/image_raw', self.on_image, qos)

    def on_image(self, msg):
        # ROS 图像消息 -> OpenCV BGR 图 (numpy 数组)
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('step3 color', bgr)
        # imshow 必须配 waitKey 才会真正刷新窗口
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ImageListener()
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
