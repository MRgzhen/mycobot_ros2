#!/usr/bin/env python3
"""Step-7: TF 坐标变换 + 发布 PointStamped / Marker。

在 step6 基础上, 把光学系下的 3D 点变换到 base_link (机器人基座坐标系),
发布成 PointStamped 和 RViz 球体 Marker。到这里就回到原版 color_detector 了。

前置: Gazebo 仿真运行中
查看: rviz2 -> Add -> Marker, Fixed Frame 选 base_link

知识点:
  1. tf2_ros.Buffer + TransformListener: 查询两坐标系间的变换
  2. tf_buffer.transform(): 把某个坐标系下的点变换到目标坐标系
  3. Marker: 给 RViz 看的可视化图元
"""

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

import tf2_geometry_msgs  # noqa: F401  注册 PointStamped 的变换支持
from visualization_msgs.msg import Marker

TARGET_FRAME = 'base_link'
OPTICAL_FRAME = 'camera_head_depth_optical_frame'

HSV_RANGES = [(0, 80, 60, 10, 255, 255), (170, 80, 60, 179, 255, 255)]  # red
MIN_AREA = 150
DEPTH_MAX = 1.5
WINDOW = 5


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
    h, w = depth.shape
    r = max(1, WINDOW // 2)
    win = depth[max(0, int(cy) - r):int(cy) + r + 1,
                max(0, int(cx) - r):int(cx) + r + 1].ravel()
    valid = win[np.isfinite(win) & (win > 0.01) & (win < DEPTH_MAX)]
    return float(np.median(valid)) if valid.size else None


class TfMarker(Node):

    def __init__(self):
        super().__init__('demo_tf_marker')
        self.bridge = CvBridge()

        self.fx = self.fy = self.cx0 = self.cy0 = None
        self.create_subscription(
            CameraInfo, '/camera_head/depth/camera_info', self.info_cb, 10)

        # ---- TF: 监听并缓存整棵坐标系树 ----
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self,
                                             spin_thread=True)

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
        sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.1)
        sync.registerCallback(self.on_images)

        self.pub_point = self.create_publisher(
            PointStamped, '/vision/detected_point', 10)
        self.pub_marker = self.create_publisher(Marker, '/vision/marker', 10)

    def info_cb(self, msg):
        if self.fx is None:
            k = msg.k
            self.fx, self.fy, self.cx0, self.cy0 = k[0], k[4], k[2], k[5]

    def on_images(self, color_msg, depth_msg):
        if self.fx is None:
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
            return

        # 反投影: 光学系下的点
        x = (cx - self.cx0) * z / self.fx
        y = (cy - self.cy0) * z / self.fy

        # 包装成 PointStamped (带坐标系和时间戳的消息点)
        point = PointStamped()
        point.header.frame_id = OPTICAL_FRAME
        point.header.stamp = color_msg.header.stamp
        point.point.x, point.point.y, point.point.z = \
            float(x), float(y), float(z)

        try:
            # 相机刚性固定: 用最新变换即可
            point_tf = self.tf_buffer.transform(
                point, TARGET_FRAME, timeout=Duration(seconds=0.2))
        except TransformException as err:
            self.get_logger().warning(
                f'TF {OPTICAL_FRAME} -> {TARGET_FRAME} failed: {err}',
                throttle_duration_sec=2.0)
            return

        self.pub_point.publish(point_tf)
        self.pub_marker.publish(self.make_marker(point_tf))
        p = point_tf.point
        self.get_logger().info(
            f'{TARGET_FRAME}: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} m',
            throttle_duration_sec=1.0)

    def make_marker(self, point):
        """在检测位置放一个 3cm 的红色小球, 给 RViz 看。"""
        marker = Marker()
        marker.header = point.header
        marker.ns, marker.id = 'vision', 0
        marker.type, marker.action = Marker.SPHERE, Marker.ADD
        marker.pose.position = point.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.03
        marker.color.r, marker.color.g = 1.0, 0.2
        marker.color.b, marker.color.a = 0.2, 1.0
        marker.lifetime = Duration(seconds=1.0).to_msg()
        return marker


def main():
    rclpy.init()
    node = TfMarker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
