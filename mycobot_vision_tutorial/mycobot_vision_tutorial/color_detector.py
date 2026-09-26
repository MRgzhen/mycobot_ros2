#!/usr/bin/env python3
"""Stage-1 vision warm-up node: color segmentation -> centroid -> 3D point.

Pipeline (per synced color+depth frame):
  1. Color image -> HSV threshold -> morphological cleanup -> contours
  2. Largest contour -> pixel centroid (u, v)
  3. Depth median in a small window around (u, v)
  4. Back-project with pinhole intrinsics: x=(u-cx)*z/fx, y=(v-cy)*z/fy
     (result lives in the optical frame)
  5. TF transform optical frame -> target frame (default base_link)
  6. Publish PointStamped + RViz Marker + annotated debug image

Run together with mycobot.gazebo.launch.py (use_camera:=true).
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

import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform support)
from visualization_msgs.msg import Marker

# OpenCV hue range is 0-179. Each preset is a list of
# (h_min, s_min, v_min, h_max, s_max, v_max) ranges (red wraps around 0).
COLOR_PRESETS = {
    'red': [(0, 80, 60, 10, 255, 255), (170, 80, 60, 179, 255, 255)],
    'blue': [(100, 80, 60, 130, 255, 255)],
    'green': [(35, 60, 60, 85, 255, 255)],
    'yellow': [(20, 80, 60, 40, 255, 255)],
}


def parse_hsv(text):
    """Parse 'h,s,v' (each 0-255, hue mapped to 0-179) into a tuple."""
    parts = [int(p) for p in text.split(',')]
    if len(parts) != 3:
        raise ValueError('expected "h,s,v"')
    return (min(parts[0], 179), parts[1], parts[2])


class ColorDetector(Node):

    def __init__(self):
        super().__init__('color_detector')

        # ---------------- parameters ----------------
        self.declare_parameter('target_color', 'red')
        self.declare_parameter('hsv_min', '0,80,60')
        self.declare_parameter('hsv_max', '10,255,255')
        self.declare_parameter('min_area', 150)          # px
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('optical_frame', 'camera_head_depth_optical_frame')
        self.declare_parameter('depth_max', 1.5)         # m, sensor far clip
        self.declare_parameter('depth_window', 5)        # px, median window

        self.target_color = self.get_parameter('target_color').value
        self.min_area = int(self.get_parameter('min_area').value)
        self.target_frame = self.get_parameter('target_frame').value
        self.optical_frame = self.get_parameter('optical_frame').value
        self.depth_max = float(self.get_parameter('depth_max').value)
        self.depth_window = int(self.get_parameter('depth_window').value)

        if self.target_color in COLOR_PRESETS:
            self.hsv_ranges = [
                (tuple(r[:3]), tuple(r[3:])) for r in COLOR_PRESETS[self.target_color]
            ]
        elif self.target_color == 'custom':
            lo = parse_hsv(self.get_parameter('hsv_min').value)
            hi = parse_hsv(self.get_parameter('hsv_max').value)
            self.hsv_ranges = [(lo, hi)]
        else:
            raise ValueError(
                f"unknown target_color '{self.target_color}', "
                f'choose from {list(COLOR_PRESETS) + ["custom"]}')

        # ---------------- camera intrinsics (filled by CameraInfo) ---------
        self.fx = self.fy = self.cx0 = self.cy0 = None

        # ---------------- TF ----------------
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # ---------------- ROS I/O ----------------
        self.bridge = CvBridge()

        image_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            CameraInfo, '/camera_head/depth/camera_info', self.info_cb, 10)

        color_sub = message_filters.Subscriber(
            self, Image, '/camera_head/color/image_raw', qos_profile=image_qos)
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera_head/depth/image_rect_raw', qos_profile=image_qos)
        sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.1)
        sync.registerCallback(self.on_images)

        self.pub_point = self.create_publisher(PointStamped, '/vision/detected_point', 10)
        self.pub_marker = self.create_publisher(Marker, '/vision/marker', 10)
        self.pub_debug = self.create_publisher(Image, '/vision/debug_image', 10)

        self.get_logger().info(
            f"looking for '{self.target_color}' blobs >= {self.min_area} px, "
            f"output frame '{self.target_frame}'")

    # ------------------------------------------------------------------
    def info_cb(self, msg):
        k = msg.k  # 3x3 row-major intrinsics
        if self.fx is None:
            self.fx, self.fy, self.cx0, self.cy0 = k[0], k[4], k[2], k[5]
            self.get_logger().info(
                f'camera intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} '
                f'cx={self.cx0:.1f} cy={self.cy0:.1f}')

    # ------------------------------------------------------------------
    def on_images(self, color_msg, depth_msg):
        if self.fx is None:  # camera_info not yet received
            self.get_logger().info('waiting for camera intrinsics...',
                                   throttle_duration_sec=3.0)
            return

        bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)

        debug = bgr.copy()
        found = self.detect_blob(bgr, debug)

        if found is None:
            self.get_logger().warning('no blob found', throttle_duration_sec=2.0)
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
            return

        cx, cy = found

        # ---- depth at centroid (median over a small window) ----
        z = self.read_depth(depth, cx, cy)
        if z is None:
            cv2.putText(debug, 'no valid depth', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
            return

        # ---- pinhole back-projection -> optical frame ----
        x = (cx - self.cx0) * z / self.fx
        y = (cy - self.cy0) * z / self.fy

        self.draw_result(debug, cx, cy, x, y, z)
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

        # ---- optical frame -> target frame ----
        point = PointStamped()
        point.header.frame_id = self.optical_frame
        point.header.stamp = color_msg.header.stamp
        point.point.x, point.point.y, point.point.z = float(x), float(y), float(z)

        try:
            # camera is rigidly mounted: latest transform is fine
            point_tf = self.tf_buffer.transform(
                point, self.target_frame, timeout=Duration(seconds=0.2))
        except TransformException as err:
            self.get_logger().warning(
                f'TF {self.optical_frame} -> {self.target_frame} failed: {err}',
                throttle_duration_sec=2.0)
            return

        self.pub_point.publish(point_tf)
        self.pub_marker.publish(self.make_marker(point_tf))
        p = point_tf.point
        self.get_logger().info(
            f'blob at pixel ({cx:.0f},{cy:.0f}) depth {z:.3f} m -> '
            f'{self.target_frame}: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}',
            throttle_duration_sec=2.0)

    # ------------------------------------------------------------------
    def detect_blob(self, bgr, debug):
        """Return (cx, cy) of the largest color blob, or None."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in self.hsv_ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        m = cv2.moments(largest)
        if m['m00'] == 0:
            return None
        cx = m['m10'] / m['m00']
        cy = m['m01'] / m['m00']

        cv2.drawContours(debug, [largest], -1, (0, 255, 0), 2)
        return cx, cy

    # ------------------------------------------------------------------
    def read_depth(self, depth, cx, cy):
        """Median depth in a window around (cx, cy); None if nothing valid."""
        h, w = depth.shape
        r = max(1, self.depth_window // 2)
        u0, u1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
        v0, v1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
        win = depth[v0:v1, u0:u1].ravel()
        valid = win[np.isfinite(win) & (win > 0.01) & (win < self.depth_max)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    # ------------------------------------------------------------------
    def draw_result(self, debug, cx, cy, x, y, z):
        h, w = debug.shape[:2]
        cv2.line(debug, (int(cx) - 15, int(cy)), (int(cx) + 15, int(cy)), (0, 255, 255), 2)
        cv2.line(debug, (int(cx), int(cy) - 15), (int(cx), int(cy) + 15), (0, 255, 255), 2)
        text = [f'px=({cx:.0f},{cy:.0f})',
                f'optical: x={x:.3f} y={y:.3f} z={z:.3f} m']
        for i, line in enumerate(text):
            cv2.putText(debug, line, (10, 25 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ------------------------------------------------------------------
    def make_marker(self, point):
        marker = Marker()
        marker.header = point.header
        marker.ns = 'vision'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.03
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.2, 0.2, 1.0
        marker.lifetime = Duration(seconds=1.0).to_msg()
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = ColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
