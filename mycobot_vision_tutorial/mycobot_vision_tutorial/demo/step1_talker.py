#!/usr/bin/env python3
"""Step-1a: 最小发布器。每 0.5 秒发一条字符串到 /demo/chatter。

运行:  source /opt/ros/humble/setup.bash
       python3 step1_talker.py
验证:  ros2 topic list
       ros2 topic echo /demo/chatter
       ros2 topic hz /demo/chatter
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Talker(Node):

    def __init__(self):
        super().__init__('demo_talker')
        # 发布器三要素: 消息类型 / 话题名 / 队列深度
        self.pub = self.create_publisher(String, '/demo/chatter', 10)
        # 定时器: 每 0.5 秒触发一次回调
        self.timer = self.create_timer(0.5, self.tick)
        self.count = 0

    def tick(self):
        msg = String()
        msg.data = f'hello {self.count}'
        self.pub.publish(msg)
        self.count += 1


def main():
    rclpy.init()
    node = Talker()
    try:
        rclpy.spin(node)  # 阻塞在这里, 反复执行定时器回调
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
