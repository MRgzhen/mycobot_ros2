#!/usr/bin/env python3
"""Step-1b: 最小订阅器。订阅 /demo/chatter 并打印。

运行:  先跑 step1_talker.py, 再开一个终端
       python3 step1_listener.py

知识点: 订阅 = 注册一个回调函数, 消息到达时被自动调用
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Listener(Node):

    def __init__(self):
        super().__init__('demo_listener')
        # 订阅三要素: 消息类型 / 话题名 / 回调函数
        self.create_subscription(String, '/demo/chatter', self.on_msg, 10)

    def on_msg(self, msg):
        self.get_logger().info(f'heard: {msg.data}')


def main():
    rclpy.init()
    node = Listener()
    try:
        rclpy.spin(node)  # 阻塞, 等消息到达并执行回调
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
