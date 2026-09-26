import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    
    def __ini__(self):
        super().__init__('listener')
        self.create_subscription(String, )