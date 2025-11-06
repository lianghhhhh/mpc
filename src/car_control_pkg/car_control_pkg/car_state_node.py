import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class CarStateNode(Node):
    def __init__(self):
        super().__init__('car_state_node')
        self.get_logger().info('Car State Node has been started.')
        self.car_state_subscriber = self.create_subscription(
            Float32MultiArray,
            'car_state',
            self.car_state_callback,
            10
        )
        self.car_state = []

    def car_state_callback(self, msg):
        # self.get_logger().info(f'Received car state: {msg.data}')
        data = msg.data
        angle = np.radians(data[2])  # Convert degrees to radians
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
        self.car_state = [data[0], data[1], sin_angle, cos_angle]