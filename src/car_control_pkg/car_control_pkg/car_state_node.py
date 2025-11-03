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
        self.get_logger().info(f'Received car state: {msg.data}')
        self.car_state = msg.data