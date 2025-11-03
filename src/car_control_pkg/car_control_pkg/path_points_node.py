from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class PathPointsNode(Node):
    def __init__(self):
        super().__init__('path_points_node')
        self.get_logger().info('Path Points Node has been started.')
        self.path_points_subscriber = self.create_subscription(
            Float32MultiArray,
            'path_points',
            self.path_points_callback,
            10
        )
        self.path_points = []

    def path_points_callback(self, msg):
        self.get_logger().info(f'Received path points: {msg.data}')
        self.path_points = msg.data