import numpy as np
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
        self.path_points = np.array(msg.data).reshape(-1, 2).tolist() # (N, 2) array of (x, z) points

    def get_near_points(self, current_position, num_points=10):
        if len(self.path_points) == 0:
            return []

        distances = np.linalg.norm(np.array(self.path_points) - current_position, axis=1)
        # Get the index of the nearest point
        index = np.argmin(distances)
        self.get_logger().info(f'Nearest path point index: {index}, position: {self.path_points[index]}')
        # Return the nearest points
        if index + num_points >= len(self.path_points):
            # If not enough points ahead, pad with the last point
            near_points = self.path_points[index:]
            padding = np.tile(self.path_points[-1], (num_points - len(near_points), 1))
            near_points = np.vstack([near_points, padding])
        else:
            near_points = self.path_points[index:index + num_points]
        return near_points