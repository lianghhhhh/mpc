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
        self.path_points = msg.data

    def get_near_points(self, current_position, num_points=10):
        if not self.path_points:
            return []

        distances = [self.compute_distance(current_position, point) for point in self.path_points]
        # Get the index of the nearest point
        index = np.argmin(distances)
        # Return the nearest points
        if index + num_points > len(self.path_points):
            # If not enough points ahead, pad with the last point
            near_points = self.path_points[index:]
            padding = [self.path_points[-1]] * (num_points - len(near_points))
            near_points.extend(padding)
        else:
            near_points = self.path_points[index:index + num_points]
        return near_points

    def compute_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))