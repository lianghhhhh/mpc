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
        self.last_index = None

    def path_points_callback(self, msg):
        self.get_logger().info(f'Received path points: {msg.data}')
        self.path_points = np.array(msg.data).reshape(-1, 2).tolist() # (N, 2) array of (x, z) points
        self.path_points = self.calculate_angle(self.path_points)
        self.last_index = None # reset last index on new path

    def get_near_points(self, current_state, num_points):
        if len(self.path_points) == 0:
            return []

        current_pos = current_state[:2]
        heading_vec = np.array([current_state[3], current_state[2]])  # [cos(theta), sin(theta)]

        if self.last_index is None:
            distances = np.linalg.norm(np.array(self.path_points)[:, :2] - current_pos, axis=1)
            # Get the index of the nearest point
            index = np.argmin(distances)
        else:
            search_window = 20  # number of points to search ahead
            start_index = self.last_index
            end_index = min(start_index + search_window, len(self.path_points))
            points_subset = np.array(self.path_points[start_index:end_index])

            if len(points_subset) == 0:
                index = self.last_index
            else:
                distances = np.linalg.norm(points_subset[:, :2] - current_pos, axis=1)
                index = start_index + np.argmin(distances)

        closest_point = np.array(self.path_points[index])[:2]
        vec_to_point = closest_point - current_pos
        dot_product = np.dot(heading_vec, vec_to_point)
        dist_to_point = np.linalg.norm(vec_to_point)

        if dot_product < 0 and dist_to_point < 0.2:
            # If the closest point is behind the car, choose the next point
            index = min(index + 3, len(self.path_points) - 1)

        self.last_index = index

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
    
    def calculate_angle(self, points):
        for i in range(len(points)-10):
            delta_x = points[i+10][0] - points[i][0]
            delta_z = points[i+10][1] - points[i][1]
            angle = np.arctan2(delta_z, delta_x)  # angle in radians
            points[i].append(np.sin(angle))
            points[i].append(np.cos(angle))
        # For the last 10 points, replicate the angle of the last computed point
        last_angle_sin = points[len(points)-11][2]
        last_angle_cos = points[len(points)-11][3]
        for i in range(len(points)-10, len(points)):
            points[i].append(last_angle_sin)
            points[i].append(last_angle_cos)
        return points