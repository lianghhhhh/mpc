import numpy as np
import casadi as ca
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from car_control_pkg.utils import loadModelFunc, createMpcSolver, normalize, denormalize

class CarControlNode(Node):
    def __init__(self, car_state_node, path_points_node):
        super().__init__('car_control_node')
        self.get_logger().info('Car Control Node has been started.')
        self.front_wheel_pub = self.create_publisher(Float32MultiArray, "car_C_front_wheel", 10)
        self.rear_wheel_pub = self.create_publisher(Float32MultiArray, "car_C_rear_wheel", 10)
        self.car_state_node = car_state_node
        self.path_points_node = path_points_node
        # MPC/cache handles
        self._model_func = loadModelFunc()
        self._solver, self._u_pred, self._next_x_pred, self._current_x, self._target_path = createMpcSolver(self._model_func, N=10)

        # Run MPC periodically while the node is spinning (10 Hz)
        self.create_timer(0.01, lambda: self.find_control_command(10))

    def publish_control_command(self, control_input):
        self.get_logger().info(f'Publishing control command: {control_input}')
        front_msg = Float32MultiArray()
        rear_msg = Float32MultiArray()
        vals = [float(x) for x in control_input]
        front_msg.data = vals[0:2]
        rear_msg.data = vals[2:4]
        self.front_wheel_pub.publish(front_msg)
        self.rear_wheel_pub.publish(rear_msg)

    def find_control_command(self, N=10):
        current_state = self.car_state_node.car_state
        path_points = self.path_points_node.get_near_points(current_state[0:2], num_points=(N+1)*2)
        if len(current_state) != 4:
            self.get_logger().warn(f'Invalid car state received: {current_state}')
            return
        if len(path_points) != (N+1)*2:
            self.get_logger().warn(f'Invalid path points received: {path_points}')
            return
        
        current_state = normalize(current_state, "x")
        path_points = normalize(np.array(path_points).reshape(2, -1), "x") # reshape to (N+1, 2)
        target_path_data = ca.DM(path_points).reshape((N+1, 2))

        self._solver.set_value(self._current_x, current_state)
        self._solver.set_value(self._target_path, target_path_data)

        try:
            solution = self._solver.solve()
            optimal_control = solution.value(self._u_pred)
            control_input = optimal_control[0, :]
            control_input = denormalize(control_input, "u")
            self.publish_control_command(control_input)
        except Exception as e:
            self.get_logger().error(f'MPC solver failed: {e}')