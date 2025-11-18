import joblib
import numpy as np
import casadi as ca
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from car_control_pkg.utils import loadModelFunc, createMpcSolver, normalize, denormalize, createAcadosSolver

class CarControlNode(Node):
    def __init__(self, car_state_node, path_points_node, model_path, N_steps, dt):
        super().__init__('car_control_node')
        self.get_logger().info('Car Control Node has been started.')
        # self.front_wheel_pub = self.create_publisher(Float32MultiArray, "car_C_front_wheel", 10)
        # self.rear_wheel_pub = self.create_publisher(Float32MultiArray, "car_C_rear_wheel", 10)
        self.wheel_vel_pub = self.create_publisher(Float32MultiArray, "wheel_velocity", 10)
        self.pred_path_pub = self.create_publisher(Float32MultiArray, "predicted_path", 10)

        self.car_state_node = car_state_node
        self.path_points_node = path_points_node

        self.model_path = model_path
        self.x_scaler = joblib.load(f'{self.model_path}/x_scaler.save')
        self.u_scaler = joblib.load(f'{self.model_path}/u_scaler.save')

        self.N_steps = N_steps
        self.dt = dt

        # MPC/cache handles
        self._model_func, self._lib_dir, self._lib_name = loadModelFunc(self.model_path, self.dt)
        # self._solver, self._u_pred, self._next_x_pred, self._current_x, self._target_path = createMpcSolver(self._model_func, N=10)
        self._acados_solver = createAcadosSolver(self._model_func, self._lib_dir, self._lib_name, self.N_steps, self.dt)
        
        # Run MPC periodically while the node is spinning (10 Hz)
        self.create_timer(0.1, lambda: self.find_control_command())

    def publish_control_command(self, control_input):
        self.get_logger().info(f'Publishing control command: {control_input}')
        # front_msg = Float32MultiArray()
        # rear_msg = Float32MultiArray()
        # vals = [float(x) for x in control_input]
        msg = Float32MultiArray()
        vals = [float(x) for x in control_input]
        msg.data = vals
        # front_msg.data = vals[0:2]
        # rear_msg.data = vals[2:4]
        # self.front_wheel_pub.publish(front_msg)
        # self.rear_wheel_pub.publish(rear_msg)
        self.wheel_vel_pub.publish(msg)

    def publish_predicted_path(self, predicted_path):
        msg = Float32MultiArray()
        vals = predicted_path.flatten().tolist()
        msg.data = vals
        self.get_logger().info(f'Publishing predicted path: {predicted_path}')
        self.pred_path_pub.publish(msg)

    def find_control_command(self):
        start_time = self.get_clock().now()
        current_state = self.car_state_node.car_state
        path_points = self.path_points_node.get_near_points(current_state, num_points=(self.N_steps+1))
        if len(current_state) != 4:
            self.get_logger().warn(f'Invalid car state received: {current_state}')
            return
        if len(path_points) != (self.N_steps+1):
            self.get_logger().warn(f'Invalid path points received: {path_points}')
            return
        
        self.get_logger().info(f'Current state: {current_state}')
        self.get_logger().info(f'Target path: {path_points}')
        
        current_state = normalize(current_state, "state", self.x_scaler)
        path_points = normalize(path_points, "path", self.x_scaler)
        # state_data = ca.DM(current_state)
        # target_path_data = ca.DM(path_points)

        # self._solver.set_value(self._current_x, state_data)
        # self._solver.set_value(self._target_path, target_path_data)
        self._acados_solver.set(0, "lbx", current_state)
        self._acados_solver.set(0, "ubx", current_state)
        for t in range(self.N_steps+1):
            self._acados_solver.set(t, "p", path_points[t, :])

        try:
            # solution = self._solver.solve()
            # optimal_control = solution.value(self._u_pred)
            # control_input = optimal_control[0, :]
            # control_input = denormalize(control_input, "u", self.u_scaler)
            # self.publish_control_command(control_input)
            status = self._acados_solver.solve()
            if status != 0:
                self.get_logger().error(f'Acados solver failed with status {status}')
                return

            control_input = self._acados_solver.get(0, "u") # get optimal control at time 0
            control_input = denormalize(control_input, "u", self.u_scaler)
            predict_path = []
            for t in range(self.N_steps+1):
                pred_postion = self._acados_solver.get(t, "x")[:2]
                pred_postion = denormalize(pred_postion.reshape(1, -1), "x", self.x_scaler)
                predict_path.append(pred_postion)

            end_time = self.get_clock().now()
            elapsed_time = (end_time - start_time).nanoseconds / 1e9 # seconds
            self.get_logger().info(f'MPC computation time: {elapsed_time:.6f} s')
            
            self.publish_control_command(control_input)
            self.publish_predicted_path(np.array(predict_path).reshape(self.N_steps+1, 2))

        except Exception as e:
            self.get_logger().error(f'MPC solver failed: {e}')