import rclpy
from rclpy.executors import MultiThreadedExecutor
from car_control_pkg.car_state_node import CarStateNode
from car_control_pkg.car_control_node import CarControlNode
from car_control_pkg.path_points_node import PathPointsNode

def main():
    rclpy.init()

    car_state_node = CarStateNode()
    path_points_node = PathPointsNode()
    car_control_node = CarControlNode(car_state_node, path_points_node)

    executor = MultiThreadedExecutor()
    executor.add_node(car_state_node)
    executor.add_node(path_points_node)
    executor.add_node(car_control_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
