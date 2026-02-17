
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
from argparse import Namespace


class MPCCNode(Node):
    def __init__(self):
        super().__init__('mpcc_node')

        # Import params for params file
        self.horizon = self.declare_parameter('horizon', 8)
        self.v_min = self.declare_parameter("v_min", 2.0)
        self.v_max = self.declare_parameter("v_max", 4.5)

        # self.scan_sub = self.create_subscription(
        #     LaserScan,
        #     '/scan',
        #     self.scan_callback,
        #     10
        # )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            self.scan_callback,
            10
        )
        
        self.get_logger().info('MPPC node activated.')


def main(args=None):

    rclpy.init(args=args)
    node = MPCCNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()