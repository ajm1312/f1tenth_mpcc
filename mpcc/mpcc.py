
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry 
import numpy as np
from argparse import Namespace
import casadi as ca

from .utils import *


class MPCCNode(Node):
    def __init__(self):
        super().__init__('mpcc_node')

        # Import params for params file
        # MPCC settings
        self.horizon = self.declare_parameter('horizon', 8).value
        self.v_min = self.declare_parameter("v_min", 2.0).value
        self.v_max = self.declare_parameter("v_max", 4.5).value
        self.map_name = self.declare_parameter("map_name", "Oschersleben").value
        self.timestep = self.declare_parameter("timestep", 0.1).value

        self.q_c = self.declare_parameter('q_c', 0.0).value
        self.q_l = self.declare_parameter('q_l', 0.0).value
        self.q_theta = self.declare_parameter('q_theta', 0.0).value
        self.q_delta = self.declare_parameter('q_delta', 0.0).value
        self.q_v = self.declare_parameter('q_v', 0.0).value

        # Dynamics settings
        self.wheelbase = self.declare_parameter('wheelbase', 0.324).value

        self.track = Centerline(self.map_name)

        self.contour_error = 0.0
        self.lateral_error = 0.0
        self.state_len = 4
        self.input_len = 2
        self.u0 = np.zeros((self.horizon, self.input_len))
        self.x0 = np.zeros((self.horizon + 1, self.state_len)) # +1 to include current state

        # self.scan_sub = self.create_subscription(
        #     LaserScan,
        #     '/scan',
        #     self.scan_callback,
        #     10
        # )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/ego_racecar',
            self.pose_callback,
            10
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )
        
        self.get_logger().info('MPPC node activated.')


    # def scan_callback():
    #     return 0

    def init_mpcc(self):

        self.center_line = CenterLine(self.map_name)

        # state variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        s = ca.SX.sym('s')

        states = ca.vertcat(x, y, yaw, s)
        n_states = states.numel()

        # input variables
        delta = ca.MX.sym('delta')
        p = ca.MX.sym('p') # p represents speed

        inputs = ca.vertcat(delta, p)
        n_inputs = inputs.numel()

        # Kinematic Vehicle Model
        kinematics = ca.vertcat(
            p * ca.cos(yaw),
            p * ca.sin(yaw),
            (p / self.wheelbase) * ca.tan(delta),
            p
        )

        # Create kinematic function and vector for state/input horizon
        self.f = ca.Function('f', [states, controls], [kinematics])
        self.X = ca.sx.sym('X', n_states, (self.N+1))
        self.U = ca.sx.sym('U', n_inputs, self.N)

        self.init_obj()

        # Parameter optimization values
        self.P = ca.SX.sym('P', n_states + 2 * self.N + 3) # init state for solver

        optimization_variables = ca.vertcat(ca.reshape(self.X, n_states * (self.N + 1), 1),
                                ca.reshape(self.U, n_controls * self.N, 1))


    def init_obj(self):
        self.obj = 0

        # Q matrix and theta
        for i in range(self.horizon):
            st = self.X[:, i+1]
            s_i = st[3]

            t_ang = self.center_interpolant.lut_angle(s_i)
            xr = self.center_interpolant.lut_x(s_i)
            yr = self.center_interpolant.lut_y(s_i)

            e_c = -ca.sin(t_ang)*(st[0]-xr) - ca.cos(t_ang)*(st[1]-yr)
            e_l = ca.cos(t_ang)*(st[0]-xr) - ca.sin(t_ang)*(st[1]-yr)

            self.obj += self.q_c * e_c**2 + self.q_l * e_l**2 - self.qv * self.U[1, i]

        # delta matrix
        for k in range(1, self.horizon):
            d_delta = self.U[0, k] - self.U[0, k-1]
            d_v = self.U[1, k] - self.U[1, k-1]
            delta_penalty = self.q_delta * d_delta**2
            vel_penalty = self.q_v * d_v**2 
            self.obj += delta_penalty + vel_penalty


    def init_constraints(self):
        # initial state we get from the pose callback

        self.g = self.X[:, 0] - self.P[:self.n_states]

        for i in range(self.horizon):
            x_i = self.X[:, i]
            u_i = self.U[:, i]
            x_ip1 = self.X[:,i+1]

            # Evaluate the next timestep relative to the kinematics equations defined earlier.
            f_x_i = self.f(x_i, u_i)
            x_euler = x_i + self.timestep * f_x_i
            # Compare observed vs calculated.
            self.g = ca.vertcat(self.g, x_ip1 - x_euler)

        # Force solver to calculate a velocity and steering angle that matches to predicted.
        total = self.g.shape[0]
        self.lbg = np.zeros((total,1))
        self.ubg = np.zeros((total,1))

        print(f"[init_bounds] constraints: {total}")

    def set_path_constraints(self):
        for i in range(self.horizon):
            # Find left and right points relative to current pos.
            s_i = float(self.X0[i, 3])
            right_pt = self.right_interpolate.get_point(s_i)
            left_pt = self.left_interpolate.get_point(s_i)

            # Normalize the points.
            norm = right_pt - left_pt
            norm[0] = -norm[0]
            a_i, b_i = norm

            self.optimization_params[self.n_states + (2*i)] = a_i
            self.optimization_params[self.n_states + (2*i)+1] = b_i


    def pose_callback(self, pose_msg):
        start = time.time()

        x_state = pose_msg.pose.pose.position.x
        y_state = pose_msg.pose.pose.position.y
        orientation = pose_msg.pose.pose.orientation
        quan = [orientation.x, orientation.y, orientation.z, orientation.w]
        yaw_state = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))

    def plan(self):
        return 0

    def solver():
        nlp = {'x': self.X, 'f': self.obj}
        return 0

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