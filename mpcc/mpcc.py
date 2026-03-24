
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry 
import numpy as np
from argparse import Namespace
import casadi as ca
import time
import math

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

        self.weight_contour = self.declare_parameter('weight_contour', 0.0).value
        self.weight_lag = self.declare_parameter('weight_lag', 0.0).value
        self.weight_theta = self.declare_parameter('weight_theta', 0.0).value
        self.weight_steering = self.declare_parameter('weight_steering', 0.0).value
        self.weight_acceleration = self.declare_parameter('weight_acceleration', 0.0).value
        self.weight_steering_acceleration = self.declare_parameter('weight_steering_acceleration', 0.0).value

        # MPCC Constraints
        self.min_pos = self.declare_parameter('min_pos', -100).value
        self.max_pos = self.declare_parameter('max_pos', 100).value
        self.min_heading = self.declare_parameter('min_heading', -20.0).value
        self.max_heading = self.declare_parameter('max_heading', 20.0).value
        self.min_steering = self.declare_parameter('min_steering', -0.526).value
        self.max_steering = self.declare_parameter('max_steering', 0.526).value
        self.min_speed = self.declare_parameter('min_speed', 2.0).value
        self.max_speed = self.declare_parameter('max_speed', 8.0).value
        self.max_track_len = self.declare_parameter('max_track_len', 300.0).value
        self.min_progress = self.declare_parameter('min_progress', 1.0).value
        self.max_progress = self.declare_parameter('max_progress', 12.0).value   

        #TODO: Update these params  
        self.max_force = self.declare_parameter('max_force', 20.0).value
        self.max_deceleration = self.declare_parameter('max_deceleration', 20.0).value 

        # Dynamics settings
        self.wheelbase = self.declare_parameter('wheelbase', 0.324).value
        self.vehicle_mass = self.declare_parameter('vehicle_mass', 3.47).value

        self.track = Centerline(self.map_name)

        # self.contour_error = 0.0
        # self.lateral_error = 0.0
        self.step_counter = 0
        self.state_len = 4
        self.input_len = 3
        self.U0 = np.zeros((self.horizon, self.input_len))
        self.X0 = np.zeros((self.horizon + 1, self.state_len)) # +1 to include current state

        self.optimization_params = np.zeros(self.state_len + 2 * self.horizon + 1)

        self.pose_sub = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.pose_callback,
            10
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )
        
        self.get_logger().info('MPPC node activated.')

        self.init_mpcc()
        self.init_obj()
        self.init_variable_constraints()
        self.init_function_constraints()
        self.init_bounds()
        self.init_solver()


    def init_mpcc(self):

        # state variables
        states = ca.MX.sym('states', 4) # [x, y, psi, s]
        controls = ca.MX.sym('controls', 3) # [delta, v, p]

        self.n_states = states.numel()
        self.n_controls = controls.numel()

        # Kinematic Vehicle Model
        kinematics = ca.vertcat(
            controls[1] * ca.cos(states[2]),
            controls[1] * ca.sin(states[2]),
            (controls[1] / self.wheelbase) * ca.tan(controls[0]),
            controls[2]
        )

        # Create kinematic function and vector for state/input horizon
        self.f = ca.Function('f', [states, controls], [kinematics])
        self.X = ca.MX.sym('X', self.n_states, (self.horizon+1))
        self.U = ca.MX.sym('U', self.n_controls, self.horizon)
        self.P = ca.MX.sym('P', self.n_states + 2 * self.horizon + 1)

    def init_obj(self):
        self.obj = 0

        # Q matrix and theta
        for i in range(self.horizon):
            nextState = self.X[:, i+1]
            tangent_ang = self.track.lut_phi(nextState[3])
            delta_x = nextState[0] - self.track.lut_x(nextState[3])
            delta_y = nextState[1] - self.track.lut_y(nextState[3])

            # Create error expressions.
            contour_err = (ca.sin(tangent_ang) * delta_x) - (ca.cos(tangent_ang) * delta_y)
            lag_err = -(ca.cos(tangent_ang) * delta_x) - (ca.sin(tangent_ang) * delta_y)

            # Error matrix.
            self.obj = self.obj + (self.weight_contour * (contour_err ** 2)) + (self.weight_lag * (lag_err ** 2))

            # Progress smoothing.
            self.obj = self.obj - self.U[2, i] * self.weight_theta

            # Smoothing matrix.
            self.obj = self.obj + ((self.U[0, i]**2) * self.weight_steering)
            if i > 0:
                self.obj = self.obj + (self.U[1, i] - self.U[1, i-1])**2 * self.weight_acceleration
                self.obj = self.obj + (self.U[0, i] - self.U[0, i-1])**2 * self.weight_steering_acceleration

    def init_variable_constraints(self):
        lbx = [self.min_pos, self.min_pos, self.min_heading, 0] * (self.horizon + 1) + [self.min_steering, self.min_speed, self.min_progress] * (self.horizon)
        self.lbx = np.array(lbx)
        ubx = [self.max_pos, self.max_pos, self.max_heading, self.max_track_len] * (self.horizon + 1) + [self.max_steering, self.max_speed, self.max_progress] * (self.horizon)
        self.ubx = np.array(ubx)

    def init_function_constraints(self):
        # initial state we get from the pose callback
        self.g = []
        # Enforce initial condition constraint
        self.g = ca.vertcat(self.g, self.X[:, 0] - self.P[:self.state_len])

        for i in range(self.horizon):
            # Enforce kinematic constraint
            nextState = self.X[:, i] + (self.timestep * self.f(self.X[:, i], self.U[:, i]))
            self.g = ca.vertcat(self.g, self.X[:, i+1] - nextState)

            # Represent current position as dot product (delta point vector * current pos)
            self.g = ca.vertcat(self.g, self.P[self.n_states + 2 * i] * self.X[0, i + 1] - self.P[self.n_states + 2 * i + 1] * self.X[1, i + 1])

            # Define the force
            lateral_force = self.U[1, i]**2 / self.wheelbase * ca.tan(ca.fabs(self.U[0, i])) * self.vehicle_mass
            self.g = ca.vertcat(self.g, lateral_force)

            if i == 0: 
                self.g = ca.vertcat(self.g, ca.fabs(self.U[1, i] - self.P[-1])) # ensure initial speed matches current speed
            else:
                self.g = ca.vertcat(self.g, ca.fabs(self.U[1, i] - self.U[1, i - 1]))  # limit decceleration

    def init_bounds(self):
        self.lbg = np.zeros((self.g.shape[0], 1))
        self.ubg = np.zeros((self.g.shape[0], 1))
        for i in range(self.horizon):
            self.lbg[self.n_states - 2 + (self.n_states + 3) * (i + 1), 0] = -self.max_force
            self.ubg[self.n_states - 2 + (self.n_states + 3) * (i + 1), 0] = self.max_force
            self.lbg[self.n_states - 1 + (self.n_states + 3) * (i + 1), 0] = - self.max_deceleration
            self.ubg[self.n_states - 1 + (self.n_states + 3) * (i + 1), 0] = np.inf

    def set_path_constraints(self):
        for i in range(self.horizon):
            right_point = self.track.get_left_point(self.X0[i, 3])
            left_point = self.track.get_right_point(self.X0[i, 3])
            delta_point = right_point - left_point

            delta_point[0] = -delta_point[0]

            # Keep track of delta vector for constraint function
            self.optimization_params[self.n_states + 2 * i:self.n_states + 2 * i + 2] = delta_point

            right_bound = delta_point[0] * right_point[0] - delta_point[1] * right_point[1]
            left_bound = delta_point[0] * left_point[0] - delta_point[1] * left_point[1]

            self.lbg[self.n_states - 3 + (self.n_states + 3) * (i + 1), 0] = min(left_bound, right_bound)
            self.ubg[self.n_states - 3 + (self.n_states + 3) * (i + 1), 0] = max(left_bound, right_bound)

    def init_solver(self):
        variables = ca.vertcat(ca.reshape(self.X, self.n_states * (self.horizon + 1), 1), ca.reshape(self.U, self.n_controls * self.horizon, 1))
        nlp = {'x': variables, 'f': self.obj, 'g': self.g, 'p': self.P}
        opts = {"ipopt": {"max_iter": 1000, "print_level": 5, "output_file": "debug_log.txt"}, "print_time": 1}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)


    def solve(self):
        x_init = ca.vertcat(ca.reshape(self.X0.T, self.n_states * (self.horizon + 1), 1), ca.reshape(self.U0.T, self.n_controls * self.horizon, 1))

        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=self.optimization_params)

        self.X0 = ca.reshape(sol['x'][0:self.n_states * (self.horizon + 1)], self.n_states, self.horizon + 1).T
        u = ca.reshape(sol['x'][self.n_states * (self.horizon + 1):], self.n_controls, self.horizon).T

        trajectory = self.X0.full()
        inputs = u.full()
        solved = True
        if self.solver.stats()['return_status'] == 'Infeasible_Problem_Detected':
            solved = False
        
        return trajectory, inputs, solved
    
    def plan(self, obs, speed):
        self.step_counter += 1
        controls = [0.0, 0.0]

        self.optimization_params[:self.n_states] = obs
        self.optimization_params[-1] = max(speed, 1.0)

        self.set_path_constraints()

        trajectory, inputs, solved = self.solve()

        if not solved:
            self.get_logger().warn(f"Step {self.step_counter}: SOLVER FAILED")
            exit()
        
        controls = [float(inputs[0, 0]), float(inputs[0, 1])]

        self.X0 = np.vstack((trajectory[1:, :], trajectory[-1, :]))
        self.U0 = np.vstack((inputs[1:, :], inputs[-1, :]))

        return controls

    # This will act as my prepare input
    def pose_callback(self, pose_msg):
        start = time.time()

        x_state = pose_msg.pose.pose.position.x
        y_state = pose_msg.pose.pose.position.y
        orientation = pose_msg.pose.pose.orientation
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w

        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        speed = pose_msg.twist.twist.linear.x

        obs = [x_state, y_state, yaw]

        x0 = np.append(obs, self.track.calculate_progress(obs[0], obs[1]))

        controls = self.plan(x0, speed)

        self.publish_drive(controls)

    def publish_drive(self, controls):
        msg = AckermannDriveStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "ego_racecar/base_link"

        msg.drive.steering_angle = float(controls[0])
        msg.drive.speed = float(controls[1])

        self.drive_pub.publish(msg)


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