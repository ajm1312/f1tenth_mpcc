# import trajectory_planning_helpers as tph
import numpy as np
from pathlib import Path
import trajectory_planning_helpers as tph
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory
import casadi as ca

class Centerline():

    # Load the centerline
    def __init__(self, map_name):
        # dir = Path(__file__).parent
        # traj_path = dir.parent / "racelines"
        share_dir = get_package_share_directory('mpcc') 
        traj_path = Path(share_dir) / "racelines"
        filename = str(traj_path) + "/" + map_name + "_centerline.csv"

        self.track = np.loadtxt(filename, delimiter=",", skiprows=1)
        self.path_x = self.track[:, 0]
        self.path_y = self.track[:, 1]
        self.width_r = self.track[:, 2]
        self.width_l = self.track[:, 3]

        # Interpolant definitions
        self.center_interpolant = None
        self.right_interpolant = None
        self.left_interpolant = None

        self.lut_phi = None
        self.lut_x = None
        self.lut_y = None

        self.init_interpolant()

    def init_interpolant(self):
        X_coords = self.path_x
        Y_coords = self.path_y

        dx = np.gradient(X_coords)
        dy = np.gradient(Y_coords)

        heading_angle = np.unwrap(np.arctan2(dy, dx))

        ds = np.sqrt(dx**2 + dy**2)

        s_grid = np.cumsum(ds)
        s_grid = s_grid - s_grid[0]
        self.s_grid = s_grid

        self.s_max = s_grid[-1]

        # print(s_grid)

        # heading_angle = np.append(heading_angle, heading_angle[-1])

        self.lut_x = ca.interpolant('lut_x', 'bspline', [s_grid], X_coords.flatten())
        self.lut_y = ca.interpolant('lut_y', 'bspline', [s_grid], Y_coords.flatten())
        self.lut_phi = ca.interpolant('lut_angle', 'bspline', [s_grid], heading_angle.flatten())

        # Calculate border interpolants
        x_left = X_coords - self.width_l * np.sin(heading_angle)
        y_left = Y_coords + self.width_l * np.cos(heading_angle)

        x_right = X_coords + self.width_r * np.sin(heading_angle)
        y_right = Y_coords - self.width_r * np.cos(heading_angle)

        self.lut_x_left = ca.interpolant('lut_x_l', 'bspline', [s_grid], x_left.flatten())
        self.lut_y_left = ca.interpolant('lut_y_l', 'bspline', [s_grid], y_left.flatten())

        self.lut_x_right = ca.interpolant('lut_x_r', 'bspline', [s_grid], x_right.flatten())
        self.lut_y_right = ca.interpolant('lut_y_r', 'bspline', [s_grid], y_right.flatten())

    def get_left_point(self, s):
        return np.array([self.lut_x_left(s).full()[0, 0], 
                        self.lut_y_left(s).full()[0, 0]])

    def get_right_point(self, s):
        return np.array([self.lut_x_right(s).full()[0, 0], 
                        self.lut_y_right(s).full()[0, 0]])

    def calculate_progress(self, x, y):

        distances_sq = (self.path_x - x)**2 + (self.path_y - y)**2

        point = np.argmin(distances_sq)
        # print("s_grid:" + str(self.s_grid))

        return self.s_grid[point]