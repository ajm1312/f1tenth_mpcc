# import trajectory_planning_helpers as tph
import sys
import numpy as np
from pathlib import Path
import trajectory_planning_helpers as tph
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory

def init_track_interpolants(center_line, exclusion_width):
    widths = np.row_stack((center_line.widths, center_line.widths[1:int(center_line.widths.shape[0] / 2), :]))
    path = np.row_stack((center_line.path, center_line.path[1:int(center_line.path.shape[0] / 2), :]))
    extended_track = TrackLine(path)
    extended_track.init_path()
    extended_track.init_track()

    center_interpolant = LineInterpolant(extended_track.path, extended_track.s_path, extended_track.psi + np.pi / 2) # add pi/2 to account for coord frame change

    left_path = extended_track.path - extended_track.nvecs * np.clip((widths[:, 0][:, None]  - exclusion_width), 0, np.inf)
    left_interpolant = LineInterpolant(left_path, extended_track.s_path)
    right_path = extended_track.path + extended_track.nvecs * np.clip((widths[:, 1][:, None] - exclusion_width), 0, np.inf)
    right_interpolant = LineInterpolant(right_path, extended_track.s_path)

    return center_interpolant, left_interpolant, right_interpolant


class LineInterpolant:
    def __init__(self, path, s_path, angles=None):
        self.lut_x = ca.interpolant('lut_x', 'bspline', [s_path], path[:, 0])
        self.lut_y = ca.interpolant('lut_y', 'bspline', [s_path], path[:, 1])
        if angles is not None:
            self.lut_angle = ca.interpolant('lut_angle', 'bspline', [s_path], angles)

    def get_point(self, s):
        return np.array([self.lut_x(s).full()[0, 0], self.lut_y(s).full()[0, 0]])


# class Track:
class Track():
    def __init__(self, path):
        self.path = path

    def init_path(self):
        self.diffs = self.track[:, 0] - self.track[:, 1]

        #make spline curve fit
        self.tck, self.u = splprep(x = [self.track[:, 0], self.track[:, 1]], k = 3, s=0, per=True)

        # progress variable

    # def get_point(self, x, y):
        
        # 




class Centerline(Track):

    # Load the centerline
    def __init__(self, map_name):
        # dir = Path(__file__).parent
        # traj_path = dir.parent / "racelines"
        share_dir = get_package_share_directory('mpcc') 
        traj_path = Path(share_dir) / "racelines"

        self.map_name = map_name
        self.load_track(self.map_name, traj_path)
        self.init_path()


    def load_track(self, map_name, directory):
        print("\n\n\n\n\n")
        print(map_name)
        filename = str(directory) + "/" + map_name + "_centerline.csv"

        self.track = np.loadtxt(filename, delimiter=",", skiprows=1)
        self.path_x = self.track[:, 0]
        self.path_y = self.track[:, 1]
        self.path = self.track[:, 0:2]
        self.widths = self.track[:, 2:]
        print(self.path_x)
        self.width = self.track[:, 2:]



if __name__ == '__main__':
    test = Centerline("Oschersleben")
    x_track = test.path_x[:]
    y_track = test.path_y[:]

    u_new = np.linspace(0, 1, 2000)
    x_spline, y_spline = splev(u_new, test.tck)

    # Debug plotting
    plt.figure(figsize=(12, 8))
    
    # Plot original points (red dots)
    plt.plot(x_track, y_track, 'ro', markersize=3, label='Centerline')
    
    # Plot the spline fit (smooth blue line)
    plt.plot(x_spline, y_spline, 'b-', linewidth=2, label='Spline Fit')

    # Formatting the plot
    plt.title(f'Track Centerline and Spline Fit: {test.map_name}')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.axis('equal') # Prevents distortion of the track shape
    plt.grid(True)
    
    # Render the visualization
    plt.show()