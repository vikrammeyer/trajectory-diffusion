from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from math import degrees

class Visualizations:
    def __init__(self, cfg):
        self.car_length = cfg.car_length
        self.car_width = cfg.car_width
        self.x_lims = (0, cfg.car_horizon + 30)
        self.y_lims = (-cfg.lane_width / 2, cfg.lane_width / 2)
        self.car_horizon = cfg.car_horizon
        self.lane_width = cfg.lane_width
        self.n_obst = cfg.n_obstacles
        self.n = cfg.n_intervals

    def plot_trajectory(self, ax, obstacles, x):
        ax.axis('equal')
        ax.set_xlim(*self.x_lims)
        ax.set_ylim(*self.y_lims)
        self.plot_lanes(ax)
        self.plot_car(ax, [0, 0, 0, 0])
        self.plot_obstacles(ax, obstacles)
        self.plot_x(ax, x)

    def save_compared_trajectories(self, obstacles, x1, x2, file_name):
        fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2)
        self.plot_trajectory(ax1, obstacles, x1)
        self.plot_trajectory(ax2, obstacles, x2)
        fig.savefig(file_name)
        plt.close()

    def save_trajectory(self, x, obstacles, file_name):
        fig, ax = plt.subplots()
        self.plot_trajectory(ax, obstacles, x)
        fig.savefig(file_name)
        plt.close(fig)

    def show_trajectory(self, x, obstacles):
        fig, ax = plt.subplots()
        self.plot_trajectory(ax, obstacles, x)
        plt.show()
        plt.close(fig)

    def create_gif_from_pngs(self, folder_path, output_path):
        from glob import glob
        from PIL import Image

        imgs = glob(folder_path + '/*.png')
        imgs.sort()
        frames = [Image.open(i) for i in imgs]

        im = Image.new(frames[0].mode, frames[0].size)
        im.save(output_path, format='GIF', append_images=frames[0:], save_all=True, duration=50, loop=0)

    def plot_lanes(self, ax):
        top = Line2D(self.x_lims, [self.y_lims[1], self.y_lims[1]])
        bottom = Line2D(self.x_lims, [self.y_lims[0], self.y_lims[0]])
        ax.add_line(top)
        ax.add_line(bottom)

    def plot_car(self, ax, pose):
        x, y, _, theta = pose
        top_left = (x - self.car_length / 2, y - self.car_width / 2)
        rect = Rectangle(top_left, self.car_length, self.car_width, angle=degrees(theta), fill=False)
        ax.add_patch(rect)

    def plot_obstacles(self, ax, obstacles):
        for i in range(0,len(obstacles),3):
            obst = Circle((obstacles[i],obstacles[i+1]),obstacles[i+2],fill=True)
            buffer = Circle((obstacles[i],obstacles[i+1]),obstacles[i+2] + self.car_width,fill=False)
            ax.add_patch(obst)
            ax.add_patch(buffer)

    def plot_x(self, ax, x):
        for i in range(0,len(x),4):
            self.plot_car(ax, x[i:i+4])