from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl

class Visualizations:
    def __init__(self, cfg):
        self.car_length = cfg.car_length
        self.car_width = cfg.car_width
        self.car_horizon = cfg.car_horizon
        self.lane_width = cfg.lane_width
        self.n_obst = cfg.n_obstacles
        self.n = cfg.n_intervals

    def plot_trajectory(self, ax, x0, obstacles, x, iters = None, time = None):
        ax.axis('equal')
        ax.set_xlim(0,120)
        ax.set_ylim(0,self.lane_width + 20)
        self.plot_lanes(ax)
        self.plot_car(ax, x0)
        self.plot_obstacles(ax, obstacles)
        self.plot_x(ax, x)
        if iters and time:
            s = f'Iters: {iters} Time: {time}'
            ax.text(0.5, 0.5, s, transform=ax.transAxes)

    def save_compared_trajectories(self, x0, obstacles, x1, x2, file_name, iters= None, time = None):
        fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2)
        self.plot_trajectory(ax1, x0, obstacles, x1)
        self.plot_trajectory(ax2, x0, obstacles, x2, iters, time)
        fig.savefig(file_name)
        plt.close()

    def save_trajectory(self, x, x0, obstacles, file_name, iters = None, time = None):
        fig, ax = plt.subplots()
        self.plot_trajectory(ax, x0, obstacles, x, iters,time)
        fig.savefig(file_name)
        plt.close(fig)

    def show_trajectory(self, x, x0, obstacles):
        fig, ax = plt.subplots()
        self.plot_trajectory(ax, x0, obstacles, x)
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

    def create_imgs_from_csv_results(self, csv_file_path, imgs_folder_path, num_imgs):
        import csv
        import os, shutil

        if os.path.exists(imgs_folder_path): shutil.rmtree(imgs_folder_path)
        os.mkdir(imgs_folder_path)

        t = 0
        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f)

            for i, row in enumerate(reader):
                if t > num_imgs: break

                if i % 10 == 0:
                    x0 = [float(i) for i in row[:4]]
                    obst_end_idx = 4+3*self.n_obst
                    obstacles = [float(i) for i in row[4:obst_end_idx]]
                    x = [float(i) for i in row[obst_end_idx:obst_end_idx + 4 * self.n]]

                    self.save_trajectory(x,x0,obstacles, f'{imgs_folder_path}/im{i}.png')
                    t += 1

    def plot_lanes(self, ax):
        top = Line2D([0,120],[self.lane_width/2, self.lane_width/2])
        bottom = Line2D([0,120],[-self.lane_width/2,-self.lane_width/2])
        ax.add_line(top)
        ax.add_line(bottom)

    def plot_car(self, ax, x0):
        x = x0[0]
        y = x0[1]
        top_left = (x - self.car_length / 2, y - self.car_width / 2)
        rect = Rectangle(top_left, self.car_length, self.car_width, fill=False) #, angle = x0[3])
        tf = mpl.transforms.Affine2D().rotate_around(x,y,x0[3]) + ax.transData
        rect.set_transform(tf)
        ax.add_patch(rect)

    def plot_obstacles(self, ax, obstacles):
        for i in range(0,len(obstacles),3):
            obst = Circle((obstacles[i],obstacles[i+1]),obstacles[i+2],fill=True)
            buffer = Circle((obstacles[i],obstacles[i+1]),obstacles[i+2] + self.car_width,fill=False)
            ax.add_patch(obst)
            ax.add_patch(buffer)

    def plot_x(self, ax, x):
        xs = []
        ys = []

        for i in range(0,len(x),4):
            xs.append(x[i])
            ys.append(x[i+1])

        ax.scatter(xs,ys,facecolors='none',edgecolors='purple')