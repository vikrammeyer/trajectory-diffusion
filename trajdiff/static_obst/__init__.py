from trajdiff.static_obst.dataset import StateDataset
from trajdiff.static_obst.obstacles import generate_obstacles
from trajdiff.static_obst.viz import Visualizations

import math
from types import SimpleNamespace

# in meters and seconds
cfg = SimpleNamespace(
    lane_width=20,
    car_length=4.48,  # 14.7 ft is avg car length
    car_width=1.77,  # 5.8 ft is avg car width
    car_horizon=40,
    dist_b4_obst=10,
    min_obst_radius=2,
    max_obst_radius=3,
    min_theta=-math.pi / 4,
    max_theta=math.pi / 4,
    n_obstacles=3,
    n_intervals=40,
    interval_dur=0.25,
    max_vel=15.65,  # 35 mph
    max_ang_vel=1,  # 1 G = 1 m/s^2 is max force you want to feel in the car
    max_accel=1,
    traj_length=160,
    controls_length=80,
    params_length=9,
)
