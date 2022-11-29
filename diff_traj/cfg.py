from types import SimpleNamespace
import math

# in meters and seconds
cfg = SimpleNamespace(
    lane_width = 20,
    car_length = 4.48, # 14.7 ft is avg car length
    car_width = 1.77, # 5.8 ft is avg car width
    car_horizon = 40,
    dist_b4_obst = 10,
    min_obst_radius = 1.2,
    max_obst_radius = 3,
    min_theta = 0,
    max_theta = 2 * math.pi,
    n_obstacles = 3,
    n_intervals = 40,
    interval_dur = 0.25,
    max_vel = 15.65, # 35 mph
    max_ang_vel = 1, # 1 G = 1 m/s^2 is max force you want to feel in the car
    max_accel = 1,
    rng_seed = 0,
    traj_length = 160,
    controls_length = 80,
    params_length = 9
)