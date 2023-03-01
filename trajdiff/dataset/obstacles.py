import random
from math import sqrt

import numpy as np


def get_dist(c1, c2):
    """Calc distance between two circlular obstacles c1 = [x1, y1, r1] c2 = [x2, y2, r2]
    sqrt((x2 - x1)**2 + (y2 - y1)**2) - (r2 + r1)
    """
    return sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) - (c1[2] + c2[2])


def rand_in_range(a: float, b: float) -> float:
    return (b - a) * random.random() + a


def generate_obstacles(cfg):
    obstacles = np.zeros(3 * cfg.n_obstacles, dtype=np.float32)
    obst_idx = 0

    DIST_BTW_OBSTS = cfg.car_width + 2.0  # 1 meter clearance on either side of the car

    while obst_idx < 3 * cfg.n_obstacles:
        x = rand_in_range(cfg.dist_b4_obst, cfg.car_horizon)
        y = rand_in_range(-cfg.lane_width / 2, cfg.lane_width / 2)
        r = rand_in_range(cfg.min_obst_radius, cfg.max_obst_radius)
        new_obstacle = np.array([x, y, r], dtype=np.float32)
        infeasible = False

        # check for path btw it and other obstacles
        j = 0
        while not infeasible and j < obst_idx:
            infeasible = get_dist(obstacles[j : j + 3], new_obstacle) < DIST_BTW_OBSTS
            j = j + 3

        if not infeasible:
            obstacles[obst_idx : obst_idx + 3] = new_obstacle
            obst_idx += 3

    return obstacles
