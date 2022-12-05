"""
- collsion free trajectories (this is what was used in the Neurips workshop paper)
- dynamics violations ?? (this could be interesting to see the "quality" of the generated trajectories)
"""
from collections import namedtuple
from diff_traj.cfg import cfg
from diff_traj.utils.geometry import *

def n_collision_states(state_traj, obstacles):
    obsts = []
    for i in range(0, len(obstacles), 3):
        x, y, r = obstacles[i:i+3]
        obsts.append(Circle(x, y, r))

    collisions = 0
    for i in range(0, len(state_traj), 4):
        x, y, theta = state_traj[i:i+3]



        car = Rect(x, y, cfg.car_length, cfg.car_width)
        for obst in obsts:
            if collision(obst, car): collisions += 1

    return collisions

