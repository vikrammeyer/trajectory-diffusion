"""
- collsion free trajectories (this is what was used in the Neurips workshop paper)
- dynamics violations ?? (this could be interesting to see the "quality" of the generated trajectories)
"""
from collections import namedtuple
from diff_traj.cfg import cfg
import math

Circle = namedtuple("Circle", ["x", "y", "r"])
Rect = namedtuple("Rect", ["a", "b", "c", "d"])

def point_in_rect(pt, R: Rect):
    pass
#     left =
#     right =
#     return left and right


def intersect_circle(C, linesegment):
    #ax + by = c --> y = -a/b x + c/b
    dist = ((abs()))

def checkCollision(a, b, c, x, y, radius):

    # Finding the distance of line
    # from center.
    dist = ((abs(a * x + b * y + c)) /
            math.sqrt(a * a + b * b))

    # Checking if the distance is less
    # than, greater than or equal to radius.
    if (radius == dist):
        print("Touch")
    elif (radius > dist):
        print("Intersect")
    else:
        print("Outside")

def collision(C: Circle, R: Rect):
    return (point_in_rect((C.x, C.y), R) or
            intersect_circle(C, (R.a, R.b)) or
            intersect_circle(C, (R.b, R.c)) or
            intersect_circle(C, (R.c, R.d)) or
            intersect_circle(C, (R.d, R.a)) or)


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