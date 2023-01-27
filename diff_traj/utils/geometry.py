from collections import namedtuple
from dataclasses import dataclass
from math import cos, sin, sqrt

Point = namedtuple("Point", ["x", "y"])
Circle = namedtuple("Circle", ["x", "y", "r"])
Rect = namedtuple("Rect", ["a", "b", "c", "d"])

@dataclass
class Point:
    x: float
    y: float

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

def rotate(pt, theta):
    # https://math.stackexchange.com/questions/2581058/rotating-rectangle-by-its-center
    # equivalent of constructing a rotation matrix and multiplying the point/vec2D by it
    return Point(pt.x * cos(theta) - pt.y * sin(theta), pt.x * sin(theta) + pt.y * cos(theta))

def form_rect(x, y, theta, length, width):
    center = Point(x,y)
    l2 = length / 2
    w2 = width / 2

    # construct car rect at origin, rotate appropriately and then translate to its actual position
    a = rotate(Point(-l2, -w2), theta) + center
    b = rotate(Point(l2, -w2), theta) + center
    c = rotate(Point(-l2, w2), theta) + center
    d = rotate(Point(l2, w2), theta) + center

    return Rect(a,b,c,d)

def dot(pt1, pt2):
    return pt1.x * pt2.x + pt1.y * pt2.y

def vec(pt1, pt2):
    return Point(pt2.x - pt1.x, pt2.y - pt1.y)

def point_in_rect(pt, rect):
    """ Check if the point lies in the rectangle
        https://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not
    """
    ab = vec(rect.a, rect.b)
    am = vec(rect.a, pt)
    bc = vec(rect.b, rect.c)
    bm = vec(rect.b, pt)

    return 0 <= dot(ab, am) <= dot(ab, ab) and 0 <= dot(bc, bm) <= dot(bc, bc)

def intersect_circle(circle, pt1, pt2):
    """ Determine if the line segemnt between pt1 and pt2 intersects with the circle
        https://math.stackexchange.com/questions/275529/check-if-line-intersects-with-circles-perimeter
        (2nd answer that works for line segments)
    """

    ax = pt1.x - circle.x
    ay = pt1.y - circle.y
    bx = pt2.x - circle.x
    by = pt2.y - circle.y

    a = (bx - ax)**2 + (by - ay)**2
    b = 2*(ax*(bx - ax) + ay*(by - ay))
    c = ax**2 + ay**2 - circle.r**2

    disc = b**2 - 4*a*c
    if disc <= 0: return False
    sqrtdisc = sqrt(disc)

    t1 = (-b + sqrtdisc) / (2*a)
    t2 = (-b - sqrtdisc) / (2*a)

    if (0 < t1 < 1) or (0 < t2 < 1): return True

    return False

def collision(C: Circle, R: Rect):
    """ https://stackoverflow.com/questions/401847/circle-rectangle-collision-detection-intersection
        (2nd answer for non-axis aligned)
    """
    return (point_in_rect(Point(C.x, C.y), R) or
            intersect_circle(C, R.a, R.b) or
            intersect_circle(C, R.b, R.c) or
            intersect_circle(C, R.c, R.d) or
            intersect_circle(C, R.d, R.a))

if __name__ == "__main__":
    from diff_traj.cfg import cfg
    r = Rect(Point(1,2), Point(1,1), Point(3,1), Point(3,2))
    p = Point(2,1.5)
    assert point_in_rect(p, r)

    c = Circle(4, 3, 2)
    assert not point_in_rect(Point(c.x, c.y), r)
    assert collision(c, r)

    c2 = Circle(8, 2, 1)
    assert not point_in_rect(Point(c2.x, c2.y), r)
    assert not collision(c2, r)

    # Issues that occur if car length and width are wrong (should be collision free for all of them)
    # (S18, O1), (19, 1), (26,0), (31, 2), (32,3)
    # Obstacle 0: [19.83637047 -7.77676535  2.43013287]
    # Obstacle 1: [12.3777914   6.1901207   2.40626478]
    # Obstacle 2: [31.11339951 -5.0329566   2.13223362]
    # State 18: [10.68750095  0.          4.75        0]
    # State 19: [11.875       0.          4.99999809  0]
    # State 26: [21.9375  0.      6.75    0.    ]
    # State 31: [31.  0.  8.  0.]
    # State 32: [33.    0.    8.25  0.  ]

    o0 = Circle(19.83637047, -7.77676535,  2.43013287)
    o1 = Circle(12.3777914, 6.1901207, 2.40626478)
    o2 = Circle(31.11339951, -5.0329566, 2.13223362)

    s18 = form_rect(10.68750095, 0, 0, cfg.car_width, cfg.car_length)

    assert not collision(o1, s18)


    # More weird numerical issues
    # Obstacle 0: [22.32681656  2.87552452  2.48682308]
    # Obstacle 1: [39.42609406  1.69980526  2.75029588]
    # State 25: [20.27127647 -1.10565281  6.5        -0.15556586]
    # State 26: [21.87665367 -1.35742855  6.75        0.09358335]
    # State 27: [23.55676842 -1.19973755  7.          0.29575819]
    # State 28: [25.23078537 -0.68967247  7.25        0.3133443 ]
    # State 34: [35.5183754   4.85861301  8.75        0.43904358]
    # State 35: [37.49841309  5.78846169  9.          0.18904358]
    # State 36: [39.70832825  6.21128273  9.25       -0.06095648]
    # State 37: [42.0165329   6.07040596  9.5        -0.31095642]

    # o0 = Circle(22.32681656, 2.87552452, 2.48682308)
    # o1 = Circle(39.42609406,  1.69980526,  2.75029588)

    # s37 = form_rect(42.0165329,   6.07040596, -0.31095642, cfg.car_width, cfg.car_length)
    # assert not collision(o1, s37)

