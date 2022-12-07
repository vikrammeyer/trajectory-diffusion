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
    r = Rect(Point(1,2), Point(1,1), Point(3,1), Point(3,2))
    p = Point(2,1.5)
    assert point_in_rect(p, r)

    c = Circle(4, 3, 2)
    assert not point_in_rect(Point(c.x, c.y), r)
    assert collision(c, r)

    c2 = Circle(8, 2, 1)
    assert not point_in_rect(Point(c2.x, c2.y), r)
    assert not collision(c2, r)
