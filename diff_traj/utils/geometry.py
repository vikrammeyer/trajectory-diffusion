from collections import namedtuple
from dataclasses import dataclass
import math

Point = namedtuple("Point", ["x", "y"])
Circle = namedtuple("Circle", ["x", "y", "r"])
Rect = namedtuple("Rect", ["a", "b", "c", "d"])


def standard_line(pt1, pt2):
    """ Given two points (x1, y1) and (x2, y2)
        return the coefficients for ax + by + c 0
        https://math.stackexchange.com/questions/422602/convert-two-points-to-line-eq-ax-by-c-0
    """
    a = pt1.y - pt2.y
    b = pt2.x - pt1.x
    c = pt1.x * pt2.y - pt2.x * pt1.y

    return (a, b, c)

def point_on_lhs(pt, start_pt, end_pt):
    d = (start_pt.x - end_pt.x) * (pt.y - end_pt.y) - (pt.x - end_pt.x) * (start_pt.y - end_pt.y)
    return d > 0

def point_in_rect(pt, rect):
    """ Check if the point lies in the rectangle
        https://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not
    """
    return point_on_lhs(pt, rect.a, rect.d) or \
            point_on_lhs(pt, rect.d, rect.c) or \
            point_on_lhs(pt, rect.c, rect.b) or \
            point_on_lhs(pt, rect.b, rect.a)

def intersect_circle(circle, pt1, pt2):
    """ Determine if the line segemnt between pt1 and pt2 intersects with the circle
        https://math.stackexchange.com/questions/275529/check-if-line-intersects-with-circles-perimeter
        https://www.geeksforgeeks.org/check-line-touches-intersects-circle/
    """
    a, b, c = standard_line(pt1, pt2) # (x1, y1), (x2,y2) to ax + by + c = 0

    dist_line_to_center = ((abs(a * circle.x + b * circle.y + c) / math.sqrt(a**2 + b**2)))

    return circle.radius > dist_line_to_center

def collision(C: Circle, R: Rect):
    return point_in_rect(Point(C.x, C.y), R) or \
            intersect_circle(C, (R.a, R.b)) or \
            intersect_circle(C, (R.b, R.c)) or \
            intersect_circle(C, (R.c, R.d)) or \
            intersect_circle(C, (R.d, R.a))

if __name__ == "__main__":
    r = Rect(Point(1,2), Point(3,2), Point(3,1), Point(1,1))
    c = Circle(4, 3, 2)
    c2 = Circle(12, 1, 1)

    assert collision(c, r)
    assert not collision(c2, r)