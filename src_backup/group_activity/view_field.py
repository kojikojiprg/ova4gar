from typing import List, Tuple

import numpy as np
from common.functions import rotation


class Line:
    def __init__(self, point, vector, length):
        self.p1 = point
        self.p2 = point + vector * length

    def is_online(self, point):
        x = point[0]
        y = point[1]

        if int(self.func(x) - y) == 0:
            return self.limit(x, y)
        else:
            return False


class ViewField:
    def __init__(self, position, vector, angle):
        self.left_vector = rotation(vector, -1 * angle)
        self.right_vector = rotation(vector, angle)

        self.left_line = Line(position, self.left_vector)
        self.right_line = Line(position, self.right_vector)


def calc_cross(l1: Line, l2: Line):
    (x11, y11), (x12, y12) = l1.p1, l1.p2
    (x21, y21), (x22, y22) = l2.p1, l2.p2

    a1, b1 = x12 - x11, y12 - y11
    a2, b2 = x22 - x21, y22 - y21

    d = a1 * b2 - a2 * b1
    if d == 0:
        # two lines are parallel
        return None

    sn = b2 * (x21 - x11) - a2 * (y21 - y11)
    s = sn / d
    if 0 <= s and s <= 1:
        x = x11 + a1 * s
        y = y11 + b1 * s
        return x, y
    else:
        return None


def inside_polygon(point: tuple, polygon_points: List[Tuple]):
    # Ray casting algorithm
    cnt = 0
    x, y = point
    for i in range(len(polygon_points)):
        x0, y0 = polygon_points[i - 1]
        x1, y1 = polygon_points[i]
        x0 -= x
        y0 -= y
        x1 -= x
        y1 -= y

        cv = x0 * x1 + y0 * y1
        sv = x0 * y1 - x1 * y0
        if sv == 0 and cv <= 0:
            # a point is on a segment
            return True

        if not y0 < y1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        if y0 <= 0 < y1 and x0 * (y1 - y0) > y0 * (x1 - x0):
            cnt += 1
    return cnt % 2 == 1


def calc_overlap(vf1: ViewField, vf2: ViewField):
    lv1, rv1 = vf1.left_vector, vf1.right_vector
    lv2, rv2 = vf2.left_vector, vf2.right_vector

    cross_points = []
    point = calc_cross(lv1, lv2)
    if point is not None:
        cross_points.append(point)

    point = calc_cross(lv1, rv2)
    if point is not None:
        cross_points.append(point)

    point = calc_cross(rv1, lv2)
    if point is not None:
        cross_points.append(point)

    point = calc_cross(rv1, rv2)
    if point is not None:
        cross_points.append(point)

    bbox = [np.min(cross_points, axis=0), np.max(cross_points, axis=0)]

    return cross_points, bbox
