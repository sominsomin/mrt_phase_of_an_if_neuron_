import numpy as np
import timeit

from mrt_phase_algorithmic.src.line_intersection.line_intersection import curves_overlap, get_curve_intersection, get_curve_intersection_custom
from shapely.geometry import LineString


class Segments:
    def intersection(self, s1, s2):
        left = max(min(s1[0], s1[2]), min(s2[0], s2[2]))
        right = min(max(s1[0], s1[2]), max(s2[0], s2[2]))
        top = max(min(s1[1], s1[3]), min(s2[1], s2[3]))
        bottom = min(max(s1[1], s1[3]), max(s2[1], s2[3]))

        if top > bottom or left > right:
            return ('NO INTERSECTION', list())
        if (top, left) == (bottom, right):
            return ('POINT INTERSECTION', list((left, top)))
        return ('SEGMENT INTERSECTION', list((left, bottom, right, top)))
