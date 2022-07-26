import numpy as np
import timeit

from mrt_phase_algorithmic.src.line_intersection.line_intersection import curves_overlap, get_curve_intersection,\
    get_curve_intersection_custom, get_curve_intersection_faster, intersect_shapely
from mrt_phase_algorithmic.src.line_intersection.test_alt_line_segment_intersection import seg_intersect

test_func = [
    get_curve_intersection,
    get_curve_intersection_faster,
    get_curve_intersection_custom,
    intersect_shapely,
    # seg_intersect,
]

if __name__ == '__main__':
    x = np.arange(0, 1000)
    f = np.arange(0, 1000)
    g = np.sin(np.arange(0, 10, 0.01) * 2) * 1000 + 1

    curve_1 = np.column_stack((x, f))
    curve_2 = np.column_stack((x, g))

    curve_1 = curve_1[1:]
    curve_2 = curve_2[1:]

    for func in test_func:
        start = timeit.default_timer()

        res = func(curve_1, curve_2)

        stop = timeit.default_timer()
        print(res)
        print(f'{func.__name__} Time: ', stop - start)
