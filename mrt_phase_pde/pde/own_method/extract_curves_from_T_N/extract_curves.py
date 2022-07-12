import numpy as np
import math
import matplotlib.pyplot as plt_

from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.util.save_util import read_curve_from_file
from mrt_phase_pde.pde.own_method.T_bar import T_bar
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N

limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def get_curves_from_cs(cs):
    all_curves = []
    for collection in cs.collections:
        try:
            p = collection.get_paths()[0]
        except:
            print()
        curve = p.vertices
        all_curves.append(curve)

    return all_curves


phi_ = np.linspace(0.0, 0.9, 10)


def get_levels(T_N: T_N, D: float, type=None):
    if D == 0.0 or type == 'deterministic':
        levels = []
        for phi in phi_:
            point = limit_cycle[int(np.floor((phi * (len(limit_cycle) - 1)))), :]

            idx_v = find_nearest(T_N.v, point[0])
            idx_a = find_nearest(T_N.a, point[1])

            level = T_N[idx_v, idx_a]
            level_2 = level + T_bar[D]
            level_1 = level - T_bar[D]
            level_0 = level - 2 * T_bar[D]

            levels.append(level)
            if level_2 < np.max(T_N.T):
                levels.append(level_2)
            if level_1 > np.min(T_N.T):
                levels.append(level_1)
            if level_0 > np.min(T_N.T):
                levels.append(level_0)

        levels = np.sort(levels)
    else:
        min_T = np.min(T_N.T)
        max_T = np.max(T_N.T)
        mean_T = T_bar[D]

        stepsize = np.diff(phi_)[0]
        levels = np.arange(min_T, max_T, mean_T * stepsize)

    return levels


def get_curve_for_T_N(T_N: T_N, D: float, type=None):
    levels = get_levels(T_N, D, type=type)

    __x, __y = np.meshgrid(T_N.v, T_N.a)
    cs = plt_.contour(__x, __y, T_N.T.transpose(), levels=np.sort(levels)[1:])  # , v_min=-30, v_max=30)
    all_curves = get_curves_from_cs(cs)

    return all_curves