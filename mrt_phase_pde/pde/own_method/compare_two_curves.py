import copy

import numpy as np
import matplotlib.pyplot as plt
import math

from mrt_phase_numeric.src.update_equ.update import integrate_forwards
from mrt_phase_numeric.src.config import equation_config
from mrt_phase_pde.pde.own_method.src.exit_point_distribution.exit_point_distribution import ExitPointDistribution
from mrt_phase_pde.pde.own_method.src.T_0.T_0 import T_0
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N as T_N_class
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.isochrones.plot_isochrones.plot_util import load_isochrones, plot_isochrones
from mrt_phase_numeric.src.util.save_util import read_curve_from_file
from mrt_phase_numeric.src.config import equation_config
from mrt_phase_pde.pde.own_method.extract_curves_from_T_N import get_curve_from_cs

limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)

phi_ = np.linspace(0.0, 1, 11)


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def get_curve_for_T_N(T_N):
    levels = []
    for phi in phi_:
        point = limit_cycle[int(np.floor((phi * (len(limit_cycle) - 1)))), :]

        idx_v = find_nearest(T_N.v, point[0])
        idx_a = find_nearest(T_N.a, point[1])

        levels.append(T_N[idx_v, idx_a])

    __x, __y = np.meshgrid(T_N.v, T_N.a)
    cs = plt.contour(__x, __y, T_N.T.transpose(), levels=np.sort(levels)[1:])  # , v_min=-30, v_max=30)
    all_curves = get_curve_from_cs(cs)

    return all_curves


D_list = [0.0, 0.25]

if __name__ == '__main__':
    # T_N = T_N_class.load(f'data/T_0_D_{D}_sim_n_thr_15.pickle')

    plt.figure()

    draw_style = ['b-', 'r-'] #, 'm:', 'c.' ]

    for j, D in enumerate(D_list):
        T_N = T_N_class.load(f'result/T_N_6_D_{D}.pickle')
        T_N_curves = get_curve_for_T_N(T_N)

        for i, curve in enumerate(T_N_curves):
            if i == 0:
                plt.plot(curve[:, 0], curve[:, 1], draw_style[j], label=f'D={D}')
            else:
                plt.plot(curve[:, 0], curve[:, 1], draw_style[j])

    plt.plot(limit_cycle[:, 0], limit_cycle[:, 1], 'g.', label='limit cycle')

    plt.xlabel('v')
    plt.ylabel('a')
    plt.ylim([0, 4])
    plt.title(f'Isochrones')
    plt.legend()

    # det_file_paths = '..\\..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries_grid\\deterministic\\D_0.0'
    # stochastic = f'..\\..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}'

    # isochrones = load_isochrones(det_file_paths)
    # isochrones = load_isochrones(stochastic)
    # plot_isochrones(isochrones, plt, 'g--')
    # plt.xlim([-1,1])
