import numpy as np
import matplotlib.pyplot as plt
import math

from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N as T_N_class
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.util.save_util import read_curve_from_file
from mrt_phase_pde.pde.own_method.extract_curves_from_T_N.extract_curves import get_curve_for_T_N


limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)

D = .25


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


if __name__ == '__main__':
    # T_N = T_N_class.load(f'data/T_0_D_{D}_sim_n_thr_15.pickle')

    T_N = T_N_class.load(f'..\\result\\T_N_6_D_{D}.pickle')

    curves = get_curve_for_T_N(T_N, D)

    __x, __y = np.meshgrid(T_N.v, T_N.a)

    plt.figure()
    for curve in curves:
        plt.plot(curve[:, 0], curve[:, 1], 'b-')
        plt.text(1, curve[-1, 1], '')

    plt.xlabel('v')
    plt.ylabel('a')
    plt.ylim([0, 4])

    plt.plot(limit_cycle[:, 0], limit_cycle[:, 1], 'g.')

    det_file_paths = '../../../../mrt_phase_numeric/data/results/isochrones/from_timeseries_grid/deterministic/D_0.0'
    stochastic = f'..\\..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}'

    # isochrones = load_isochrones(det_file_paths)
    # isochrones = load_isochrones(stochastic)
    # plot_isochrones(isochrones, plt, 'g--')
    # plt.xlim([-1,1])


    plt.show()
