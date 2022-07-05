import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.signal import savgol_filter

from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N as T_N_class
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.util.save_util import read_curve_from_file
from mrt_phase_pde.pde.own_method.extract_curves_from_T_N.extract_curves import get_curve_for_T_N
from config import equation_config


mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
delta_a = equation_config['delta_a']

limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)


def plot_T_N(D):
    # T_N = T_N_class.load(f'data/T_0_D_{D}_sim_n_thr_15.pickle')
    T_N = T_N_class.load(f'..\\result\\T_N_6_D_{D}.pickle')

    curves = get_curve_for_T_N(T_N, D)

    __x, __y = np.meshgrid(T_N.v, T_N.a)

    plt.figure()
    for i, curve in enumerate(curves):
        step_size = int(len(curve) / 3)
        curve[:, 1] = savgol_filter(curve[:, 1], step_size, np.min([step_size, 5]) - 1)
        if i == 0:
            plt.plot(curve[:, 0], curve[:, 1], 'b-', label='isochrones')
        else:
            plt.plot(curve[:, 0], curve[:, 1], 'b-')
        # plt.plot(curve[:, 0], curve[:, 1], 'b-')
        # plt.text(1, curve[-1, 1], '')

    plt.xlabel('v')
    plt.ylabel('a')
    plt.ylim([0, 3])
    plt.xlim([-1, 1])
    plt.title(f'$D={D}$, $\mu={mu}$, $\\tau_a={tau_a}$, $\Delta_a={delta_a}$')
    plt.plot(limit_cycle[:, 0], limit_cycle[:, 1], 'g.', label='deterministic limit cycle')
    plt.legend()

    # plt.savefig(f'img\\isochrones_for_D_{D}_comparison_numeric_isochrones.png')
    plt.savefig(f'img\\isochrones_for_D_{D}.png')

    det_file_paths = '..\\..\\..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries_grid\\deterministic\\D_0.0'
    stochastic = f'..\\..\\..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}'

    # if D == 0.0:
    #     isochrones = load_isochrones(det_file_paths)
    # else:
    #     isochrones = load_isochrones(stochastic)
    #
    # plot_isochrones(isochrones, plt, 'g--')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # plt.xlim([-1,1])

    plt.show()


if __name__ == '__main__':
    D_list = [0.0, 0.1, 0.25, 0.5, 1.0]

    for D in D_list:
        plot_T_N(D)