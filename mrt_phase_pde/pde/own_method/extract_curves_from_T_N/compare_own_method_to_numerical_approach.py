import matplotlib.pyplot as plt
import numpy as np

from collections import OrderedDict
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N as T_N_class
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.util.save_util import read_curve_from_file
from scipy.signal import savgol_filter
from mrt_phase_pde.pde.own_method.extract_curves_from_T_N.extract_curves import get_curve_for_T_N
from config import equation_config
from mrt_phase_numeric.isochrones.plot_isochrones.plot_util import load_isochrones, plot_isochrones
from mrt_phase_pde.pde.own_method.config_T_N import v_min, v_max, a_min, a_max


mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
delta_a = equation_config['delta_a']

limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)


D = 1.


if __name__ == '__main__':
    plt.figure()
    draw_style = ['b-', 'r--', 'm-']  # , 'm:', 'c.' ]

    T_N = T_N_class.load(f'..\\result\\T_N_6_D_{D}.pickle')
    T_N_curves = get_curve_for_T_N(T_N, D)  # , type='deterministic')

    plt.plot(limit_cycle[:, 0], limit_cycle[:, 1], 'g.', label='deterministic limit cycle')

    for i, curve in enumerate(T_N_curves):
        step_size = int(len(curve) / 3)
        curve[:, 1] = savgol_filter(curve[:, 1], step_size, np.min([step_size, 5]) - 1)

        if i == 0:
            plt.plot(curve[:, 0], curve[:, 1], draw_style[0], label=f'isochrones')
        else:
            plt.plot(curve[:, 0], curve[:, 1], draw_style[0])

    plt.xlabel('v')
    plt.ylabel('a')
    plt.ylim([a_min, a_max])
    # plt.title(f'Isochrones for $\mu={mu}$, $\\tau_a={tau_a}$, $\Delta_a={delta_a}$')
    plt.title(f'$D={D}$')
    # plt.show()

    det_file_paths = '..\\..\\..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries_grid\\deterministic\\D_0.0'
    stochastic = f'..\\..\\..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}'

    if D == 0.0:
        isochrones = load_isochrones(det_file_paths)
    else:
        isochrones = load_isochrones(stochastic)

    for key in isochrones.keys():
        curves = isochrones[key]
        for i, curve in enumerate(curves):
            if curve.points.any() and len(curve.points) > 5: # i == 1:
                plt.plot(curve[:, 0], curve[:, 1], 'm--', label='numerical isochrones') #, label='stochastic isochrones')

    plt.xlim([-1, 1])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(f'img\\isochrones_D_{D}_to_numerics.png')