import matplotlib.pyplot as plt
import numpy as np

from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N as T_N_class
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.util.save_util import read_curve_from_file
from scipy.signal import savgol_filter
from mrt_phase_pde.pde.own_method.extract_curves_from_T_N.extract_curves import get_curve_for_T_N
from config import equation_config

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
delta_a = equation_config['delta_a']

limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)

D_list = [0.5, 1.0]

if __name__ == '__main__':

    plt.figure()
    draw_style = ['b-', 'r--', 'm-'] #, 'm:', 'c.' ]

    for j, D in enumerate(D_list):
        T_N = T_N_class.load(f'..\\result\\T_N_6_D_{D}.pickle')
        T_N_curves = get_curve_for_T_N(T_N, D )#, type='deterministic')

        for i, curve in enumerate(T_N_curves):
            curve[:, 1] = savgol_filter(curve[:, 1], len(curve), np.min([5, len(curve)]) - 1)

            if i == 0:
                plt.plot(curve[:, 0], curve[:, 1], draw_style[j], label=f'D={D}')
            else:
                plt.plot(curve[:, 0], curve[:, 1], draw_style[j])

    plt.plot(limit_cycle[:, 0], limit_cycle[:, 1], 'g.', label='deterministic limit cycle')

    plt.xlabel('v')
    plt.ylabel('a')
    plt.ylim([0, 3])
    plt.title(f'Isochrones for $\mu={mu}$, $\\tau_a={tau_a}$, $\Delta_a={delta_a}$')
    # plt.title(f'$D={D}$, ')
    plt.legend()

    plt.show()

    plt.savefig(f'img\\isochrones_comparison_D_{D_list}.png')
