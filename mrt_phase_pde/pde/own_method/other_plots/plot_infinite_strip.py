import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import equation_config
from mrt_phase_pde.pde.own_method.src.exit_point_distribution.exit_point_distribution import ExitPointDistribution
from mrt_phase_pde.pde.own_method.src.T_0.T_0 import T_0 as T_0_class
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N as T_N_class
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.update_equ.update import integrate_forwards, update_
from mrt_phase_numeric.src.util.save_util import read_curve_from_file

limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
Delta_a = equation_config['delta_a']

D = .0

v_min = -0.5
v_max = 1
a_min = 0
a_max = 1

N = 5
dt = 0.1


if __name__ == '__main__':
    pass

    v_init = .5
    a_init = .5

    v_all = []
    a_all = []
    y_all = []

    for n in range(N):
        v_, a_, y_ = integrate_forwards(v_init, a_init, 1, D, dt)

        a_all.extend((np.array(a_) - n*Delta_a).tolist())
        v_all.extend((np.array(v_) + n*v_th).tolist())
        y_all.extend(y_)

        v_new, a_new, y_new, offset = update_(v_[-1], a_[-1], y_[-1], dt=dt, D=D)
        v_init = v_new
        a_init = a_new

    fig, axs = plt.subplots()
    axs.plot(v_all, a_all, 'b')
    # for entry in y_newy

    for reset in np.where(np.array(y_all) == 1)[0]:
        plt.plot(v_all[reset], a_all[reset], 'rx')

    da = 0.1
    dv = 0.1

    plt.plot(v_all[0], a_all[0], 'bx')
    plt.text(v_all[0], a_all[0], '$(v,a)$')

    for i in range(N):
        if i == 0:
            line_x = Line2D([v_min, i + v_th + dv], [-i*Delta_a, -i*Delta_a], linestyle='-', color='k')
        else:
            line_x = Line2D([i, i + v_th + dv], [-i * Delta_a, -i * Delta_a], linestyle='-', color='k')
        line_y = Line2D([i, i], [-i*Delta_a, a_max], linestyle='-', color='k')
        axs.add_line(line_x)
        axs.add_line(line_y)

        # text
        plt.text(i - dv, -i*Delta_a - da, '0')
        plt.text(i + v_th - dv, -i * Delta_a - da, '$v_{thr}$')

        plt.text((i - i + v_th + dv)/2 + i - dv, a_max - Delta_a/3, f'$T_{{{N-i}}}$')

        # arrow
        plt.arrow(i, a_max, 0.0 , 0.0, width=0.015, color='k')
        plt.text(i - dv, a_max, 'a')
        plt.arrow(i + v_th + dv, -i * Delta_a, 0.01, 0.0, width=0.015, color='k')
        plt.text(i + v_th + 2*dv,  -i * Delta_a - 0.5*da, 'v')
        # plt.text(i - 2 * dv, a_max, 'v')

        # Delta a
        if i != 0:
            delta_a_line_min = -i * Delta_a #+ da
            delta_a_line_max = -i * Delta_a + Delta_a #- da
            delta_a_line_v = i-1.5*dv
            line_delta_a = Line2D([delta_a_line_v, delta_a_line_v], [delta_a_line_min, delta_a_line_max], linestyle='-', color='k')
            plt.arrow(delta_a_line_v, delta_a_line_min, 0.0, -.01, width=0.01, color='k', length_includes_head=True)
            plt.arrow(delta_a_line_v, delta_a_line_max, 0.0, .01, width=0.01, color='k', length_includes_head=True)
            axs.add_line(line_delta_a)
            plt.text(i - 3*dv, (delta_a_line_max - delta_a_line_min)/2 + delta_a_line_min, '$\\Delta_a$')

    # final line
    line_y = Line2D([i+v_th, i+v_th], [-i * Delta_a, a_max], linestyle='-', color='orange')
    axs.add_line(line_y)
    plt.text(i + v_th + dv, (-i * Delta_a - a_max)/2 + a_max, 'l', color="orange")

    plt.arrow(i+v_th, a_max, 0.0, 0.0, width=0.015, color='k')
    plt.text(i+v_th - dv, a_max, 'a')


    # ax = plt.gca()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    plt.axis('off')

    plt.savefig(f'calculate_T_{N}_visualization_D_{D}.png')
