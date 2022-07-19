import copy

import numpy as np
import matplotlib.pyplot as plt

from config import equation_config
from mrt_phase_pde.pde.own_method.src.exit_point_distribution.exit_point_distribution import ExitPointDistribution
from mrt_phase_pde.pde.own_method.src.T_0.T_0 import T_0 as T_0_class
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N as T_N_class
from mrt_phase_pde.pde.own_method.config_T_N import v_min, v_max, a_min, a_max
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.util.save_util import read_curve_from_file

limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
Delta_a = equation_config['delta_a']

D = .1


def get_T_N(l_max=None, D=D):
    prob = ExitPointDistribution.load(f'..\\data\\epd_sim_D_{D}.pickle')
    # T_0 = T_0.load(f'data/T_0_D_{D}_sim.pickle')
    T_0 = T_0_class.load(f'..\\data\\T_0_D_{D}_sim_n_thr_1.pickle')
    # T_0 = T_0.load('data/T_0_D_0.25_sim_n_thr_1.pickle')

    # just for saving T_0 as T_N, hackish
    if l_max == 0:
        t_n = T_N_class(T_0.v, T_0.a, T_0, 0)
        t_n.save(f'..\\result\\T_N_{0}_D_{D}.pickle')
        T_N = T_0.T
        v = T_0.v
        a_new = T_0.a
        a_max_temp = np.max(a_new)

        l = -1
    else:
        v = T_0.v
        a = T_0.a

        idx_v_zero = np.where(prob.v == 0)[0][0]

        # T N minus 1
        T_N_1 = copy.copy(T_0[np.where(T_0.v == 0)][0, :])

        T_N_all = [T_N_1]

        a_max_temp = np.max(T_0.a)
        a_new = copy.copy(a)

        if l_max is None:
            l_max = int(a_max_temp) - 4

        for l in range(l_max-1):
            print(f'l: {l}')
            a_max_temp -= Delta_a

            a_new = a_new[:np.where(a_new == a_max_temp)[0][0] + 1]
            idx_a = np.where((T_0.a <= Delta_a + a_max_temp) & (T_0.a >= Delta_a))

            # T_N at v = 0
            T_N_0 = np.zeros(len(a_new))

            for j, _a in enumerate(a_new):
                p = prob[idx_v_zero][j][:len(idx_a[0])]

                i_0 = np.where(T_0.v == 0)[0][0]
                j_0 = np.where(T_0.a == _a)[0][0]

                T_N_0[j] = T_0[i_0, j_0] + np.sum(p * T_N_1[idx_a]) * np.diff(a_new)[0]

            T_N_1 = copy.copy(T_N_0)
            T_N_all.append(T_N_0)

        a_max_temp -= Delta_a
        a_new = a_new[:np.where(a_new == a_max_temp)[0][0] + 1]
        idx_a = np.where((T_0.a <= Delta_a + a_max_temp) & (T_0.a >= Delta_a))

        T_N = np.zeros((len(v), len(a_new)))

        for i, _v in enumerate(v):
            for j, _a in enumerate(a_new):
                p = prob[i][j][:len(idx_a[0])]

                i_0 = np.where(_v == T_0.v)[0][0]
                j_0 = np.where(_a == T_0.a)[0][0]

                T_add = np.sum(p * T_N_1[idx_a]) * np.diff(a_new)[0]
                T_N[i, j] = T_0[i_0, j_0] + T_add

        t_n = T_N_class(v, a_new, T_N, l_max)
        t_n.save(f'..\\result\\T_N_{l_max}_D_{D}.pickle')

        # plot diff
        plt.figure()
        for i, T in enumerate(T_N_all):
            if i == 0:
                continue
            diff = T - T_N_all[i - 1][:len(T)]
            plt.plot(a[:len(T)], diff)

        plt.title(f'$(T_N(0, a) - T_{{N-1}}(0, a))$ for various $N$, $D={D}$')
        plt.xlabel('a')
        plt.ylabel('$\Delta t$')
        plt.legend([f'$T_{{{i + 1}}}(0, a) - T_{{{i}}}(0, a)$' for i in range(len(T_N_all))])
        plt.savefig(f'..\\img\\difference_in_T_N_at_0_D_{D}.png')

    # resize T_N for plot
    a_max_plot = a_max

    idx_max_a = np.where(a_new == a_max_plot)[0][0] + 1
    T_N = T_N[:, :idx_max_a]
    a_new = a_new[:idx_max_a]

    __x, __y = np.meshgrid(v, a_new)

    plt.figure()
    plt.contourf(__x, __y, T_N.transpose(), levels=30)  # , v_min=-30, v_max=30)
    plt.xlabel('v')
    plt.ylabel('a')
    # plt.ylim([0, a_max])
    plt.title(
        f'$T_{{{l_max}}}(v,a)$, $D={D}$')
        # f'\n $v_{{thr}}={v_th}$, $\\mu={mu}$, $\\tau_a={tau_a}$, $\\Delta_a={Delta_a}$')
    # plt.ylim([0, 3])
    plt.colorbar()
    plt.plot(limit_cycle[:, 0], limit_cycle[:, 1], 'g.', label='deterministic limit cycle')
    plt.legend()

    plt.savefig(f'..\\img\\T_N_{l_max}_D_{D}.png')

    # det_file_paths = '../../../../mrt_phase_numeric/data/results/isochrones/from_timeseries_grid/deterministic/D_0.0'
    # stochastic = f'..\\..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}'
    # isochrones = load_isochrones(det_file_paths)
    # isochrones = load_isochrones(stochastic)
    # plot_isochrones(isochrones, plt, 'g--')
    plt.xlim([-1,1])

    # plt.show()


if __name__ == '__main__':
    get_T_N(6)

