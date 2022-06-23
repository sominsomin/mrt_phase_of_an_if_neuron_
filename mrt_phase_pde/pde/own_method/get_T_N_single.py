import copy

import numpy as np
import matplotlib.pyplot as plt

from mrt_phase_numeric.src.update_equ.update import integrate_forwards
from mrt_phase_numeric.src.config import equation_config
from mrt_phase_pde.pde.own_method.src.exit_point_distribution.exit_point_distribution import ExitPointDistribution
from mrt_phase_pde.pde.own_method.src.T_0.T_0 import T_0
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N as T_N_class
from mrt_phase_numeric.isochrones.plot_isochrones.plot_util import load_isochrones

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
Delta_a = equation_config['delta_a']

D = .0

v_min = -1
v_max = 1
a_min = 0
a_max = 3

n_v = int(v_max - v_min) * 20 + 1
n_a = int(a_max - a_min) * 20 + 1

v = np.linspace(v_min, v_max, n_v)
a = np.linspace(a_min, a_max, n_a)


def plot_isochrones(isochrones_list, draw=None):
    legend_str = []
    for key in isochrones_list.keys():
        curves = isochrones_list[key]
        for i, curve in enumerate(curves):
            if curve.points.any():
                if draw:
                    plt.plot(curve[:, 0], curve[:, 1], draw)
                else:
                    plt.plot(curve[:, 0], curve[:, 1])


if __name__ == '__main__':
    prob = ExitPointDistribution.load(f'data/epd_sim_D_{D}.pickle')
    # T_0 = T_0.load(f'data/T_0_D_{D}_sim.pickle')
    T_0 = T_0.load(f'data/T_0_D_{D}_sim_n_thr_1.pickle')
    # T_0 = T_0.load('data/T_0_D_0.25_sim_n_thr_1.pickle')

    v = T_0.v
    a = T_0.a

    idx_v_zero = np.where(prob.v == 0)[0][0]

    # T N minus 1
    T_N_1 = copy.copy(T_0[np.where(T_0.v == 0)][0, :])

    T_N_all = [T_N_1]

    a_max = np.max(T_0.a)

    l_max = int(a_max) - 4 # 5

    for l in range(l_max):
        print(f'l: {l}')
        a_max -= 1

        n_a = int(a_max - a_min) * 20 + 1
        a_new = np.linspace(a_min, a_max, n_a)
        idx_a = np.where((T_0.a <= Delta_a + a_max) & (T_0.a >= Delta_a))
        # a_bins = np.insert(a, len(a), a[-1] + np.diff(a)[0])

        # T_N at v = 0
        T_N_0 = np.zeros(len(a_new))

        for j, _a in enumerate(a_new):
            # pass
            # hist = plt.hist(prob[idx_v_zero][j][:, 0], bins=a_bins, density=True)
            p = prob[idx_v_zero][j][:len(idx_a[0])]

            i_0 = np.where(T_0.v == 0)[0][0]
            j_0 = np.where(T_0.a == _a)[0][0]

            T_N_0[j] = T_0[i_0, j_0] + np.sum(p * T_N_1[idx_a]) * np.diff(a_new)[0]

        T_N_1 = copy.copy(T_N_0)
        T_N_all.append(T_N_0)

    plt.clf()

    for i, T in enumerate(T_N_all):
        if i == 0:
            continue
        diff = T - T_N_all[i - 1][:len(T)]
        plt.plot(np.linspace(a_min, np.max(T_0.a), int(np.max(T_0.a) - a_min) * 20 + 1)[:len(T)], diff)

    plt.title(f'$(T_N(0, a) - T_{{N-1}}(0, a))$ for various $N$, $D={D}$')
    plt.xlabel('a')
    plt.ylabel('$\Delta t$')
    plt.legend([f'$T_{{{i+1}}}(0, a) - T_{{{i}}}(0, a)$' for i in range(len(T_N_all))])
    plt.savefig(f'img\\difference_in_T_N_at_0_D_{D}.png')

    a_max -= 1
    n_a = int(a_max - a_min) * 20 + 1
    a_new = np.linspace(a_min, a_max, n_a)
    idx_a = np.where((T_0.a <= Delta_a + a_max) & (T_0.a >= Delta_a))
    a_bins = np.insert(a_new, len(a_new), a_new[-1] + np.diff(a_new)[0])

    T_N = np.zeros((len(v), len(a_new)))

    for i, _v in enumerate(v):
        for j, _a in enumerate(a_new):
            # pass
            p = prob[i][j][:len(idx_a[0])]

            i_0 = np.where(_v == T_0.v)[0][0]
            j_0 = np.where(_a == T_0.a)[0][0]

            T_add = np.sum(p * T_N_1[idx_a]) * np.diff(a_new)[0]
            T_N[i, j] = T_0[i_0, j_0] + T_add

    t_n = T_N_class(v, a_new, T_N, l + 1)
    # l + 1
    t_n.save(f'result\\T_N_{l + 1}_D_{D}.pickle')

    __x, __y = np.meshgrid(v, a_new)

    plt.figure()
    plt.contourf(__x, __y, T_N.transpose(), levels=30)  # , v_min=-30, v_max=30)
    plt.colorbar()
    plt.xlabel('v')
    plt.ylabel('a')
    plt.ylim([0, a_max])
    plt.title(
        f'$T_{{{l + 1}}}(v,a)$, $D={D}$')
        # f'\n $v_{{thr}}={v_th}$, $\\mu={mu}$, $\\tau_a={tau_a}$, $\\Delta_a={Delta_a}$')

    plt.savefig(f'img\\T_N_{l + 1}_D_{D}.png')

    det_file_paths = '..\\..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries_grid\\deterministic\\D_0.0'
    stochastic = f'..\\..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}'

    isochrones = load_isochrones(det_file_paths)
    # isochrones = load_isochrones(stochastic)
    plot_isochrones(isochrones, 'g--')
    plt.xlim([-1,1])
