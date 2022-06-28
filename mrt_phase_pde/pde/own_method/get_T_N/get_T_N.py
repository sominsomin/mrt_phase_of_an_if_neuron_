import copy

import numpy as np
import matplotlib.pyplot as plt

from mrt_phase_numeric.src.update_equ.update import integrate_forwards
from config import equation_config
from mrt_phase_pde.pde.own_method.src.exit_point_distribution.exit_point_distribution import ExitPointDistribution
from mrt_phase_pde.pde.own_method.src.T_0.T_0 import T_0

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
Delta_a = equation_config['delta_a']


D = 0.0
dt = 0.01

v_min = -1
v_max = 1
a_min = 0
a_max = 4

n_v = int(v_max - v_min) * 10 + 1
n_a = int(a_max - a_min) * 10 + 1

v = np.linspace(v_min, v_max, n_v)
a = np.linspace(a_min, a_max, n_a)

v_thr = 1.0
n_trajectories = 1000

n_thr_crossings = 1


def update_until_line_cross(v, a, n_thr_crossings=n_thr_crossings):
    v_, a_, y_ = integrate_forwards(v, a, n_thr_crossings, D, dt)

    return v_[-1], a_[-1]


def get_a_distr(v, a, n_thr_crossings=n_thr_crossings):
    a_distr = []
    for i in range(n_trajectories):
        v_, a_ = update_until_line_cross(v, a, n_thr_crossings=n_thr_crossings)
        a_distr.append(a_)

    return a_distr


def get_exit_point_distribution():
    prob = [[None for j in a] for i in v]

    for i, _v in enumerate(v):
        print(i)
        for j, _a in enumerate(a):
            a_distr = get_a_distr(_v, _a)
            prob[i][j] = a_distr

    return prob


def get_T_0():
    pass


if __name__ == '__main__':
    prob = ExitPointDistribution.load(f'data/epd_sim_D_{D}.pickle')
    T_0 = T_0.load('data/T_0.pickle')

    idx_v_zero = np.where(T_0.v == 0)[0][0]

    T_N = np.zeros((len(v), len(a)))
    T_N_1 = copy.copy(T_0)

    T_N_all = [T_0]

    a_max = 10

    for l in range(10):
        a_min = 0
        a_max -= 1

        n_a = int(a_max - a_min) * 10 + 1

        a = np.linspace(a_min, a_max, n_a)
        idx_a = np.where((T_0.a <= Delta_a + a_max) & (T_0.a >= Delta_a))

        T_N = np.zeros((len(v), len(a)))

        a_bins = np.insert(a, len(a), a[-1] + np.diff(a)[0])

        for i, _v in enumerate(v):
            for j, _a in enumerate(a):
                # pass
                hist = plt.hist(prob[i][j][:, 0], bins=a_bins, density=True)
                p = hist[0]

                i_0 = np.where(_v == T_0.v)[0][0]
                j_0 = np.where(_a == T_0.a)[0][0]

                T_N[i, j] = T_0[i_0, j_0] + np.sum(p * T_N_1[idx_v_zero, idx_a]) * np.diff(a)[0]

        T_N_1 = copy.copy(T_N)
        T_N_all.append(T_N)

    #
    # T_1 = T_0.transpose() +
    #
    __x, __y = np.meshgrid(v, a)

    # plt.figure()
    # plt.contourf(__x, __y, T_0.transpose(), levels=20)  # , v_min=-30, v_max=30)
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('y')

    plt.figure()
    plt.contourf(__x, __y, T_N.transpose(), levels=20)  # , v_min=-30, v_max=30)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')







