import numpy as np
import matplotlib.pyplot as plt
import pickle

from mrt_phase_numeric.src.update_equ.update import update_, integrate_forwards
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.util.save_util import read_curve_from_file
from mrt_phase_numeric.isochrones.plot_isochrones.plot_util import load_isochrones


limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)

D = 0.25
dt = 0.01

v_min = 0.5
v_max = 1
a_min = 0
a_max = 5

n_v = 4
n_a = int(a_max - a_min) * 10 + 1

# n_v = 20
# n_a = 40

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


if __name__ == '__main__':
    # prob = get_exit_point_distribution()

    v = 0.5
    a = 1.0

    plt.figure()
    legend = []

    for i in range(1, 10):
        print(i)

        a_distr = get_a_distr(v, a, n_thr_crossings=i)
        legend.append(i)

        plt.figure()
        plt.hist(a_distr, bins=20)
        plt.xlim([0, 2])
        plt.title(f'n {i}')

    # plt.legend(legend)
    # plt.title(f'exit point distribution (v = 1.0) after n thr crossings for v = {v}, a= {a}')
    plt.xlabel('a')
    plt.show()


    # V, A = np.meshgrid(v, a)
    #
    # plt.contourf(V, A, prob.transpose(), levels=30)
    # plt.xlabel('v')
    # plt.ylabel('a')
    # plt.colorbar()
    #
    # plt.plot(limit_cycle[:, 0], limit_cycle[:, 1])
    #
    # plt.title(f'mean first passage time to do {n_thr_crossings} threshold-crossings\n(n trajectories {n_trajectories}), $D = {D}$, $v_{{thr}} = {v_thr}$')
    #
    # # plt.savefig(f'../img/long_timeseries_mfpt_solution_D_{D}_n_trjectories_{n_trajectories}_n_trh_crossings_{n_thr_crossings}.png')
    # plt.savefig(f'../img/comparison_long_timeseries_mfpt_solution_D_{D}_n_trjectories_{n_trajectories}_n_trh_crossings_{n_thr_crossings}.png')
