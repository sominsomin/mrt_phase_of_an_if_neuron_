import numpy as np
import matplotlib.pyplot as plt

from mrt_phase_numeric.src.update_equ.update import integrate_forwards
from mrt_phase_pde.pde.own_method.src.exit_point_distribution.exit_point_distribution import ExitPointDistribution

D = 0.0
dt = 0.01

v_min = 1
v_max = 1
a_min = 0
a_max = 20

n_v = int(v_max - v_min) * 20 + 1
n_a = int(a_max - a_min) * 10 + 1

v = np.linspace(v_min, v_max, n_v)
a = np.linspace(a_min, a_max, n_a)

v_thr = 1.0
n_trajectories = 1

n_thr_crossings = 1


def update_until_line_cross(_v, _a, n_thr_crossings=n_thr_crossings):
    v_at_thr = []
    a_at_thr = []
    for i in range(n_thr_crossings):
        v_, a_, y_ = integrate_forwards(_v, _a, 1, D, dt)

        _v = v_[-1]
        _a = a_[-1]

        v_at_thr.append(_v)
        a_at_thr.append(_a)

    return v_at_thr, a_at_thr


def get_a_distr(v, a, n_thr_crossings=n_thr_crossings):
    a_distr = []
    for i in range(n_trajectories):
        v_, a_ = update_until_line_cross(v, a, n_thr_crossings=n_thr_crossings)
        a_distr.append(a_)

    a_distr = np.array(a_distr)
    return a_distr


def get_exit_point_distribution(v, a):
    prob = [[None for j in a] for i in v]

    a_bins = np.insert(a, len(a), a[-1] + np.diff(a)[0])
    a_bins = a_bins - np.diff(a)[0]/2

    for i, _v in enumerate(v):
        print(i)
        print(f'v: {_v}')
        for j, _a in enumerate(a):
            print(f'a: {_a}')
            a_distr = get_a_distr(_v, _a)
            a_distr = a_distr[:, 0]

            hist = plt.hist(a_distr, bins=a_bins, density=True)
            p = hist[0]

            prob[i][j] = p

    return prob


if __name__ == '__main__':
    # for a  in range(0, 20):
    #     v_, a_, y_ = integrate_forwards(1.0, a, 1, D, dt)
    #     print(len(v_) * dt)
    #     plt.plot(v_,a_)

    prob = get_exit_point_distribution(v, a)

    # exit_point_distribution = ExitPointDistribution(v, a, prob)
    # exit_point_distribution.save(f'..\\..\\data\\epd_sim_D_{D}.pickle')
