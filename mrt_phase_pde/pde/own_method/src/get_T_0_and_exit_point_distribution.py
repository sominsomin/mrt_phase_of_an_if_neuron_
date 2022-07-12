import numpy as np
import matplotlib.pyplot as plt
import math

from config import equation_config
from mrt_phase_pde.pde.own_method.src.T_0.T_0 import T_0
from mrt_phase_numeric.src.update_equ.update import integrate_forwards
from mrt_phase_pde.pde.own_method.src.exit_point_distribution.exit_point_distribution import ExitPointDistribution


mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
Delta_a = equation_config['delta_a']

D = .0
dt = 0.01

v_min = -1
v_max = 1.0
a_min = 0
a_max = 90

n_v = int((v_max - v_min)) * 20 + 1
n_a = int((a_max - a_min)) * 10 + 1

# n_v = 20
# n_a = 40

v = np.linspace(v_min, v_max, n_v)
a = np.linspace(a_min, a_max, n_a)

v_thr = 1.0
n_trajectories = 1

n_thr_crossings = 1


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def update_until_line_cross(v, a):
    v_, a_, y_ = integrate_forwards(v, a, n_thr_crossings, D, dt)

    return v_, a_


def custom_hist(value, a):
    """used for n trajectories == 1, to linearly interpolate bins"""
    hist = np.zeros(len(a))

    if value in a:
        idx_exact = np.where(value == a)[0][0]
        hist[idx_exact] = 1
    else:
        idx_left = np.searchsorted(a, value, side="left") - 1
        idx_right = idx_left + 1
        value_left = a[idx_left]
        value_right = a[idx_right]

        diff = value_right - value_left
        weight_left = (value - value_left)/diff
        weight_right = (value_right - value)/diff

        hist[idx_left] = weight_left
        hist[idx_right] = weight_right

    return hist * 1/np.diff(a)[0]


def get_rt_a_distr(_v, _a):
    rt = []
    a_distr = []

    a_bins = np.insert(a, len(a), a[-1] + np.diff(a)[0])
    a_bins = a_bins - np.diff(a)[0] / 2

    for i in range(n_trajectories):
        v_, a_ = update_until_line_cross(_v, _a)

        n_timesteps = len(v_)
        rt.append(n_timesteps * dt)
        a_distr.append(a_[-1])

    a_distr = np.array(a_distr)
    # hist = plt.hist(a_distr, bins=a_bins, density=True)
    # if n_trajectories == 1:
    #     hist = custom_hist(a_distr[0], a)
    #     p = hist
    # else:
    hist = np.histogram(a_distr, bins=a_bins, density=True)
    p = hist[0]

    idx = find_nearest(a_bins, a_[-1])

    mean_rt = np.mean(rt)

    return mean_rt, p


def _get():
    T = np.zeros((len(v), len(a)))
    prob = [[None for j in a] for i in v]

    for i, _v in enumerate(v):
        print(f'v : {_v}')
        for j, _a in enumerate(a):
            print(f'a : {_a}')
            mean_rt, p = get_rt_a_distr(_v, _a)

            prob[i][j] = p
            T[i, j] = mean_rt

    return T, prob


if __name__=='__main__':
    T, prob = _get()

    T_0_ = T_0(v, a, T)
    T_0_.save(f'..\\data\\T_0_D_{D}_sim_n_thr_{n_thr_crossings}.pickle')

    exit_point_distribution = ExitPointDistribution(v, a, prob)
    exit_point_distribution.save(f'..\\data\\epd_sim_D_{D}.pickle')

    __x, __y = np.meshgrid(v, a)

    plt.figure()
    plt.contourf(__x, __y, T.transpose(), levels=20) #, v_min=-30, v_max=30)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
