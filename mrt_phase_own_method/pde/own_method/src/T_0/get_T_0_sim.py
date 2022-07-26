import numpy as np
import matplotlib.pyplot as plt

from config import equation_config
from mrt_phase_own_method.pde.own_method.src.T_0.T_0 import T_0
from mrt_phase_algorithmic.src.update_equ.update import integrate_forwards

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
Delta_a = equation_config['delta_a']

D = 0.0
dt = 0.01

v_min = -1
v_max = 1.0
a_min = 0
a_max = 10

n_v = int((v_max - v_min)) * 20 + 1
n_a = int((a_max - a_min)) * 10 + 1

# n_v = 20
# n_a = 40

v = np.linspace(v_min, v_max, n_v)
a = np.linspace(a_min, a_max, n_a)

v_thr = 1.0
n_trajectories = 1

n_thr_crossings = 1


def update_until_line_cross(v, a):
    v_, a_, y_ = integrate_forwards(v, a, n_thr_crossings, D, dt)

    n_timesteps = len(v_)
    return n_timesteps


def get_mean_rt(v, a):
    rt = []
    for i in range(n_trajectories):
        n_timesteps = update_until_line_cross(v, a)
        rt.append(n_timesteps * dt)

    return np.mean(rt)


def get_mean_T():
    T = np.zeros((len(v), len(a)))

    for i, _v in enumerate(v):
        print(f'v : {_v}')
        for j, _a in enumerate(a):
            print(f'a : {_a}')
            mean_rt = get_mean_rt(_v, _a)
            T[i, j] = mean_rt

    return T


if __name__=='__main__':

    T = get_mean_T()

    T_0_ = T_0(v, a, T)
    T_0_.save(f'..\\..\\data\\T_0_D_{D}_sim_n_thr_{n_thr_crossings}.pickle')

    __x, __y = np.meshgrid(v, a)

    plt.figure()
    plt.contourf(__x, __y, T.transpose(), levels=20) #, v_min=-30, v_max=30)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
