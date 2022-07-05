import matplotlib.pyplot as plt
import numpy as np

from mrt_phase_pde.pde.own_method.src.T_0.T_0 import T_0
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N

n = list(range(1, 6))


def get_diff(D):
    diff_mean = []
    diff_max = []
    diff_min = []

    for i in n:
        n_1 = i
        n_2 = i-1

        T_1 = f'..\\result\\T_N_{n_1}_D_{D}.pickle'
        T_2 = f'..\\result\\T_N_{n_2}_D_{D}.pickle'

        t_1 = T_N.load(T_1)
        t_2 = T_0.load(T_2)

        max_a = 3
        max_n_a = np.where(t_1.a == max_a)[0][0] + 1

        diff = t_1.T[:, :max_n_a] - t_2.T[:, :max_n_a]

        diff_mean.append(np.mean(diff))
        diff_max.append(np.max(diff))
        diff_min.append(np.min(diff))

    return diff_mean, diff_min, diff_max


def plot_T_N_over_time(D):
    diff_mean, diff_min, diff_max = get_diff(D)

    plt.figure()
    plt.plot(n, diff_mean, label='mean($T_N(v,a) - T_{N-1}(v,a)$)')
    plt.plot(n, diff_max, label='max($T_N(v,a) - T_{N-1}(v,a)$)')
    plt.plot(n, diff_min, label='min($T_N(v,a) - T_{N-1}(v,a)$)')
    plt.xlabel('N')
    plt.ylabel('$T_N(v,a) - T_{{N-1}}(v,a)$')
    plt.title(f'$D = {D}$')
    plt.legend()

    plt.savefig(f'..\\img\\T_N_diff_over_N_D_{D}.png')
    # plt.show()

    print(f'D = {D}: {diff_mean[-1]}')


D_list = [0.0, 0.1, 0.25, 0.5, 1.0]


if __name__=='__main__':
    for D in D_list:
        plot_T_N_over_time(D)
