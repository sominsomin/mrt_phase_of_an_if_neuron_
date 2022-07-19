import matplotlib.pyplot as plt
import numpy as np
import math

from mrt_phase_pde.pde.own_method.src.T_0.T_0 import T_0
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


D = 0.0

diff_mean = []
diff_max = []
diff_min = []
diff_std = []

for i in range(0, 6):
    T_1 = f'..\\data\\T_{i}_D_{D}_sim_n_thr_{i+1}.pickle'
    T_2 = f'..\\result\\T_N_{i}_D_{D}.pickle'

    t_1 = T_N.load(T_1)
    t_2 = T_0.load(T_2)

    # max_n_a = 31
    a = np.linspace(0, 3, 31)
    v = np.linspace(-1, 1, 21)

    idx_a_t_1 = [find_nearest(t_1.a, _a) for _a in a]
    idx_a_t_2 = [find_nearest(t_2.a, _a) for _a in a]
    idx_v_t_1 = [find_nearest(t_1.v, _v) for _v in v]
    idx_v_t_2 = [find_nearest(t_2.v, _v) for _v in v]

    diff = t_2.T[np.ix_(idx_v_t_2, idx_a_t_2)] - t_1.T[np.ix_(idx_v_t_1, idx_a_t_1)]

    diff_mean.append(np.mean(diff))
    diff_min.append(np.min(diff))
    diff_max.append(np.max(diff))
    diff_std.append(np.std(diff))

    # __x, __y = np.meshgrid(t_2.v[idx_v_t_2], t_2.a[idx_a_t_2])
    #
    # plt.contourf(__x, __y, diff.transpose())
    # plt.colorbar()

    # plt.savefig(f'img\\compare_T_{n_1}_with_T_{n_2}_D_{D}.png')


n_vec = list(range(1, len(diff_mean) + 1))

plt.plot(n_vec, diff_mean, label='mean(d)')
plt.plot(n_vec, np.array(diff_mean) - np.array(diff_max), label='min(d)')
plt.plot(n_vec, diff_max, label='max(d)')
plt.plot(n_vec, np.array(diff_mean) + np.array(diff_std), '--', label='mean(d) + std(d)')
plt.plot(n_vec, np.array(diff_mean) - np.array(diff_std), '--', label='mean(d) - std(d)')
plt.xlabel('N')
plt.ylabel('$T_N(v,a) - T_{N,sim}(v,a)$')
plt.ylim([-0.1, 0.1])
plt.legend()
plt.title(f'$D = {D}$,\n mean, max , min and std (Standard Deviation)\n of $d = T_N(v,a) - T_{{N,sim}}(v,a)$')
plt.tight_layout()

plt.savefig(f'..\\img\\T_N_vs_T_sim_over_time_D_{D}.png')
